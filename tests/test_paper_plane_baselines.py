"""CPU-only audits, job planning and retry tests; no GPU training."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
import paper_plane_baselines as driver


def regular_settings():
    return {'background_plane_type': driver.SPHERE_TYPE,
            'background_plane_kwargs': {'color_mask_resolution': [960, 1920], 'network_flow_config': 'large'},
            'name': 'reference', 'seed': 42, 'max_epochs': 80, 'deterministic': False,
            'checkpoint_path': None, 'checkpoint_tracker_path': None,
            'save_intermediate_checkpoints': False, 'background_ellipsoid_path': None,
            'background_ellipsoid_scale_margin': .02, 'background_ellipsoid_texture_orientation': 'legacy',
            'background_sphere_radius_override': None, 'pinhole_camera_config': {'focal_length': [500, 500]}}


def yaml_family():
    regular = regular_settings()
    plane = copy.deepcopy(regular)
    plane.update(background_plane_type=driver.PLANE_TYPE, name='plane')
    visible = copy.deepcopy(regular)
    visible['name'] = 'visuv'
    visible['background_plane_kwargs']['sphere_visible_uv_texture'] = copy.deepcopy(driver.UV_SETTINGS)
    return regular, visible, plane


def references():
    minutes = [27.25, 52.4, 18.1, 23.87, 34.61, 36.34, 55.63]
    result = {}
    for label, duration in zip(driver.LABELS, minutes):
        _, first, last = label.split('_')
        settings = regular_settings()
        settings['background_sphere_radius_from_visible_plane_centers'] = True
        result[label] = {'config': 'plane.yaml', 'config_sha256': 'hash',
                         'frame_count': int(last) - int(first) + 1,
                         'reference_settings': settings, 'seconds': duration * 60}
    return result


class PlaneBaselineTests(unittest.TestCase):
    def test_21_unique_fresh_jobs_and_conservative_sub_ten_hour_batches(self):
        jobs = []
        for batch, size in [('part1of2', 12), ('part2of2', 9)]:
            group = driver.make_jobs(batch, references())
            self.assertEqual(len(group), size)
            self.assertLess(sum(j['buffered_seconds'] for j in group), 10 * 3600)
            jobs += group
        self.assertEqual(len(jobs), 21)
        self.assertEqual(len({j['id'] for j in jobs}), 21)
        self.assertEqual({j['repeat'] for j in jobs}, {1, 2, 3})
        self.assertTrue(all(j['seed'] == 42 for j in jobs))
        for label in driver.LABELS:
            self.assertEqual(sum(j['label'] == label for j in jobs), 3)

    def test_plane_command_has_no_curved_geometry_or_resume_options(self):
        cmd = driver.command(driver.make_jobs('part1of2', references())[0], Path('attempt_001'))
        self.assertEqual(cmd[cmd.index('--background-plane-type') + 1], driver.PLANE_TYPE)
        self.assertEqual(cmd[cmd.index('--seed') + 1], '42')
        self.assertFalse(any('sphere' in arg or 'ellipsoid' in arg or 'checkpoint' in arg for arg in cmd))

    def test_yaml_audit_accepts_only_intended_changes(self):
        regular, visible, plane = yaml_family()
        report = driver.audit_yaml_family(regular, visible, plane)
        self.assertEqual(report['plane_vs_regular_differences'], ['/background_plane_type', '/name'])
        self.assertEqual(visible['background_plane_kwargs']['sphere_visible_uv_texture'], driver.UV_SETTINGS)

    def test_yaml_audit_rejects_foreground_camera_loss_or_schedule_changes(self):
        for key, value in [('pinhole_camera_config', {'focal_length': [501, 500]}),
                           ('losses', ['different']), ('max_epochs', 81), ('has_camera_aberration_plane', True)]:
            regular, visible, plane = yaml_family()
            plane[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.audit_yaml_family(regular, visible, plane)

    def test_yaml_audit_rejects_doubled_resolution_or_unapproved_uv(self):
        for treatment in ['plane_resolution', 'visible_resolution', 'visible_margin', 'plane_uv']:
            regular, visible, plane = yaml_family()
            if treatment == 'plane_resolution':
                plane['background_plane_kwargs']['color_mask_resolution'] = [1920, 3840]
            elif treatment == 'visible_resolution':
                visible['background_plane_kwargs']['color_mask_resolution'] = [1920, 3840]
            elif treatment == 'visible_margin':
                visible['background_plane_kwargs']['sphere_visible_uv_texture']['margin_uv'] = [.1, .1]
            else:
                plane['background_plane_kwargs']['sphere_visible_uv_texture'] = {'enabled': True}
            with self.subTest(treatment=treatment), self.assertRaises(ValueError):
                driver.audit_yaml_family(regular, visible, plane)

    def test_effective_settings_reject_resume_checkpoint_and_geometry_overrides(self):
        reference = regular_settings()
        reference['background_sphere_radius_from_visible_plane_centers'] = True
        plane = copy.deepcopy(reference)
        plane.update(background_plane_type=driver.PLANE_TYPE, background_sphere_radius_from_visible_plane_centers=False)
        driver.audit_effective_settings(plane, reference, 'plane')
        for key, value in [('checkpoint_path', 'resume.ckpt'), ('save_intermediate_checkpoints', True),
                           ('background_ellipsoid_path', 'geometry.json'), ('background_sphere_radius_override', 30.),
                           ('seed', 43), ('deterministic', True)]:
            wrong = {**plane, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.audit_effective_settings(wrong, reference, 'plane')

    def test_failed_exit_is_not_counted_as_complete(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            root = Path(directory)
            (root / 'exit_000.txt').write_text('Exit code: 1.')
            with patch.object(driver, 'completed', return_value={'MPSNR': 1}):
                self.assertIsNone(driver.basic_completed(root, {'frame_count': 1}))

    def test_effective_foreground_or_network_differences_are_not_ignored(self):
        reference = regular_settings()
        plane = copy.deepcopy(reference)
        plane['background_plane_type'] = driver.PLANE_TYPE
        plane['background_plane_kwargs']['network_flow_config'] = 'small'
        with self.assertRaises(ValueError):
            driver.audit_effective_settings(plane, reference, 'plane')

    def test_nonempty_unowned_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            output = Path(directory) / 'autowork_existing'
            output.mkdir()
            (output / 'user.txt').write_text('preserve')
            with patch.object(sys, 'argv', ['driver', '--batch', 'part1of2', '--output-root', str(output)]), \
                    patch.object(driver, 'load_audit', return_value=(references(), {})), \
                    patch.object(driver, 'code_snapshot', return_value={}), \
                    patch.object(driver, 'LOCK_ROOT', Path(directory) / 'autowork_lock'), \
                    patch.object(driver.subprocess, 'run') as run, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                driver.main()
            run.assert_not_called()
            self.assertEqual((output / 'user.txt').read_text(), 'preserve')
            self.assertFalse((output / 'manifest.json').exists())

    def test_audit_and_dry_run_write_nothing_and_never_train(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            base = Path(directory)
            for options in [['--audit-only'], ['--batch', 'part1of2', '--dry-run']]:
                output = base / 'autowork_output'
                with patch.object(sys, 'argv', ['driver', '--output-root', str(output), *options]), \
                        patch.object(driver, 'load_audit', return_value=(references(), {})), \
                        patch.object(driver, 'code_snapshot', return_value={}), \
                        patch.object(driver, 'LOCK_ROOT', base / 'autowork_lock'), \
                        patch.object(driver.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                    driver.main()
                self.assertFalse(output.exists())
                self.assertFalse((base / 'autowork_lock').exists())
                run.assert_not_called()

    def test_retry_preserves_partial_attempt_and_skips_all_completed_jobs(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            base = Path(directory)
            output = base / 'autowork_output'
            calls = []
            ref = references()
            jobs = driver.make_jobs('part2of2', ref)
            by_id = {j['id']: j for j in jobs}

            def train(cmd, **kwargs):
                path = Path(cmd[cmd.index('--output-path') + 1])
                job = by_id[path.parent.name]
                calls.append(path)
                path.mkdir(parents=True)
                if len(calls) == 1:
                    (path / 'partial.txt').touch()
                    raise subprocess.CalledProcessError(1, cmd)
                settings = copy.deepcopy(job['reference_settings'])
                settings.update(background_plane_type=driver.PLANE_TYPE,
                                background_sphere_radius_from_visible_plane_centers=False, name=job['id'])
                (path / 'training_config.yaml').write_text(yaml.safe_dump({'NAGConfig': settings}))
                (path / 'exit_000.txt').write_text('Exit code: 0.')
                (path / 'final/complete').mkdir(parents=True)
                for frame in range(job['frame_count']):
                    (path / 'final/complete' / f'{frame}_t.png').touch()
                (path / 'tracker').mkdir()
                for metric in driver.METRICS:
                    (path / 'tracker' / f'train_epoch_Mean{metric}.csv').write_text('step;value\n80;1\n')

            with patch.object(sys, 'argv', ['driver', '--batch', 'part2of2', '--output-root', str(output)]), \
                    patch.object(driver, 'load_audit', return_value=(ref, {'code': {}, 'source_files': {}})), \
                    patch.object(driver, 'code_snapshot', return_value={}), \
                    patch.object(driver, 'LOCK_ROOT', base / 'autowork_lock'), \
                    patch.object(driver, 'assert_signature'), patch.object(driver.subprocess, 'run', side_effect=train), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    driver.main()
                driver.main()
                driver.main()
            self.assertEqual(len(calls), 10)  # one failed, nine successful
            self.assertEqual(calls[0].name, 'attempt_001')
            self.assertEqual(calls[1].name, 'attempt_002')
            self.assertTrue((calls[0] / 'partial.txt').is_file())

    def test_small_invocation_budget_stops_without_starting_training(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            output = Path(directory) / 'autowork_output'
            with patch.object(sys, 'argv', ['driver', '--batch', 'part1of2', '--output-root', str(output), '--max-hours', '.001']), \
                    patch.object(driver, 'load_audit', return_value=(references(), {})), \
                    patch.object(driver, 'code_snapshot', return_value={}), \
                    patch.object(driver, 'LOCK_ROOT', Path(directory) / 'autowork_lock'), \
                    patch.object(driver.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            run.assert_not_called()
            self.assertTrue((output / 'manifest.json').is_file())
            self.assertEqual(list(output.glob('*/attempt_*')), [])

    def test_summary_does_not_reaudit_inputs_or_train(self):
        with tempfile.TemporaryDirectory(prefix='autowork_plane_test_') as directory:
            output = Path(directory)
            (output / 'manifest.json').write_text(json.dumps({'jobs': []}))
            with patch.object(sys, 'argv', ['driver', '--batch', 'part1of2', '--output-root', str(output), '--summary']), \
                    patch.object(driver, 'load_audit') as audit, patch.object(driver.subprocess, 'run') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            audit.assert_not_called()
            run.assert_not_called()

    def test_source_code_drift_is_rejected(self):
        with patch.object(driver, 'code_snapshot', return_value={'model': 'changed'}), self.assertRaises(ValueError):
            driver.assert_signature({'code': {'model': 'original'}, 'source_files': {}})


if __name__ == '__main__':
    unittest.main()
