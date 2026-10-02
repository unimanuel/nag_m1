"""CPU-only planning and retry checks; no real trainings are launched."""
import contextlib
import hashlib
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
import paper_sphere_baselines as driver


def references():
    minutes = {
        'W191_0_40': (18.07, 19.09), 'W191_128_198': (17.81, 18.29),
        'W254_0_37': (16.60, 16.56), 'W254_86_137': (25.19, 25.35),
        'W191_40_128': (36.24, 36.70), 'W254_37_81': (34.83, 35.),
        'W254_139_195': (34.12, 34.99),
    }
    return {(label, mode): {'frame_count': int(label.split('_')[2])-int(label.split('_')[1])+1,
                           'config': 'config.yaml', 'config_sha256': 'config-hash',
                           'path': 'reference', 'geometry': 'geometry.txt',
                           'geometry_sha256': 'geometry-hash', 'seconds': duration * 60}
            for label, durations in minutes.items() for mode, duration in zip(driver.MODES, durations)}


class SphereBaselineTests(unittest.TestCase):
    def test_complete_42_job_coverage_and_buffered_batch_times(self):
        jobs = []
        for index, batch in enumerate(driver.BATCHES):
            group = driver.make_jobs(batch, references())
            self.assertEqual(len(group), 12 if index < 2 else 6)
            self.assertLess(sum(j['buffered_seconds'] for j in group), 6 * 3600)
            self.assertTrue(all(j['seed'] == 42 for j in group))
            for label in driver.BATCHES[batch]:
                for mode in driver.MODES:
                    self.assertEqual(sum((j['label'], j['mode']) == (label, mode) for j in group), 3)
            jobs.extend(group)
        self.assertEqual(len(jobs), 42)
        self.assertEqual(len({j['id'] for j in jobs}), 42)
        self.assertEqual({j['repeat'] for j in jobs}, {1, 2, 3})
        group = driver.make_jobs('part3of5', references())
        self.assertEqual([j['mode'] for j in group], ['regular', 'visuv', 'visuv', 'regular', 'regular', 'visuv'])

    def test_command_preserves_radius_rule_and_contains_no_ellipsoid_options(self):
        job = driver.make_jobs('part3of5', references())[0]
        cmd = driver.command(job, Path('output'))
        self.assertIn('--background-sphere-radius-from-visible-plane-centers', cmd)
        self.assertEqual(cmd[cmd.index('--seed') + 1], '42')
        self.assertEqual(cmd[cmd.index('--background-plane-type') + 1], driver.SPHERE_TYPE)
        self.assertFalse(any('ellipsoid' in item for item in cmd))

    def test_duration_uses_start_end_and_rejects_nonpositive(self):
        with tempfile.TemporaryDirectory(prefix='autowork_sphere_test_') as directory:
            p = Path(directory) / 'exit.txt'
            p.write_text('Start-Date: 2026-10-02 10:00:00+02:00\nEnd-Date: 2026-10-02 10:30:00+02:00\n')
            self.assertEqual(driver.duration_seconds(p), 1800)
            p.write_text('Start-Date: 2026-10-02 10:00:00+02:00\nEnd-Date: 2026-10-02 09:30:00+02:00\n')
            with self.assertRaises(ValueError):
                driver.duration_seconds(p)

    def test_scheduling_budget_does_not_start_a_job_that_would_exceed_limit(self):
        job = {'buffered_seconds': 2000}
        self.assertTrue(driver.budget_allows(19000, job, 21600))
        self.assertFalse(driver.budget_allows(20000, job, 21600))

    def test_config_rejects_uv_mismatch_and_changed_paired_yaml(self):
        with tempfile.TemporaryDirectory(prefix='autowork_sphere_test_') as directory:
            p = Path(directory) / 'config.yaml'
            p.write_text('example')
            settings = {'frame_indices_filter': {'start': 0, 'stop': 41}, 'max_epochs': 80,
                        'background_plane_type': driver.SPHERE_TYPE, 'background_plane_kwargs': {}}
            driver.validate_config(settings, 'W191_0_40', 'regular', driver.sha(p), p)
            with self.assertRaises(ValueError):
                driver.validate_config(settings, 'W191_0_40', 'visuv', driver.sha(p), p)
            with self.assertRaises(ValueError):
                driver.validate_config(settings, 'W191_0_40', 'regular', 'changed', p)
            settings['save_intermediate_checkpoints'] = True
            with self.assertRaises(ValueError):
                driver.validate_config(settings, 'W191_0_40', 'regular', driver.sha(p), p)

    def test_training_compatibility_rejects_model_changes(self):
        with self.assertRaises(ValueError), patch.object(driver.subprocess, 'check_output') as git:
            driver.check_training_compatibility({'nag/model/model.py': 'old'}, {'nag/model/model.py': 'new'})
        git.assert_not_called()
        result = driver.check_training_compatibility({'nag/sampling_floor_ellipsoid.py': 'old'},
                                                     {'nag/sampling_floor_ellipsoid.py': 'new'})
        self.assertEqual(result['unused_fitter_changes_excluded'], ['nag/sampling_floor_ellipsoid.py'])

    def test_only_audited_checkpoint_change_is_accepted(self):
        old, new = b'old policy', b'new policy'
        path = 'nag/callbacks/nag_callback.py'
        with patch.object(driver.subprocess, 'check_output', side_effect=[new, old]), \
                patch.object(Path, 'read_bytes', return_value=new):
            driver.check_training_compatibility({path: hashlib.sha256(old).hexdigest()},
                                                {path: hashlib.sha256(new).hexdigest()})

    def test_audited_mixed_newlines_reconstruct_exact_historical_hash(self):
        path = 'nag/callbacks/nag_callback.py'
        definition = b'    def save_model(self, trainer: pl.Trainer, model: NAGModel, prefix: str = ""):'
        directory = b'        checkpoint_dir = os.path.join(model.config.output_path, "checkpoints")'
        insertion = (b'        if prefix != "final-" and not getattr(model.config, "save_intermediate_checkpoints", False):\n'
                     b'            logging.debug("Skipping intermediate checkpoint; only the final checkpoint is enabled.")\n'
                     b'            return\n')
        old = b'unchanged\r\n' + definition + b'\r\n' + directory + b'\r\n'
        new = b'unchanged\r\n' + definition + b'\n' + insertion + directory + b'\n'
        old_hash = hashlib.sha256(old).hexdigest()
        with patch.object(driver.subprocess, 'check_output',
                          side_effect=[new.replace(b'\r\n', b'\n'), old.replace(b'\r\n', b'\n')]), \
                patch.object(Path, 'read_bytes', return_value=new):
            driver.check_training_compatibility({path: old_hash}, {path: hashlib.sha256(new).hexdigest()})
        modified = new.replace(b'unchanged', b'changed')
        with patch.object(driver.subprocess, 'check_output',
                          side_effect=[modified.replace(b'\r\n', b'\n'), old.replace(b'\r\n', b'\n')]), \
                patch.object(Path, 'read_bytes', return_value=modified), self.assertRaises(ValueError):
            driver.check_training_compatibility({path: old_hash}, {path: hashlib.sha256(modified).hexdigest()})

    def test_unreviewed_callback_change_is_rejected(self):
        path = 'nag/callbacks/nag_callback.py'
        with patch.object(driver.subprocess, 'check_output', side_effect=[b'new', b'old']), \
                patch.object(Path, 'read_bytes', return_value=b'new'):
            with self.assertRaises(ValueError):
                driver.check_training_compatibility({path: 'old-hash'}, {path: 'changed-hash'})

    def test_completed_sphere_checks_radius_and_uv_mode(self):
        with tempfile.TemporaryDirectory(prefix='autowork_sphere_test_') as directory:
            root = Path(directory)
            (root / 'initial/background_diagnostics').mkdir(parents=True)
            (root / 'exit_000.txt').write_text('Exit code: 0.')
            (root / 'training_config.yaml').write_text(yaml.safe_dump({'NAGConfig': {
                'seed': 42, 'max_epochs': 80, 'checkpoint_path': None,
                'save_intermediate_checkpoints': False}}))
            reference = root / 'reference.txt'
            actual = root / 'initial/background_diagnostics/sphere_geometry.txt'
            content = ('radius_source: visible_plane_centers\nradius: 30.0\ncenter: [0, 0, 0]\n'
                       'color_mask_resolution: [960, 1920]\nvisible_uv_texture: {"enabled": False}\n')
            reference.write_text(content)
            actual.write_text(content)
            job = {'frame_count': 2, 'mode': 'regular', 'reference_geometry': str(reference)}
            with patch.object(driver, 'completed', return_value={'MPSNR': 1}):
                self.assertEqual(driver.completed_sphere(root, job), {'MPSNR': 1})
                actual.write_text(content.replace('radius: 30.0', 'radius: 31.0'))
                with self.assertRaises(ValueError):
                    driver.completed_sphere(root, job)
                actual.write_text(content)
                with self.assertRaises(ValueError):
                    driver.completed_sphere(root, {**job, 'mode': 'visuv'})

    def test_fresh_attempts_skip_completed_jobs_and_preserve_failed_attempt(self):
        with tempfile.TemporaryDirectory(prefix='autowork_sphere_test_') as directory:
            base = Path(directory)
            inventory = base / 'sources.json'
            inventory.write_text('{}')
            root = base / 'autowork_results'
            ref = references()
            labels = driver.BATCHES['part3of5']
            calls = []
            failed = [False]

            def fake_training(cmd, **kwargs):
                output = Path(cmd[cmd.index('--output-path') + 1])
                calls.append(output)
                output.mkdir(parents=True)
                if not failed[0]:
                    failed[0] = True
                    (output / 'partial.txt').touch()
                    raise subprocess.CalledProcessError(1, cmd)
                (output / 'exit_000.txt').write_text('Exit code: 0.')
                (output / 'final/complete').mkdir(parents=True)
                for index in range(ref[labels[0], 'regular']['frame_count']):
                    (output / 'final/complete' / f'{index}_t.png').touch()
                (output / 'tracker').mkdir()
                for metric in ['MPSNR', 'SSIM', 'LPIPS']:
                    (output / 'tracker' / f'train_epoch_Mean{metric}.csv').write_text('step;value\n80;1\n')

            argv = ['driver', '--batch', 'part3of5', '--output-root', str(root)]
            with patch.object(sys, 'argv', argv), patch.object(driver, 'REFERENCES', inventory), \
                    patch.object(driver, 'LOCK_ROOT', base / 'autowork_lock'), \
                    patch.object(driver, 'code_snapshot', return_value={}), \
                    patch.object(driver, 'load_references', return_value=(ref, [], [])), \
                    patch.object(driver, 'assert_signature'), \
                    patch.object(driver, 'completed_sphere', side_effect=lambda p, j: driver.completed(p, j['frame_count'])), \
                    patch.object(driver.subprocess, 'run', side_effect=fake_training), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    driver.main()
                driver.main()
                driver.main()
            self.assertEqual(len(calls), 7)  # one failed attempt plus six successful jobs
            self.assertEqual(calls[0].name, 'attempt_001')
            self.assertEqual(calls[1].name, 'attempt_002')
            self.assertTrue((calls[0] / 'partial.txt').is_file())

    def test_summary_never_trains_or_revalidates_current_inputs(self):
        with tempfile.TemporaryDirectory(prefix='autowork_sphere_test_') as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'jobs': []}))
            with patch.object(sys, 'argv', ['driver', '--batch', 'part1of5', '--summary', '--output-root', str(root)]), \
                    patch.object(driver.subprocess, 'run') as run, \
                    patch.object(driver, 'load_references') as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            run.assert_not_called()
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
