"""CPU-only checks for the non-ego paper cohort; never launch GPU training."""
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

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
import paper_final_baselines as driver


class FinalBaselineTests(unittest.TestCase):
    def test_training_budget_defaults_to_twelve_hours_and_accepts_override(self):
        class Parsed(Exception):
            pass
        original = driver.argparse.ArgumentParser.parse_args
        for flags, expected in [((), 12.), (('--max-hours', '6.5'), 6.5)]:
            def check(parser):
                args = original(parser)
                self.assertEqual(args.max_hours, expected)
                raise Parsed
            with patch.object(driver.argparse.ArgumentParser, 'parse_args', autospec=True, side_effect=check), \
                    patch.object(sys, 'argv', ['driver', '--segment', 'W125', '--dry-run', *flags]), \
                    self.assertRaises(Parsed):
                driver.main()

    def test_all_75_real_yamls_match_ego_template_and_treatments(self):
        items = driver.intervals()
        self.assertEqual(len(items), 25)
        driver.yaml_audit(items)
        jobs = driver.make_jobs(items, {k: [] for k in items})
        self.assertEqual(len(jobs), 225)
        self.assertEqual(len({j['id'] for j in jobs}), 225)
        self.assertEqual({j['repeat'] for j in jobs}, {1, 2, 3})
        self.assertTrue(all(j['seed'] == 42 for j in jobs))
        self.assertEqual({j['treatment'] for j in jobs}, set(driver.TREATMENTS))

    def test_archival_settings_are_not_carried_into_new_expected_configs(self):
        label, item = next(iter(driver.intervals().items()))
        for treatment in driver.TREATMENTS:
            c = driver.expected_config(item, label, treatment)
            self.assertEqual(c['sampler_kwargs'], {'num_batches': 140, 'num_rays': 100000, 'num_timestamps': 20})
            self.assertEqual(c['num_workers'], 10)
            self.assertFalse(c['save_intermediate_checkpoints'])
            self.assertEqual(c['background_plane_kwargs']['color_mask_resolution'], [960, 1920])

    def test_changed_network_or_foreground_settings_are_rejected(self):
        label, item = next(iter(driver.intervals().items()))
        original = driver.config
        def changed(path):
            c = original(path)
            if path.parent == driver.CONFIG_ROOT:
                c['plane_flow_weight'] = .2
            return c
        with patch.object(driver, 'config', side_effect=changed), self.assertRaises(ValueError):
            driver.yaml_audit({label: item})

    def test_commands_have_correct_policy_and_no_ellipsoid_or_resume(self):
        items = dict(list(driver.intervals().items())[:1])
        for job in driver.make_jobs(items, {k: [] for k in items}):
            cmd = driver.command(job, Path('attempt_001'))
            self.assertEqual('--background-sphere-radius-from-visible-plane-centers' in cmd,
                             job['treatment'] != 'plane')
            self.assertFalse(any('ellipsoid' in x or 'checkpoint' in x for x in cmd))

    def test_environment_cannot_redirect_training_away_from_audited_inputs(self):
        with patch.dict(driver.os.environ, {'DATA_PATH': '/tmp/autowork_unrelated_dataset'}), \
                patch.object(driver, 'code_snapshot') as snapshot, self.assertRaises(ValueError):
            driver.training_context()
        snapshot.assert_not_called()

    def test_ready_assets_boxes_are_object_level_and_filter_is_optional(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            root = Path(directory) / 'segment-123'
            for folder in ['images_undistorted', 'depth', 'masks', 'boxes']:
                (root / folder).mkdir(parents=True, exist_ok=True)
            for n in range(2):
                (root / 'images_undistorted' / f'{n:03d}.png').write_bytes(b'image')
                (root / 'depth' / f'{n}.tiff').write_bytes(b'depth')
                (root / 'masks' / f'img_{n:03d}_ov_0.png').write_bytes(b'mask')
            (root / 'boxes/77.json').write_text('{}')
            (root / 'box_id_mapping.json').write_text('{}')
            poses = np.repeat(np.eye(4)[None], 2, axis=0).tolist()
            (root / 'waymo_pose_config_no_distortion.json').write_text(json.dumps({'position': poses}))
            item = {'segment': '123', 'first': 0, 'stop': 2}
            result = driver.data_audit(item, Path(directory))
            self.assertTrue(result['ready'])
            self.assertFalse(result['optional_mask_filter_present'])
            self.assertEqual(result['camera_path_length_m'], 0)
            (root / 'depth/1.tiff').unlink()
            result = driver.data_audit(item, Path(directory))
            self.assertFalse(result['ready'])
            self.assertEqual(result['missing'], {'depth': [1]})

    def test_missing_depth_blocks_training_instead_of_changing_frame_selection(self):
        with patch.object(driver, 'intervals', return_value={'W125_0_1': {
                    'segment':'123','stem':'example','first':0,'stop':2}}), \
                patch.object(driver, 'yaml_audit'), \
                patch.object(driver, 'data_audit', return_value={'ready': False, 'missing': {'depth': [1]},
                                                              'camera_path_length_m': 0}), \
                patch.object(driver, 'historical_runs', return_value={}), \
                patch.object(driver.subprocess, 'run') as run, \
                patch.object(sys, 'argv', ['driver', '--segment', 'W125']), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            driver.main()
        run.assert_not_called()

    def test_historical_timing_excludes_doubled_resolution_and_incomplete_runs(self):
        items = {'W125_0_1': {'first': 0, 'stop': 2, 'segment': '123', 'stem': 'example'}}
        rows = [{'complete': True, 'seconds': 1000, 'texture_resolution': [960, 1920], 'class': driver.SPHERE_TYPE},
                {'complete': True, 'seconds': 99999, 'texture_resolution': [1920, 3840], 'class': driver.SPHERE_TYPE},
                {'complete': False, 'seconds': 99999, 'texture_resolution': [960, 1920], 'class': driver.SPHERE_TYPE}]
        jobs = driver.make_jobs(items, {'W125_0_1': rows})
        self.assertTrue(all(j['historical_seconds'] == 1000 for j in jobs))

    def test_dry_run_and_audit_write_nothing_and_never_train(self):
        for flag in ['--dry-run', '--audit-only']:
            with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
                output = Path(directory) / 'autowork_output'
                items = {'W125_0_1': {'first': 0, 'stop': 2, 'segment': '123', 'stem': 'example'}}
                ready = {'ready': True, 'missing': {}, 'camera_path_length_m': 0}
                with patch.object(driver, 'intervals', return_value=items), patch.object(driver, 'yaml_audit'), \
                        patch.object(driver, 'data_audit', return_value=ready), \
                        patch.object(driver, 'historical_runs', return_value={'W125_0_1': []}), \
                        patch.object(driver, 'training_context', return_value={}), \
                        patch.object(driver, 'assert_job_worker_policy'), \
                        patch.object(driver.subprocess, 'run') as run, \
                        patch.object(sys, 'argv', ['driver', '--segment', 'W125', '--output-root', str(output), flag]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    driver.main()
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_nonempty_unowned_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            output = Path(directory) / 'autowork_output'
            output.mkdir()
            (output / 'user.txt').write_text('preserve')
            with patch.object(driver, 'intervals', return_value={'W125_0_1': {
                        'segment':'123','stem':'example','first':0,'stop':2}}), \
                    patch.object(driver, 'yaml_audit'), \
                    patch.object(driver, 'data_audit', return_value={'ready': True, 'missing': {}, 'camera_path_length_m': 0}), \
                    patch.object(driver, 'historical_runs', return_value={}), \
                    patch.object(driver, 'training_context', return_value={}), \
                    patch.object(driver.subprocess, 'run') as run, \
                    patch.object(sys, 'argv', ['driver', '--segment', 'W125', '--output-root', str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                driver.main()
            run.assert_not_called()
            self.assertEqual((output / 'user.txt').read_text(), 'preserve')
            self.assertFalse((output / 'manifest.json').exists())

    def test_real_saved_ego_serialization_is_accepted_only_when_settings_match(self):
        # Validate serialization rules against an existing successful training;
        # no new forward pass, rendering, checkpoint load or training is run.
        root = driver.REPO / 'runs/autowork_paper_plane_baselines_part1of2'
        if not (root / 'manifest.json').exists():
            self.skipTest('Completed cohort not available in this checkout')
        job = json.loads((root / 'manifest.json').read_text())['jobs'][0]
        source = next((root / job['id']).glob('attempt_*'))
        reference = driver.config(job['config'])
        _, first, last = job['label'].split('_')
        reference['save_intermediate_checkpoints'] = False
        new_job = {**job, 'treatment': 'plane', 'segment': '1918764220984209654',
                   'first': int(first), 'stop': int(last) + 1}
        original = driver.config
        def settings(path):
            return copy.deepcopy(reference) if str(path) == job['config'] else original(path)
        with patch.object(driver, 'config', side_effect=settings):
            self.assertIsNotNone(driver.completed_job(source, new_job))
        reference['sampler_kwargs']['num_rays'] = 70000
        with patch.object(driver, 'config', side_effect=settings), self.assertRaises(ValueError):
            driver.completed_job(source, new_job)

    def exercise_batch(self, base, jobs, flags=()):
        root = base / 'runs/autowork_test'
        items = {'W125_0_1': {'first': 0, 'stop': 2, 'segment': '123', 'stem': 'example'}}
        ready = {'ready': True, 'missing': {}, 'camera_path_length_m': 0}
        def successful(path, job):
            return dict.fromkeys(driver.METRICS, 1.) if (path / 'done').exists() else None
        def training(cmd, **kwargs):
            output = Path(cmd[cmd.index('--output-path') + 1])
            output.mkdir(parents=True)
            (output / 'done').write_text('successful fake training')
        with patch.object(driver, 'REPO', base), \
                patch.object(driver, 'intervals', return_value=items), patch.object(driver, 'yaml_audit'), \
                patch.object(driver, 'data_audit', return_value=ready), \
                patch.object(driver, 'historical_runs', return_value={'W125_0_1': []}), \
                patch.object(driver, 'training_context', return_value={}), \
                patch.object(driver, 'assert_context'), patch.object(driver, 'make_jobs', return_value=jobs), \
                patch.object(driver, 'assert_job_worker_policy'), \
                patch.object(driver, 'completed_job', side_effect=successful), \
                patch.object(driver.subprocess, 'run', side_effect=training) as run, \
                patch.object(sys, 'argv', ['driver', '--segment', 'W125', '--output-root', str(root), *flags]), \
                contextlib.redirect_stdout(io.StringIO()):
            driver.main()
        return root, run

    def test_retries_preserve_failed_attempts_and_skip_finished_jobs(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            base = Path(directory)
            job = {'id': 'autowork_fake', 'label': 'W125_0_1', 'treatment': 'plane',
                   'config': 'fake.yaml', 'buffered_seconds': 1}
            root, first = self.exercise_batch(base, [job])
            self.assertEqual(first.call_count, 1)
            _, second = self.exercise_batch(base, [job])
            second.assert_not_called()
            # Retain the first attempt as failed, simulating interruption after a
            # partial output; the next invocation must create another directory.
            output = root / job['id'] / 'attempt_001'
            (output / 'done').rename(output / 'partial')
            _, retry = self.exercise_batch(base, [job])
            self.assertEqual(retry.call_count, 1)
            self.assertTrue((output / 'partial').exists())
            self.assertTrue((root / job['id'] / 'attempt_002/done').exists())

    def test_estimated_over_budget_job_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            base = Path(directory)
            jobs = [{'id': f'autowork_fake_{n}', 'label': 'W125_0_1', 'treatment': 'plane',
                     'config': 'fake.yaml', 'buffered_seconds': 13 * 3600} for n in range(2)]
            _, stopped = self.exercise_batch(base, jobs)
            stopped.assert_not_called()
            root, allowed = self.exercise_batch(base, jobs, ['--allow-long-job'])
            self.assertEqual(allowed.call_count, 1)
            self.assertFalse((root / jobs[1]['id']).exists())

    def test_default_budget_admits_eleven_hour_estimate_without_opt_in(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            job = {'id': 'autowork_fake', 'label': 'W125_0_1', 'treatment': 'plane',
                   'config': 'fake.yaml', 'buffered_seconds': 11 * 3600}
            _, launched = self.exercise_batch(Path(directory), [job])
            self.assertEqual(launched.call_count, 1)

    def test_scheduler_learns_duration_only_from_completed_same_treatment(self):
        with tempfile.TemporaryDirectory(prefix='autowork_final_test_') as directory:
            root = Path(directory)
            job = {'id': 'autowork_example', 'label': 'W125_0_1', 'treatment': 'plane', 'buffered_seconds': 9999}
            path = root / job['id'] / 'attempt_001'
            path.mkdir(parents=True)
            (path / 'exit_000.txt').write_text('Start-Date: 2026-10-06 10:00:00+02:00\nEnd-Date: 2026-10-06 10:30:00+02:00\n')
            with patch.object(driver, 'completed_job', return_value={}):
                self.assertEqual(driver.scheduling_seconds(root, job, [job]), 1800 * 1.35)
                wrong = {**job, 'treatment': 'sphere_regular'}
                self.assertEqual(driver.scheduling_seconds(root, wrong, [job]), 9999)
            with patch.object(driver, 'completed_job', return_value=None):
                self.assertEqual(driver.scheduling_seconds(root, job, [job]), 9999)

    def test_worker_policy_is_matched_across_all_final_treatments(self):
        items=driver.intervals()
        for policy in [driver.REPO/'config/waymo/autowork_paper-final_worker_policy.json',
                       driver.REPO/'tests/fixtures/final_four_worker_policy.json']:
            with patch('preparation.jobs.FINAL_WORKER_POLICY',policy):
                jobs=driver.make_jobs(items,{label:[]for label in items})
                for label,item in items.items():
                    group=[j for j in jobs if j['label']==label]
                    four=policy.parent.name=='fixtures' and label.startswith('W324_')
                    self.assertEqual({j.get('num_workers',10)for j in group},{4 if four else 10})
                    self.assertEqual(driver.worker_output_suffix({label:item}),'_4workers'if four else'')
                    for job in group:
                        driver.assert_job_worker_policy(job)
                        cmd=driver.command(job,Path('out'))
                        if four:
                            self.assertEqual(cmd[-2:],['--num-workers','4'])
                            self.assertTrue(job['id'].endswith('_workers4'))
                            self.assertIn('worker_policy_sha256',job)
                        else:
                            self.assertNotIn('--num-workers',cmd)
                            self.assertNotIn('num_workers',job)

    def test_completed_four_worker_run_rejects_ten_worker_settings(self):
        root=driver.REPO/'runs/autowork_paper_plane_baselines_part1of2'
        if not(root/'manifest.json').exists():self.skipTest('Completed cohort not available')
        job=json.loads((root/'manifest.json').read_text())['jobs'][0]
        source=next((root/job['id']).glob('attempt_*'))
        reference=driver.config(job['config'])
        reference['save_intermediate_checkpoints']=False
        _,first,last=job['label'].split('_')
        four={**job,'treatment':'plane','segment':'1918764220984209654',
              'first':int(first),'stop':int(last)+1,'num_workers':4}
        original=driver.config
        def effective(path):
            if str(path)==job['config']:return copy.deepcopy(reference)
            c=original(path);c['num_workers']=4;return c
        with patch.object(driver,'config',side_effect=effective):
            self.assertIsNotNone(driver.completed_job(source,four))
        def wrong(path):
            return copy.deepcopy(reference)if str(path)==job['config']else original(path)
        with patch.object(driver,'config',side_effect=wrong),self.assertRaisesRegex(ValueError,'num_workers'):
            driver.completed_job(source,four)


if __name__ == '__main__':
    unittest.main()
