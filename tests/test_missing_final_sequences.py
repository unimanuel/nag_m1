"""Native CLI forwarding and independent restored-cohort checks; never train or fit."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'nag/scripts'))
import paper_final_baselines as baseline

NEW_SCRIPTS = (
    'missing_final_sequences_common.sh', 'prepare_missing_final_ellipsoids.sh',
    'run_missing_final_baselines_W203.sh', 'run_missing_final_baselines_W975.sh',
    'run_missing_final_ellipsoids_W203.sh', 'run_missing_final_ellipsoids_W975.sh',
)
NATIVE_SCRIPTS = (
    'prepare_final_ellipsoids.sh', 'run_final_ellipsoid_benchmarks.sh',
    'generalized_ellipsoid_preparation_common.sh', 'run_paper_final_baselines.sh',
)
LABELS = {'W203': ['W203_60_107'], 'W975': ['W975_0_68', 'W975_69_99', 'W975_99_162']}


class MissingFinalSequencesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='autowork_restored_final_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in NEW_SCRIPTS + NATIVE_SCRIPTS:
            shutil.copyfile(REPO / name, self.root / name)
        self.parent_manifests = []
        for scene in LABELS:
            parent = self.root / 'runs' / ('autowork_paper_final_baselines_' + scene)
            parent.mkdir(parents=True)
            path = parent / 'manifest.json'
            path.write_text(json.dumps({'immutable_test_parent': scene}))
            self.parent_manifests.append(path)

    def run_script(self, name, *flags, check=True):
        return subprocess.run(['bash', str(self.root / name), *flags], check=check,
            env={**os.environ, 'PYTHON_BIN': '/bin/echo', 'DRY_RUN': '1'},
            text=True, capture_output=True)

    @staticmethod
    def values(tokens, key):
        pos = tokens.index(key) + 1
        result = []
        while pos < len(tokens) and not tokens[pos].startswith('--'):
            result.append(tokens[pos])
            pos += 1
        return result

    def test_prepare_selects_only_four_restored_intervals_and_separate_d_namespace(self):
        tokens = shlex.split(self.run_script('prepare_missing_final_ellipsoids.sh').stdout)
        self.assertEqual(tokens[1], 'prepare')
        self.assertIn('--dry-run', tokens)
        self.assertEqual(self.values(tokens, '--intervals'), LABELS['W203'] + LABELS['W975'])
        self.assertEqual(self.values(tokens, '--output-root'),
            ['/mnt/d/CGProjectData/waymo/ellipsoids/autowork_restored_final_legacy_v1'])

    def test_four_training_queues_use_top_level_roots_and_disjoint_native_manifests(self):
        selections = {}
        for stage, child in [('baselines', 'autowork_restored_baselines'),
                             ('ellipsoids', 'autowork_restored_ellipsoids')]:
            for scene, expected in LABELS.items():
                tokens = shlex.split(self.run_script(
                    f'run_missing_final_{stage}_{scene}.sh', '--max-hours', '6').stdout)
                self.assertIn('--dry-run', tokens)
                self.assertEqual(tokens[-2:], ['--max-hours', '6'])
                self.assertEqual(self.values(tokens, '--intervals'), expected)
                parent = self.root / 'runs' / ('autowork_paper_final_baselines_' + scene)
                flag = '--output-root' if stage == 'baselines' else '--training-root'
                output = self.root / 'runs' / (child + '_' + scene)
                self.assertEqual(self.values(tokens, flag), [str(output)])
                self.assertEqual(output.parent, self.root / 'runs')
                self.assertNotEqual(Path(self.values(tokens, flag)[0]), parent)
                if stage == 'ellipsoids':
                    self.assertEqual(tokens[1], 'train')
                    self.assertEqual(self.values(tokens, '--output-root'),
                        ['/mnt/d/CGProjectData/waymo/ellipsoids/autowork_restored_final_legacy_v1'])
                selections[stage, scene] = expected
        self.assertEqual(sum(len(v) * (9 if k[0] == 'baselines' else 6)
                             for k, v in selections.items()), 60)

    def test_forwarding_preserves_existing_manifests_and_creates_no_outputs(self):
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob('*') if p.is_file()}
        for stage in ['baselines', 'ellipsoids']:
            for scene in LABELS:
                self.run_script(f'run_missing_final_{stage}_{scene}.sh')
        self.run_script('prepare_missing_final_ellipsoids.sh')
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(list(self.root.glob('runs/*/autowork_restored_*')), [])
        self.assertEqual(list(self.root.glob('runs/autowork_restored_*')), [])

    def test_missing_old_scene_does_not_block_or_get_recreated_by_new_queue(self):
        self.parent_manifests[0].unlink()
        self.parent_manifests[0].parent.rmdir()
        result = self.run_script('run_missing_final_baselines_W203.sh')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.values(shlex.split(result.stdout), '--output-root'),
            [str(self.root / 'runs/autowork_restored_baselines_W203')])
        self.assertFalse(self.parent_manifests[0].parent.exists())

    def test_existing_nested_outputs_block_silent_duplicate_queues(self):
        for stage, family, flag in [('baselines', 'baselines', '--output-root'),
                                    ('ellipsoids', 'ellipsoids', '--training-root')]:
            for scene in LABELS:
                old = self.root / 'runs' / ('autowork_paper_final_baselines_' + scene) / (
                    'autowork_restored_' + family)
                old.mkdir()
                (old / 'manifest.json').write_text('retained old cohort')
                script = f'run_missing_final_{stage}_{scene}.sh'
                rejected = self.run_script(script, check=False)
                self.assertEqual(rejected.returncode, 2)
                self.assertIn('Legacy nested restored output exists', rejected.stderr)
                self.assertEqual(rejected.stdout, '')
                explicit = shlex.split(self.run_script(script, flag, str(old)).stdout)
                self.assertEqual(explicit[-2:], [flag, str(old)])
                equal_form = shlex.split(self.run_script(script, f'{flag}={old}').stdout)
                self.assertEqual(equal_form[-1], f'{flag}={old}')
                self.assertEqual((old / 'manifest.json').read_text(), 'retained old cohort')
        self.assertEqual(list(self.root.glob('runs/autowork_restored_*')), [])

    def test_geometry_override_does_not_bypass_legacy_ellipsoid_training_guard(self):
        old = self.root / 'runs/autowork_paper_final_baselines_W203/autowork_restored_ellipsoids'
        old.mkdir()
        result = self.run_script('run_missing_final_ellipsoids_W203.sh',
            '--output-root', '/tmp/autowork_geometry_override', check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--training-root explicitly', result.stderr)

    def test_invalid_stage_scene_and_combined_training_are_rejected(self):
        for args in [(), ('unknown', 'W203'), ('baseline', 'all'), ('ellipsoid', 'all'),
                     ('prepare', 'W324')]:
            result = self.run_script('missing_final_sequences_common.sh', *args, check=False)
            self.assertEqual(result.returncode, 2, args)

    def test_native_summary_and_explicit_output_override_are_forwarded(self):
        tokens = shlex.split(self.run_script('run_missing_final_baselines_W975.sh',
            '--summary', '--output-root', '/tmp/autowork_explicit_override').stdout)
        self.assertIn('--summary', tokens)
        self.assertEqual(tokens[-2:], ['--output-root', '/tmp/autowork_explicit_override'])

    def test_native_paper_settings_and_baseline_job_count_are_unchanged(self):
        items = baseline.intervals()
        selected = {l: items[l] for values in LABELS.values() for l in values}
        baseline.yaml_audit(selected)
        jobs = baseline.make_jobs(selected, {l: [] for l in selected})
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({j['id'] for j in jobs}), 36)
        self.assertEqual({j['seed'] for j in jobs}, {42})
        self.assertEqual({j['treatment'] for j in jobs}, set(baseline.TREATMENTS))
        self.assertFalse(any(j['label'] in {'W203_3_58', 'W975_175_195', 'W324_96_161'} for j in jobs))
        for job in jobs:
            c = baseline.config(job['config'])
            self.assertEqual(c['num_workers'], 10)
            self.assertEqual(c['max_epochs'], 80)
            self.assertFalse(c['save_intermediate_checkpoints'])

    def test_new_wrappers_do_not_change_the_historical_launcher_glob(self):
        for name in NEW_SCRIPTS:
            self.assertFalse(Path(name).match('run_paper_final_baselines*.sh'))
            subprocess.run(['bash', '-n', str(REPO / name)], check=True)


if __name__ == '__main__':
    unittest.main()
