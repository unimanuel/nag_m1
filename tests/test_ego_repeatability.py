"""CPU-only tests for the repeat batch; never starts training."""
import csv
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
import ego_repeatability as driver


class RepeatabilityTests(unittest.TestCase):
    def test_default_and_extended_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, (segment, first, stop) in driver.SEQUENCES.items():
                version = 2 if label in driver.V2 else 1
                folder = root / f'segment-{segment}' / f'frames_{first:03d}_{stop-1:03d}_reprojection_depth_sampling_floor_v{version}'
                folder.mkdir(parents=True)
                (folder / 'ellipsoid.json').write_text(json.dumps({'sequence': f'segment-{segment}',
                    'frame_slice': {'start': first, 'stop': stop, 'step': 1, 'count': stop-first},
                    'fit': {'regularizer_version': f'sampling_floor_v{version}'}}))
            a = driver.plan(list(driver.SEQUENCES), ['regular', 'visuv'], 3, [], root)
            b = driver.plan(list(driver.SEQUENCES), ['regular', 'visuv'], 3, [43, 44], root)
            self.assertEqual(len(a), 42)
            self.assertEqual(len(b), 70)
            self.assertEqual(len({j['id'] for j in b}), 70)
            self.assertTrue(all(j['seed'] == 42 for j in a))
            self.assertEqual(sum(j['group'] == 'different_seed' for j in b), 28)
            cmd = driver.command(a[0], root / 'output')
            self.assertEqual(cmd[cmd.index('--seed') + 1], '42')
            self.assertEqual(cmd[cmd.index('--output-path') + 1], str(root / 'output'))
            self.assertNotIn('--deterministic', cmd)
            combined = []
            for part, labels in driver.BATCHES.items():
                batch = driver.plan(labels, ['regular', 'visuv'], 3, [], root, ['legacy', 'world_y'])
                self.assertEqual(len(batch), 36 if part == 'part1' else 48)
                for first, second in zip(batch[::2], batch[1::2]):
                    self.assertEqual(first['geometry_sha256'], second['geometry_sha256'])
                    self.assertEqual(first['config_sha256'], second['config_sha256'])
                    self.assertEqual(first['seed'], second['seed'])
                    self.assertEqual({first['orientation'], second['orientation']}, {'legacy', 'world_y'})
                    expected = 'legacy' if first['repeat'] % 2 else 'world_y'
                    self.assertEqual(first['orientation'], expected)
                    cmd = driver.command(second, root / 'output')
                    self.assertEqual(cmd[cmd.index('--background-ellipsoid-texture-orientation') + 1], second['orientation'])
                combined.extend(batch)
            self.assertEqual(len({j['id'] for j in combined}), 84)
            self.assertEqual({j['label'] for j in combined}, set(driver.SEQUENCES))

    def test_summary_separates_orientations_and_reports_paired_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for i, orientation in enumerate(['legacy', 'world_y']):
                (root / orientation / 'attempt_001').mkdir(parents=True)
                jobs.append(dict(id=orientation, label='W254_86_137', mode='regular',
                    group='same_seed', seed=42, repeat=1, orientation=orientation, frame_count=52))
            def metrics(path, count):
                return {m: (2. if path.parent.name == 'world_y' else 1.) for m in driver.METRICS}
            stream = io.StringIO()
            with patch.object(driver, 'completed', side_effect=metrics), contextlib.redirect_stdout(stream):
                driver.summarize(root, jobs)
            self.assertIn('same_seed / legacy n=1', stream.getvalue())
            self.assertIn('same_seed / world_y n=1', stream.getvalue())
            self.assertIn('mean_delta=+1.000000', stream.getvalue())

    def test_completion_requires_exit_frames_and_finite_epoch80_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.assertIsNone(driver.completed(p, 2))
            (p / 'exit_000.txt').touch()
            (p / 'final/complete').mkdir(parents=True)
            for i in range(2): (p / 'final/complete' / f'{i}_t.png').touch()
            (p / 'tracker').mkdir()
            for metric in driver.METRICS:
                (p / 'tracker' / f'train_epoch_Mean{metric}.csv').write_text('step;value\n80;0.5\n')
            self.assertIsNotNone(driver.completed(p, 2))
            (p / 'tracker/train_epoch_MeanLPIPS.csv').write_text('step;value\n79;0.5\n')
            self.assertIsNone(driver.completed(p, 2))
            (p / 'tracker/train_epoch_MeanLPIPS.csv').write_text('step;value\n80;nan\n')
            self.assertIsNone(driver.completed(p, 2))

    def test_summary_reads_manifest_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'manifest.json').write_text(json.dumps({'jobs': []}))
            with patch.object(sys, 'argv', ['driver', '--summary', '--output-root', tmp]), \
                    patch.object(driver.subprocess, 'run') as run:
                driver.main()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
