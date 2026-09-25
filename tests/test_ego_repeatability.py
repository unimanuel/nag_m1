"""CPU-only tests for the repeat batch; never starts training."""
import csv
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
