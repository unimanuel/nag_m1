"""Small dependency-light checks for report completion and reuse contracts."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / 'reports'
sys.path.insert(0, str(REPORTS))
spec = importlib.util.spec_from_file_location(
    'latest_upright_report', REPORTS / 'generate_autowork_upright_latest_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class LatestUprightReportTests(unittest.TestCase):
    def test_completion_rejects_missing_frames_wrong_epoch_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix='autowork_report_test_') as directory:
            root = Path(directory)
            (root / 'exit_000.txt').touch()
            (root / 'final/complete').mkdir(parents=True)
            (root / 'tracker').mkdir()
            for index in range(2):
                (root / 'final/complete' / f'{index}_t.png').touch()
            for metric in report.METRICS:
                (root / 'tracker' / f'train_epoch_Mean{metric}.csv').write_text('step;value\n80;1.25\n')
            self.assertEqual(report.completed(root, 2), dict.fromkeys(report.METRICS, 1.25))
            self.assertIsNone(report.completed(root, 3))
            metric = root / 'tracker/train_epoch_MeanMPSNR.csv'
            metric.write_text('step;value\n79;1.25\n')
            self.assertIsNone(report.completed(root, 2))
            metric.write_text('step;value\n80;nan\n')
            self.assertIsNone(report.completed(root, 2))

    def test_reuse_is_explicit_and_macro_is_equal_interval_weight(self):
        values = {
            ('W191_40_128', 'regular', 'upright_fit'): [np.array([10., .5, .2])],
            ('W191_0_40', 'regular', 'strong_floor'): [np.array([30., .9, .1])],
        }
        with self.assertRaises(KeyError):
            report.mean(values, 'W191_40_128', 'strong_floor')
        result = report.aggregate(values, ['W191_40_128', 'W191_0_40'],
                                  'strong_floor', reuse=True)
        np.testing.assert_allclose(result, [20., .7, .15])
        with self.assertRaises(KeyError):
            report.mean(values, 'W191_40_128', 'strong_floor', mode='visuv', reuse=True)

    def test_config_differences_preserve_nested_field_identity(self):
        self.assertEqual(report.config_differences({'nested': {'x': 1}}, {'nested': {'x': 2}}),
                         [('/nested/x', 1, 2)])


if __name__ == '__main__':
    unittest.main()
