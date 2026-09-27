import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
from poles_sampling_comparison import floor_summary


class FloorSummaryTests(unittest.TestCase):
    def test_threshold_and_huber(self):
        result = floor_summary(np.array([1., .5, .25]), np.ones(3), np.ones(3), np.ones(3, dtype=bool))
        self.assertAlmostEqual(result['floor_below_half_fraction'], 1/3)
        self.assertAlmostEqual(result['floor_huber_loss'], .2*(np.log(2)-.1)/3)

    def test_invalid_samples_are_counted_not_floor_passes(self):
        result = floor_summary(np.array([1., np.nan]), np.ones(2), np.ones(2), np.ones(2, dtype=bool))
        self.assertEqual(result['floor_valid_fraction'], .5)
        self.assertEqual(result['floor_below_half_fraction'], 0.)
        result = floor_summary(np.array([0.]), np.ones(1), np.ones(1), np.ones(1, dtype=bool))
        self.assertIsNone(result['floor_huber_loss'])


if __name__ == '__main__':
    unittest.main()
