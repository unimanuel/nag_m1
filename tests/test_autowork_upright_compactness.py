"""Isolation and naming checks for the experimental upright fitter."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
import autowork_upright_compactness as driver


class UprightCompactnessDriverTests(unittest.TestCase):
    def test_experiment_folders_are_isolated_and_prefixed(self):
        source, previous, output = driver.paths(Path('/tmp/ellipsoids'), 'W254_0_37',
                                                  driver.version(.1))
        self.assertEqual(output.name, 'autowork_frames_000_037_upright_compact_w0100_v1')
        self.assertNotEqual(source.parent, output)
        self.assertNotEqual(previous.parent, output)

    def test_weight_is_encoded_without_collisions(self):
        self.assertNotEqual(driver.version(.05), driver.version(.1))
        self.assertEqual(driver.strong_floor_version(.8), 'upright_strong_floor_t0800_v1')
        for bad in (0, -1, float('nan'), .10001):
            with self.assertRaises(ValueError):
                driver.version(bad)
        for bad in (0, .5, 1.01, float('nan'), .80001):
            with self.assertRaises(ValueError):
                driver.strong_floor_version(bad)


if __name__ == '__main__':
    unittest.main()
