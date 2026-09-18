"""Sampling-floor invariants, including preservation of a feasible starting fit."""
from pathlib import Path
import sys
import unittest
import contextlib
import io
import json
import tempfile
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag'))
from sampling_floor_ellipsoid import fit_sampling_floor, floor_statistics
from reprojection_ellipsoid import ReprojectionFitConfig, enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs


class SamplingFloorTests(unittest.TestCase):
    def test_penalty_zero_above_floor_and_increases_below_it(self):
        w = np.ones(3)
        ref = np.ones(3)
        self.assertEqual(floor_statistics(np.array([.5, 1., 2.]), ref, w)['loss'], 0)
        a = floor_statistics(np.ones(3) * .25, ref, w)['loss']
        b = floor_statistics(np.ones(3) * .1, ref, w)['loss']
        self.assertGreater(b, a)
        self.assertGreater(a, 0)

    def test_invalid_sampling_is_not_rewarded(self):
        self.assertEqual(floor_statistics(np.array([np.nan]), np.ones(1), np.ones(1))['loss'], 1e6)

    def test_both_sampling_units_scale_together(self):
        a = floor_statistics(np.array([.1, .3]), np.ones(2), np.ones(2))
        b = floor_statistics(np.array([1., 3.]), np.ones(2) * 10, np.ones(2))
        self.assertAlmostEqual(a['loss'], b['loss'])

    def test_optimizer_preserves_budget_and_best_feasible_start(self):
        pairs, geometry = synthetic_pairs()
        cfg = ReprojectionFitConfig(max_pairs=128, max_depth_rays=128,
            minimum_camera_surface_distance=3., minimum_axis_radius=5., max_iterations=2)
        geometry = enforce_safety(geometry, pairs['poses'], cfg)
        sphere = {'center': np.zeros(3), 'radii': np.ones(3) * 30, 'rotation': np.eye(3)}
        mapping = {'resolution_hw': [640, 960], 'visible_uv': {'enabled': False}}
        result = fit_sampling_floor(pairs, geometry, sphere, mapping, (960, 640), cfg, max_sampling=128)
        report = result['sampling_floor']
        self.assertLessEqual(report['final_combined_objective'], report['original_combined_objective'] + 1e-9)
        self.assertLessEqual(report['full_final']['weighted_data_loss'], report['full_original']['weighted_data_loss'] * 1.02 + 1e-6)
        self.assertLessEqual(result['diagnostics']['final']['max_camera_normalized_radius'], .65 + 1e-8)
        np.testing.assert_allclose(result['geometry']['rotation'], geometry['rotation'])

    def test_training_dry_run_emits_ten_isolated_jobs(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
        import sampling_floor_validation as driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in driver.LABELS:
                source, out = driver.paths(root, label)
                segment, first, stop = driver.SEQUENCES[label]
                doc = {'sequence': 'segment-' + segment, 'frame_slice': {'start': first, 'stop': stop, 'step': 1},
                       'fit': {'regularizer_version': driver.VERSION}}
                source.parent.mkdir(parents=True)
                source.write_text(json.dumps(doc))
                out.mkdir()
                (out / 'ellipsoid.json').write_text(json.dumps(doc))
                (out / 'sampling_floor_diagnostics.json').write_text(json.dumps({
                    'full_final': {'weighted_data_loss': 1.}, 'full_original': {'weighted_data_loss': 1.}}))
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['driver', 'train', '--ellipsoid-root', directory, '--dry-run']), \
                    patch.object(driver.subprocess, 'run') as run, contextlib.redirect_stdout(stream):
                driver.main()
            run.assert_not_called()
            self.assertEqual(stream.getvalue().count('--background-ellipsoid-scale-margin 0'), 10)
            self.assertEqual(stream.getvalue().count('--background-ellipsoid-texture-orientation legacy'), 10)


if __name__ == '__main__':
    unittest.main()
