"""Analytic camera/texture checks for the finite-difference sampling diagnostic."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag'))
from texture_sampling_diagnostics import balanced_weights, sampling_metrics, statistics


class TextureSamplingTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.eye(4)[None]
        self.k = np.diag([100., 100., 1.])
        self.geometry = {'center': np.zeros(3), 'radii': np.ones(3) * 10, 'rotation': np.eye(3)}
        self.mapping = {'resolution_hw': [100, 200], 'visible_uv': {'enabled': False}}

    def measure(self):
        return sampling_metrics(np.zeros((1, 2)), np.zeros(1, dtype=int), self.poses,
                                self.k, self.geometry, self.mapping, step=.01)

    def test_centered_sphere_analytic_jacobian(self):
        result = self.measure()
        # Horizontal W/(2*pi*f) and vertical H/(pi*f) are equal here.
        np.testing.assert_allclose(result['texels_per_pixel_min'], 1 / np.pi, rtol=1e-7)
        np.testing.assert_allclose(result['sampling_condition'], 1, rtol=1e-7)

    def test_radius_scaling_does_not_fake_sampling_gain(self):
        before = self.measure()
        self.geometry['radii'] *= 10
        np.testing.assert_allclose(self.measure()['texel_area_per_pixel'], before['texel_area_per_pixel'])

    def test_visible_uv_densification(self):
        self.mapping['visible_uv'] = {'enabled': True, 'resolution': [100, 200],
                                     'u_start': .25, 'v_start': .25, 'u_span': .5, 'v_span': .5}
        result = self.measure()
        np.testing.assert_allclose(result['texels_per_pixel_min'], 2 / np.pi, rtol=1e-7)
        np.testing.assert_allclose(result['texel_area_per_pixel'], 4 / np.pi**2, rtol=1e-7)

    def test_regular_seam_does_not_create_false_stretch(self):
        self.poses[0, :3, :3] = np.diag([-1., 1., -1.])
        np.testing.assert_allclose(self.measure()['texels_per_pixel_max'], 1 / np.pi, rtol=1e-7)

    def test_outside_tile_counted_invalid(self):
        self.mapping['visible_uv'] = {'enabled': True, 'resolution': [100, 200],
                                     'u_start': 0., 'v_start': 0., 'u_span': .1, 'v_span': .1}
        result = self.measure()
        self.assertFalse(result['valid'][0])
        self.assertTrue(np.isnan(result['sampling_condition'][0]))
        self.assertEqual(statistics(result['sampling_condition'], np.ones(1))['valid_fraction'], 0)

    def test_balancing_equal_frame_and_occupied_cell_mass(self):
        ids = np.array([0, 0, 0, 1])
        pixels = np.array([[1., 1.], [1., 1.], [99., 99.], [1., 1.]])
        weights = balanced_weights(ids, pixels, [100, 100])
        np.testing.assert_allclose(weights, [.125, .125, .25, .5])


if __name__ == '__main__':
    unittest.main()
