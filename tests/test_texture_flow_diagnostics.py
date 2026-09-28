"""Analytic sparse-field checks, periodic charts, support gates and fair subsets."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag'))
from texture_flow_diagnostics import (NeighborhoodConfig, common_masks, correspondence_fields,
                                      difference, local_field, neighborhoods, pair_weights)
from ego_texture_flow_diagnostics import mask_key, measurements


class TextureFlowTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.stack(np.meshgrid(np.arange(7)*3., np.arange(7)*3.), axis=-1).reshape(-1, 2)
        self.count = len(self.pixels)
        self.ids = np.zeros(self.count, dtype=int)
        self.config = NeighborhoodConfig(radius_pixels=10., neighbors=16)
        self.idx, self.dist = neighborhoods(self.pixels, self.ids, self.ids+1,
            np.full(self.count, 20.), np.full(self.count, 20.), self.config)
        self.uv = self.pixels*.001 + [.4, .4]
        self.valid = np.ones(self.count, dtype=bool)

    def fit(self, delta, uv=None, valid=None, period=1.):
        return local_field(self.pixels, self.uv if uv is None else uv, delta, self.idx, self.dist,
                           self.valid if valid is None else valid, self.config, period)

    def test_signed_wrap_not_absolute_distance(self):
        a, b = np.array([[.01, .2], [.99, .1]]), np.array([[.99, .1], [.01, .2]])
        np.testing.assert_allclose(difference(a, b, 1.), [[.02, .1], [-.02, -.1]])
        np.testing.assert_allclose(difference(a, b), [[-.98, .1], [.98, -.1]])

    def test_constant_translation_is_not_spatial_strain(self):
        result = self.fit(np.tile([.05, -.03], (self.count, 1)))
        self.assertTrue(result['chart_fit_valid'].all())
        np.testing.assert_allclose(result['gradient'], 0, atol=1e-12)
        np.testing.assert_allclose(result['affine_residual'], 0, atol=1e-12)
        np.testing.assert_allclose(result['transport_max'], 1, atol=1e-10)
        np.testing.assert_allclose(result['transport_min'], 1, atol=1e-10)

    def test_affine_field_recovers_transport(self):
        a = np.array([[.2, .3], [-.1, -.05]])
        result = self.fit(self.uv@a.T + [.01, -.02])
        expected = np.linalg.svd(np.eye(2)+a, compute_uv=False)
        np.testing.assert_allclose(result['gradient'], np.linalg.norm(a)*.001, rtol=1e-7)
        np.testing.assert_allclose(result['transport_max'], expected[0], rtol=1e-7)
        np.testing.assert_allclose(result['transport_min'], expected[1], rtol=1e-7)
        np.testing.assert_allclose(result['transport_deviation'], np.linalg.norm(a), rtol=1e-7)
        np.testing.assert_allclose(result['affine_residual'], 0, atol=1e-10)

    def test_periodic_chart_has_no_false_seam_gradient(self):
        uv = (self.uv + [.59, 0]) % 1
        result = self.fit(np.tile([.02, -.01], (self.count, 1)), uv=uv)
        self.assertTrue(result['chart_fit_valid'].all())
        np.testing.assert_allclose(result['transport_deviation'], 0, atol=1e-9)

    def test_nonlinear_field_has_nonzero_affine_residual(self):
        delta = np.column_stack((self.pixels[:, 0]**2*.0001, self.pixels[:, 1]**2*.0001))
        result = self.fit(delta)
        self.assertGreater(np.nanmedian(result['affine_residual']), .0001)

    def test_texel_scaling_changes_units_not_isotropic_transport(self):
        delta = self.uv*.2
        a = self.fit(delta)
        b = self.fit(delta*2000, uv=self.uv*2000, period=2000.)
        np.testing.assert_allclose(b['gradient'], a['gradient']*2000, rtol=1e-6)
        np.testing.assert_allclose(b['transport_deviation'], a['transport_deviation'], rtol=1e-6)

    def test_missing_support_remains_nan(self):
        result = self.fit(self.uv*.2, valid=np.zeros(self.count, dtype=bool))
        self.assertFalse(result['image_fit_valid'].any())
        self.assertFalse(result['chart_fit_valid'].any())
        self.assertTrue(np.isnan(result['gradient']).all())

    def test_collinear_image_neighbors_rejected(self):
        pixels = np.column_stack((np.arange(10, dtype=float), np.zeros(10)))
        ids = np.zeros(10, dtype=int)
        idx, dist = neighborhoods(pixels, ids, ids+1, np.ones(10), np.ones(10), self.config)
        result = local_field(pixels, pixels*.001, pixels*.002, idx, dist,
                             np.ones(10, dtype=bool), self.config)
        self.assertFalse(result['image_fit_valid'].any())

    def test_degenerate_texture_chart_rejected(self):
        uv = np.column_stack((self.pixels[:, 0]*.001, np.zeros(self.count)))
        result = self.fit(self.uv*.2, uv=uv)
        self.assertTrue(result['image_fit_valid'].all())
        self.assertFalse(result['chart_fit_valid'].any())

    def test_neighbors_do_not_mix_targets_or_depth_layers(self):
        pixels = np.zeros((4, 2))
        target = np.array([1, 1, 1, 2])
        depth = np.array([10., 10.5, 30., 10.])
        idx, _ = neighborhoods(pixels, np.zeros(4, dtype=int), target, depth, depth, self.config)
        self.assertEqual(set(idx[0][idx[0] >= 0]), {0, 1})
        self.assertEqual(set(idx[2][idx[2] >= 0]), {2})
        self.assertEqual(set(idx[3][idx[3] >= 0]), {3})

    def test_balancing_frames_then_targets_then_cells(self):
        source = np.array([0, 0, 0, 0, 1])
        target = np.array([1, 1, 1, 2, 0])
        pixels = np.array([[1., 1.], [1., 1.], [99., 99.], [1., 1.], [1., 1.]])
        weights = pair_weights(source, target, pixels, [100, 100])
        np.testing.assert_allclose(weights, [.0625, .0625, .125, .25, .5])

    def test_visible_uv_changes_texels_not_global_network_displacement(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 0, 3] = 1.
        pairs = dict(points=np.array([[0., 0., 5.]]), source=np.array([0]), target=np.array([1]), poses=poses)
        geometry = dict(center=np.zeros(3), radii=np.ones(3)*10, rotation=np.eye(3))
        regular = dict(resolution_hw=[100, 200], visible_uv=dict(enabled=False))
        visible = dict(resolution_hw=[100, 200], visible_uv=dict(enabled=True, resolution=[100, 200],
            u_start=.25, u_span=.5, v_start=.25, v_span=.5))
        a, b = [correspondence_fields(pairs, geometry, m) for m in (regular, visible)]
        np.testing.assert_allclose(a['delta_uv'], b['delta_uv'])
        np.testing.assert_allclose(b['delta_texels'], a['delta_texels']*2)
        visible['visible_uv']['u_start'] = 0
        visible['visible_uv']['u_span'] = .1
        c = correspondence_fields(pairs, geometry, visible)
        self.assertFalse(c['tile_covered'][0])
        self.assertTrue(np.isnan(c['delta_texels']).all())

    def test_common_subsets_and_sampling_mask_selection(self):
        keys = ('pair_valid', 'texture_valid', 'sampling_valid', 'uv_image_fit_valid',
                'uv_chart_fit_valid', 'texel_image_fit_valid', 'texel_chart_fit_valid')
        a = {k: np.array([True, True]) for k in keys}
        b = {k: np.array([True, False]) for k in keys}
        masks = common_masks([a, b])
        for value in masks.values():
            np.testing.assert_array_equal(value, [True, False])
        result = measurements(dict(texel_gradient=np.array([1., 100.])), masks, np.ones(2))
        self.assertEqual(result['common_texel_gradient_balanced_mean'], 1.)
        self.assertEqual(result['common_texel_image_fit_valid_fraction'], .5)
        self.assertEqual(mask_key('texel_area_per_pixel'), 'sampling_valid')
        self.assertEqual(mask_key('uv_transport_max'), 'uv_chart_fit_valid')


if __name__ == '__main__':
    unittest.main()
