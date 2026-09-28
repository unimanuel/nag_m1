"""Analytic flow units, periodic charts, invalid support and parity guards."""
from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
from learned_flow_diagnostics import association, flow_fields, parity, statistics, uv_difference
from ego_learned_background_flow import check_records, completed_result, file_record, saved_layers


class LearnedFlowTests(unittest.TestCase):
    def setUp(self):
        xy = np.array([[0., 0.], [1., 1.], [2., 2.]])
        offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        self.before = .4+(xy[None]+offsets[:, None])*.001
        self.valid = np.ones((5, 3), dtype=bool)

    def evaluate(self, native, after=None, tile=None, projected=None, scale=(1920, 960)):
        return flow_fields(self.before, self.before+native if after is None else after, native,
                           native*1000, self.valid, self.valid if projected is None else projected,
                           self.valid if tile is None else tile, scale)

    def test_constant_flow_has_zero_strain(self):
        fields, flags = self.evaluate(np.broadcast_to([.02, -.01], self.before.shape))
        np.testing.assert_allclose(fields['native_uv_gradient'], 0, atol=1e-12)
        np.testing.assert_allclose(fields['native_uv_curvature'], 0, atol=1e-12)
        np.testing.assert_allclose(fields['transport_max'], 1, atol=1e-10)
        np.testing.assert_allclose(fields['transport_deviation'], 0, atol=1e-10)
        self.assertTrue(flags['chart_valid'].all())

    def test_affine_flow_has_correct_chart_transport(self):
        a = np.array([[.2, .3], [-.1, -.05]])
        fields, _ = self.evaluate(self.before@a.T)
        sv = np.linalg.svd(np.eye(2)+a, compute_uv=False)
        np.testing.assert_allclose(fields['native_uv_gradient'], np.linalg.norm(a)*.001, rtol=1e-8)
        np.testing.assert_allclose(fields['transport_max'], sv[0], rtol=1e-8)
        np.testing.assert_allclose(fields['transport_min'], sv[1], rtol=1e-8)
        np.testing.assert_allclose(fields['transport_determinant'], np.linalg.det(np.eye(2)+a), rtol=1e-8)

    def test_longitude_seam_does_not_invent_transport(self):
        self.before[..., 0] = (self.before[..., 0]+.6) % 1
        native = np.broadcast_to([.02, .01], self.before.shape)
        after = (self.before+native).copy()
        after[..., 0] %= 1
        fields, _ = self.evaluate(native, after)
        np.testing.assert_allclose(fields['effective_uv_magnitude'], np.hypot(.02, .01))
        np.testing.assert_allclose(fields['transport_deviation'], 0, atol=1e-9)

    def test_visible_uv_scaling_is_not_network_scaling(self):
        native = self.before*.1
        a, _ = self.evaluate(native)
        b, _ = self.evaluate(native, scale=(3840, 1920))
        np.testing.assert_allclose(a['native_uv_magnitude'], b['native_uv_magnitude'])
        np.testing.assert_allclose(a['transport_max'], b['transport_max'])
        np.testing.assert_allclose(b['initial_texture_texel_gradient'], a['initial_texture_texel_gradient']*2)

    def test_missing_tile_does_not_erase_native_or_hit_support(self):
        fields, flags = self.evaluate(self.before*.1, tile=np.zeros((5, 3), dtype=bool))
        self.assertTrue(flags['stencil_valid'].all())
        self.assertTrue(flags['chart_valid'].all())
        self.assertTrue(np.isfinite(fields['native_uv_gradient']).all())
        self.assertTrue(np.isnan(fields['initial_texture_texel_magnitude']).all())

    def test_projected_behind_camera_is_not_zero_flow(self):
        projected = self.valid.copy()
        projected[0, 0] = False
        fields, flags = self.evaluate(self.before*.1, projected=projected)
        self.assertFalse(flags['valid'][0])
        self.assertTrue(np.isnan(fields['image_pixel_magnitude'][0]))
        self.assertTrue(np.isnan(fields['native_uv_gradient'][0]))

    def test_invalid_stencil_retains_valid_center_magnitude(self):
        projected = self.valid.copy()
        projected[1, 0] = False
        fields, _ = self.evaluate(self.before*.1, projected=projected)
        self.assertTrue(np.isfinite(fields['native_uv_magnitude'][0]))
        self.assertTrue(np.isnan(fields['native_uv_gradient'][0]))

    def test_latitude_clamp_can_collapse_transport(self):
        native = np.broadcast_to([0, .7], self.before.shape)
        after = self.before+native
        after[..., 1] = 1
        fields, _ = self.evaluate(native, after)
        np.testing.assert_allclose(fields['transport_min'], 0, atol=1e-12)
        self.assertTrue(np.isnan(fields['transport_condition']).all())

    def test_statistics_balance_frames_not_valid_pixel_counts(self):
        values = np.array([[0., 0., 0.], [10., np.nan, np.nan]])
        result = statistics(values, np.ones_like(values, dtype=bool), [0, 1])
        self.assertEqual(result['mean'], 5)
        self.assertEqual(result['frames'], 2)
        self.assertEqual(statistics(values, np.zeros_like(values, dtype=bool), [0, 1])['mean'], None)

    def test_parity_uses_bytes_and_fails_drift(self):
        a = np.zeros((10, 10, 3), np.uint8)
        self.assertTrue(parity(a, a)['passed'])
        self.assertFalse(parity(a+1, a)['passed'])
        self.assertFalse(parity(a+10, a, mean_limit=20)['passed'])
        with self.assertRaises(ValueError):
            parity(a, a[:2])

    def test_association_is_missing_for_constant_or_tiny_samples(self):
        self.assertIsNone(association(np.ones(20), np.arange(20), np.ones(20, bool)))
        self.assertEqual(association(np.arange(20), np.arange(20), np.ones(20, bool)), 1)

    def test_input_change_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'config.yaml'
            p.write_text('old')
            record = file_record(p, True)
            check_records([record])
            p.write_text('new')
            with self.assertRaises(ValueError):
                check_records([record])

    def test_saved_layers_match_local_and_dataset_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete, bg = root/'final/complete', root/'final/4_Background Plane 4'
            complete.mkdir(parents=True)
            bg.mkdir()
            Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(complete/'00_t.png')
            Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(bg/'00_t_086.png')
            result = saved_layers(root, [86])
            self.assertEqual(Path(result[0]['background']).name, '00_t_086.png')
            with self.assertRaises(ValueError):
                saved_layers(root, [87])

    def test_signed_periodic_difference(self):
        np.testing.assert_allclose(uv_difference([[.01, .3]], [[.99, .4]]), [[.02, -.1]])

    def test_incomplete_or_wrong_scope_outputs_cannot_resume_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(id='test', frames=[10, 11, 12], indices=[0, 2])
            result = dict(complete=True, parity_passed=True, job='test', frames=[10, 12])
            (root/'result.json').write_text(json.dumps(result))
            self.assertFalse(completed_result(root, job))
            for name in ('fields.npz', 'metrics.csv', 'by_frame.csv'):
                (root/name).write_text('test')
            (root/'verification.json').write_text(json.dumps(dict(passed=True)))
            self.assertTrue(completed_result(root, job))
            result['frames'] = [10]
            (root/'result.json').write_text(json.dumps(result))
            self.assertFalse(completed_result(root, job))
            result.update(frames=[10, 12], complete=False)
            (root/'result.json').write_text(json.dumps(result))
            self.assertFalse(completed_result(root, job))


if __name__ == '__main__':
    unittest.main()
