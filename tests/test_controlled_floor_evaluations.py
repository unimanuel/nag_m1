"""Closure interception and stencils, without frozen-data optimization."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'nag'))
import controlled_floor_evaluations as probe
import controlled_floor_replay as replay
from reprojection_ellipsoid import ReprojectionFitConfig, enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs


class EvaluationTests(unittest.TestCase):
    def test_aligned_copy_preserves_values_dtype_strides(self):
        for dtype in (np.float64, np.int32, np.uint8):
            a = np.arange(36, dtype=dtype).reshape(12, 3)
            b = probe.aligned_copy(a)
            self.assertEqual(b.ctypes.data % 64, 0)
            self.assertEqual(b.strides, a.strides)
            self.assertEqual(b.dtype, a.dtype)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.shares_memory(a, b))

    def test_alignment_does_not_silently_change_noncontiguous_layout(self):
        with self.assertRaises(ValueError):
            probe.aligned_copy(np.ones((3, 4)).T)

    def test_capture_all_arms_without_running_minimize(self):
        pairs, g = synthetic_pairs()
        cfg = ReprojectionFitConfig(max_iterations=1, max_pairs=128, max_depth_rays=128,
                                     minimum_camera_surface_distance=3., minimum_axis_radius=5.)
        g = enforce_safety(g, pairs['poses'], cfg)
        row = {'sphere': {k: v.tolist() for k, v in g.items()},
               'mapping': {'resolution_hw': [640, 960], 'visible_uv': {'enabled': False}}, 'size': [960, 640]}
        functions = replay.engines()
        for arm in replay.ARMS:
            function = functions[arm]
            kernel = function.__globals__.get('kernel', function)
            before = kernel.__globals__['minimize']
            # Force only the preservation decision to remain nonzero in this
            # synthetic case; the real experiment has a naturally positive loss.
            actual_stats = kernel.__globals__['floor_statistics']
            def stats(*args, **kwargs):
                result = actual_stats(*args, **kwargs)
                return {**result, 'loss': max(result['loss'], 1e-12)}
            with patch.dict(kernel.__globals__, {'floor_statistics': stats}):
                captured = probe.capture_objective(functions, arm, pairs, g, row, cfg)
            self.assertIs(kernel.__globals__['minimize'], before)
            self.assertEqual(len(captured['x0']), 6)
            with patch('scipy.optimize.minimize', side_effect=AssertionError('No optimization')):
                evaluations = probe.evaluate_stencils(captured, {'initial': captured['x0'].tolist()})
            self.assertTrue(evaluations['initial']['uncached_repeats_exact'])
            self.assertEqual(np.asarray(evaluations['initial']['derivatives']['slsqp_forward']['jacobian']).shape, (15, 6))

    def test_stencils_match_known_quadratic_and_constraint_derivatives(self):
        def measure(x):
            return {'weighted_data_loss': float(x @ x), 'reprojection_huber_loss': float(x @ x),
                    'depth_huber_loss': 0.}, {'loss': float(x[0] ** 2)}
        def score(x, w):
            m, f = measure(x)
            return m['weighted_data_loss'] + w * f['loss']
        capture = {'cache': {}, 'measure': measure, 'score': score,
                   'constraints': lambda x: np.r_[x, 0., 1.], 'bounds': [(-10., 10.)] * 6}
        x = np.array([1., 2., 3., 4., 5., 6.])
        result = probe.evaluate_stencils(capture, {'test': x.tolist()})['test']
        for method in result['derivatives'].values():
            jac = np.asarray(method['jacobian'])
            expected = 2 * x; expected[0] *= 21
            np.testing.assert_allclose(jac[2], expected, atol=1e-5)
            np.testing.assert_allclose(jac[7:13], np.eye(6), atol=1e-7)
        self.assertEqual(len(result['derivatives']['slsqp_forward']['samples']), 6)
        self.assertEqual(len(result['derivatives']['central_1e-6']['samples']), 12)

    def test_summary_rejects_inputs_and_stencils_that_differ(self):
        import copy
        value = {'parameters': [0.] * 6, 'values': [1.] * 15,
                 'uncached_repeats_exact': True, 'derivatives': {
                     'test': {'jacobian': np.zeros((15, 6)).tolist(),
                              'samples': [{'parameters': [1.] * 6, 'values': [1.] * 15}]}}}
        records = [{'job': {'arm': arm, 'layout': layout, 'repeat': repeat},
                    'input_digest': 'same', 'evaluations': {'initial': copy.deepcopy(value)}}
                   for repeat in (1, 2) for layout in ('native', 'aligned64') for arm in replay.ARMS]
        summary = probe.summarize(records)
        self.assertTrue(summary['all_uncached_repeats_exact'])
        self.assertTrue(summary['between_arms']['aligned64_historical_v1_vs_generalized_v2'][0]['initial']['values_exact'])
        records[-1]['input_digest'] = 'changed'
        with self.assertRaises(ValueError):
            probe.summarize(records)
        records[-1]['input_digest'] = 'same'
        records[-1]['evaluations']['initial']['derivatives']['test']['samples'][0]['parameters'][0] = 2.
        with self.assertRaises(ValueError):
            probe.summarize(records)

    def test_optimizer_global_restored_after_initialization_failure(self):
        functions = replay.engines()
        for arm in replay.ARMS:
            function = functions[arm]
            kernel = function.__globals__.get('kernel', function)
            before = kernel.__globals__['minimize']
            with self.assertRaises(KeyError):
                probe.capture_objective(functions, arm, {}, {}, {}, ReprojectionFitConfig())
            self.assertIs(kernel.__globals__['minimize'], before)


if __name__ == '__main__':
    unittest.main()
