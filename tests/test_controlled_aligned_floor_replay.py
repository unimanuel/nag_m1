"""Small aligned-refinement tests without full frozen-data replay."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'nag'))
import controlled_aligned_floor_replay as aligned
import controlled_floor_replay as replay
import controlled_floor_evaluations as evaluations
from reprojection_ellipsoid import ReprojectionFitConfig, enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs


class AlignedReplayTests(unittest.TestCase):
    def test_schedule_rotates_arms_and_identifies_alignment(self):
        jobs = aligned.schedule('W191_0_40', 3)
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({j['id'] for j in jobs}), 9)
        for i in range(3):
            self.assertEqual(jobs[3 * i]['arm'], replay.ARMS[i])
        self.assertTrue(all(j['layout'] == 'aligned64' and 'aligned64' in j['id'] for j in jobs))

    def test_aligned_fit_retains_inputs_and_real_optimizer_calls(self):
        pairs, g = synthetic_pairs()
        pairs = {k: evaluations.aligned_copy(v) for k, v in pairs.items()}
        cfg = ReprojectionFitConfig(max_iterations=1, max_pairs=128, max_depth_rays=128,
                                     minimum_camera_surface_distance=3., minimum_axis_radius=5.)
        g = enforce_safety(g, pairs['poses'], cfg)
        row = {'sphere': {k: v.tolist() for k, v in g.items()},
               'mapping': {'resolution_hw': [640, 960], 'visible_uv': {'enabled': False}}, 'size': [960, 640]}
        functions = replay.engines()
        before = {k: v.copy() for k, v in pairs.items()}
        function = functions['historical_v1']
        real_minimize = function.__globals__['minimize']
        with patch.dict(function.__globals__, {'minimize': Mock(wraps=real_minimize)}):
            tracked = function.__globals__['minimize']
            result = replay.fit_arm(functions, 'historical_v1', pairs, g, row, cfg)
            self.assertEqual(tracked.call_count, 3)
            for call in tracked.call_args_list:
                self.assertEqual(call.kwargs['method'], 'SLSQP')
                self.assertEqual(call.kwargs['options'], {'maxiter': 1, 'ftol': 1e-8})
                self.assertNotIn('jac', call.kwargs)
        self.assertEqual(len(result['candidates']), 3)
        for k in pairs:
            self.assertEqual(pairs[k].ctypes.data % 64, 0)
            np.testing.assert_array_equal(before[k], pairs[k])

    def test_sources_reject_changes(self):
        aligned.validate_sources({'current_sources': {'controlled_aligned_floor_replay.py':
                                 replay.sha(REPO / 'controlled_aligned_floor_replay.py')}})
        with self.assertRaises(ValueError):
            aligned.validate_sources({'current_sources': {'controlled_aligned_floor_replay.py': 'changed'}})

    def test_summary_keeps_failed_gate_even_with_matching_geometries(self):
        geometry = {'center': [0, 0, 0], 'radii': [10, 10, 10], 'rotation': np.eye(3).tolist()}
        jobs = aligned.schedule('test', 3)
        records = [{'job': j, 'input_digest': 'same', 'alignment_mod64': {'points': 0},
                    'geometry': copy.deepcopy(geometry), 'versus_stored': replay.difference(geometry, geometry),
                    'fit': {'sampling_floor': {'final_combined_objective': 1.}}} for j in jobs]
        context = {'jobs': jobs, 'row': {'label': 'test', 'floor': 'stored.json'}}
        baseline = copy.deepcopy(records)
        expected = {'within_arm': {a: {'exact_pair_count': 3, 'pair_count': 3} for a in replay.ARMS},
                    'between_arms': {'test': {'tolerance_pass_count': 9, 'pair_count': 9}}}
        with patch.object(aligned, 'baseline_records', return_value=baseline), \
             patch.object(replay, 'summarize', return_value={'test': copy.deepcopy(expected)}):
            result = aligned.summarize(records, context)
            self.assertTrue(result['all_stored_geometry_comparisons_pass'])
            self.assertTrue(result['all_between_arm_geometry_comparisons_pass'])
            self.assertFalse(result['release_authorized'])
            records[-1]['alignment_mod64']['points'] = 16
            with self.assertRaises(ValueError):
                aligned.summarize(records, context)
            records[-1]['alignment_mod64']['points'] = 0
            records[-1]['input_digest'] = 'changed'
            with self.assertRaises(ValueError):
                aligned.summarize(records, context)
            with self.assertRaises(ValueError):
                aligned.summarize(records[:-1], context)


if __name__ == '__main__':
    unittest.main()
