"""Depth/floor scientific isolation, reuse, export and read-only launcher tests."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'reports'), str(REPO / 'nag')]
import paper_depth_floor_ablations as experiment
from preparation import _legacy_floor_kernel as kernel
from reprojection_ellipsoid import enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs

driver = experiment.driver


def cohort():
    changing = {'W191_128_198', 'W254_37_81', 'W254_139_195'}
    return dict(intervals=[dict(label=label, pins={}, D0_geometry='/unused/d0.json',
        D0_runs=['/d0/one', '/d0/two', '/d0/three'], estimated_training_seconds=1200,
        initial_depth_floor_gate=dict(preserve_expected=label not in changing,
                                      floor=dict(loss=.001 if label in changing else 0.)))
        for label in experiment.INTERVALS], recipes={experiment.TREATMENT: experiment.RECIPE})


def synthetic_source(root):
    pairs, geometry = synthetic_pairs()
    pairs['frames'] = np.arange(3)
    geometry = enforce_safety(geometry, pairs['poses'], experiment.CONFIG)
    source = root / 'source'; source.mkdir()
    driver.write_json(source / 'ellipsoid.json', dict(sequence='segment-test',
        frame_slice=dict(start=0, stop=3, step=1, first=0, last=2, count=3), ablation=dict(recipe='D0', stale=True),
        fit=dict(center=geometry['center'].tolist(), radii=geometry['radii'].tolist(),
                 rotation_columns=geometry['rotation'].tolist())))
    for name in ('front_points.ply', 'background_points.ply', 'camera_centers.ply'):
        (source / name).write_text('ply\nformat ascii 1.0\nend_header\n')
    np.savez(source / 'pairs.npz', **pairs)
    item = dict(label='W191_0_40', D0_geometry=str(source / 'ellipsoid.json'),
        D0_runs=['/d0/one', '/d0/two', '/d0/three'], pairs=str(source / 'pairs.npz'),
        prepared=str(source / 'prepared.json'), first=0, stop=3, size=[960, 640], pins={},
        sphere_reference=driver.jsonable(geometry), canonical_mapping=dict(
            mode='legacy', rotation=np.eye(3).tolist(), u_offset=0,
            visible_uv=dict(enabled=False), resolution_hw=[640, 960]),
        estimated_training_seconds=1200)
    item['initial_depth_floor_gate'] = experiment.initial_gate(item)
    plan = dict(intervals=[item], pins={}, diagnostic_code={}, context={},
                recipes={experiment.TREATMENT: experiment.RECIPE})
    return pairs, geometry, item, plan


class DepthFloorNumericsTests(unittest.TestCase):
    def test_recipe_preserves_zero_floor_exactly_without_optimization(self):
        pairs, geometry = synthetic_pairs()
        geometry = enforce_safety(geometry, pairs['poses'], experiment.CONFIG)
        mapping = dict(resolution_hw=[640, 960], visible_uv=dict(enabled=False))
        with patch.object(kernel, 'minimize', side_effect=AssertionError('must preserve')):
            result = experiment.refine(pairs, geometry, geometry, mapping, [960, 640])
        self.assertEqual(result['config']['reprojection_weight'], 0.)
        self.assertEqual(result['config']['depth_weight'], .15)
        self.assertTrue(result['sampling_floor']['zero_floor_preserved'])
        self.assertEqual(result['sampling_floor']['kernel_sha256'], experiment.KERNEL_SHA)
        self.assertEqual(result['sampling_floor']['weights'], [5., 10., 20.])
        self.assertEqual(result['sampling_floor']['max_sampling_rays'], 4096)
        self.assertEqual(result['sampling_floor']['geometric_budget_fraction'], .02)
        for key in geometry:
            np.testing.assert_array_equal(result['geometry'][key], geometry[key])

    def test_active_floor_uses_slsqp_continuation_and_depth_only_budget(self):
        pairs, geometry = synthetic_pairs()
        geometry = enforce_safety(geometry, pairs['poses'], experiment.CONFIG)
        sphere = dict(center=np.zeros(3), radii=np.ones(3) * 30, rotation=np.eye(3))
        mapping = dict(resolution_hw=[640, 960], visible_uv=dict(enabled=False))

        def sampling(pixels, ids, poses, intrinsics, candidate, mapping):
            sigma = 1. if candidate is sphere else .2
            return dict(texels_per_pixel_min=np.ones(len(pixels)) * sigma)

        def unchanged(fun, x, **kwargs):
            self.assertEqual(kwargs['method'], 'SLSQP')
            self.assertEqual(kwargs['options']['maxiter'], 200)
            self.assertGreater(fun(x), 0.)
            return SimpleNamespace(x=x, nit=0, success=True, message='synthetic identity')

        with patch.object(kernel, 'sampling_metrics', side_effect=sampling), \
                patch.object(kernel, 'minimize', side_effect=unchanged) as optimizer, \
                contextlib.redirect_stdout(io.StringIO()):
            result = experiment.refine(pairs, geometry, sphere, mapping, [960, 640])
        self.assertEqual(optimizer.call_count, 3)
        self.assertEqual([c['weight'] for c in result['candidates']], [5., 10., 20.])
        floor = result['sampling_floor']
        self.assertFalse(floor.get('zero_floor_preserved', False))
        self.assertLessEqual(floor['full_final']['weighted_data_loss'],
                             floor['full_original']['weighted_data_loss'] * 1.02 + 1e-6)
        self.assertAlmostEqual(floor['full_final']['weighted_data_loss'],
                               .15 * floor['full_final']['depth_huber_loss'])
        np.testing.assert_array_equal(result['geometry']['rotation'], geometry['rotation'])

    def test_pixel_error_is_diagnostic_not_an_optimization_term(self):
        pairs, geometry = synthetic_pairs()
        def evaluate(error):
            def residual(pairs, geometry, indices):
                return dict(valid=np.ones(len(indices), bool), pixel_error=np.full(len(indices), error),
                            camera_angle_deg=np.zeros(len(indices)))
            with patch('reprojection_ellipsoid._pair_residuals', side_effect=residual):
                return driver.evaluate_geometry(pairs, geometry, np.arange(360), np.arange(180), experiment.CONFIG)
        small, large = evaluate(1.), evaluate(1e8)
        self.assertGreater(large['reprojection_huber_loss'], small['reprojection_huber_loss'])
        self.assertEqual(small['weighted_data_loss'], large['weighted_data_loss'])

    def test_nonzero_reprojection_or_other_weights_are_rejected(self):
        for config in (replace(experiment.CONFIG, reprojection_weight=1e-12),
                       replace(experiment.CONFIG, depth_weight=1.),
                       replace(experiment.CONFIG, radius_excess_weight=.1)):
            with self.assertRaisesRegex(ValueError, 'exactly zero'):
                experiment.refine({}, {}, {}, {}, [960, 640], config)

    def test_full_ray_audit_requires_exact_zero_not_a_small_positive_loss(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            _, _, item, _ = synthetic_source(Path(d))
            initial = item['initial_depth_floor_gate']
            self.assertTrue(initial['preserve_expected'])
            self.assertGreater(initial['full_ray_count'], 128)
            self.assertLessEqual(initial['full_ray_count'], 180)
            with patch.object(experiment, 'floor_statistics', return_value=dict(loss=1e-30)):
                self.assertFalse(experiment.initial_gate(item)['preserve_expected'])
            with patch.object(experiment, 'floor_statistics', return_value=dict(loss=float('nan'))):
                with self.assertRaisesRegex(ValueError, 'Invalid source'):
                    experiment.initial_gate(item)

    def test_wrong_frame_order_and_unsafe_geometry_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            _, _, item, _ = synthetic_source(Path(d))
            with self.assertRaisesRegex(ValueError, 'frame order'):
                experiment.initial_gate({**item, 'stop': 4})
            metrics = item['initial_depth_floor_gate']['full_metrics']
            for key, value in [('weighted_data_loss', float('nan')), ('valid_pair_fraction', .99),
                               ('valid_depth_fraction', .99), ('max_camera_normalized_radius', .66),
                               ('conservative_camera_clearance_m', 5.), ('minimum_axis_radius_m', 7.),
                               ('maximum_axis_ratio', 11.)]:
                with self.assertRaises(ValueError):
                    experiment.require_safe({**metrics, key: value})


class DepthFloorLauncherTests(unittest.TestCase):
    def test_matrix_covers_seven_intervals_with_nine_new_and_twelve_reused_cells(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            cells = experiment.matrix(cohort(), Path(d))
        self.assertEqual(len(cells), 21)
        self.assertEqual(sum(c['reused_run'] is None for c in cells), 9)
        self.assertEqual(sum(c['reused_run'] is None and c['label'] in experiment.BATCHES['part1of2'] for c in cells), 3)
        self.assertEqual(sum(c['reused_run'] is None and c['label'] in experiment.BATCHES['part2of2'] for c in cells), 6)
        for cell in cells:
            self.assertEqual(cell['recipe'], experiment.TREATMENT)
            self.assertEqual(cell['mode'], 'regular')
            self.assertEqual(cell['seed'], 42)
            if cell['reused_run']:
                self.assertEqual(cell['reused_run'], ['/d0/one', '/d0/two', '/d0/three'][cell['repeat'] - 1])

    def test_missing_preparation_cannot_launch_training(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            with self.assertRaisesRegex(ValueError, 'Prepare depth'):
                experiment.matrix(cohort(), Path(d), require_prepared=True)

    def test_prepared_positive_floor_fallback_reuses_only_identical_d0_geometry(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            root = Path(d); _, _, item, plan = synthetic_source(root)
            item['initial_depth_floor_gate']['preserve_expected'] = False
            output = root / 'autowork_output'
            driver.freeze(output, plan)
            current = experiment.stage(plan, item, output)
            attempt = current.attempt()
            doc = driver.load(item['D0_geometry'])
            driver.write_json(attempt / 'ellipsoid.json', doc)
            current.complete(attempt / 'ellipsoid.json', attempt)
            cells = experiment.matrix(plan, output, require_prepared=True)
            self.assertTrue(all(c['reused_run'] for c in cells))
            # A safe replacement needs fresh trainings regardless of the initial gate.
            other = root / 'autowork_changed'
            driver.freeze(other, plan)
            changed = experiment.stage(plan, item, other)
            attempt = changed.attempt()
            doc['fit']['radii'][0] += 1.
            driver.write_json(attempt / 'ellipsoid.json', doc)
            changed.complete(attempt / 'ellipsoid.json', attempt)
            self.assertTrue(all(c['reused_run'] is None for c in experiment.matrix(plan, other, True)))

    def test_prepare_dry_run_never_writes_fits_or_trains(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            prep, train = Path(d) / 'autowork_prep', Path(d) / 'autowork_train'
            with patch.object(experiment, 'inventory', return_value=cohort()), \
                    patch.object(experiment, 'refine') as fit, patch.object(experiment, 'prepare') as prepare, \
                    patch.object(driver, 'train_or_summary') as training, \
                    contextlib.redirect_stdout(io.StringIO()):
                experiment.main(['prepare', '--dry-run', '--preparation-root', str(prep), '--training-root', str(train)])
            fit.assert_not_called(); prepare.assert_not_called(); training.assert_not_called()
            self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_synthetic_preparation_exports_meshes_and_reuses_exact_geometry_safely(self):
        with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
            root = Path(d); _, geometry, item, plan = synthetic_source(root)
            output = root / 'autowork_output'
            with patch.object(driver, 'check_plan'), \
                    patch.object(driver, 'train_or_summary') as training, \
                    patch.object(kernel, 'minimize', side_effect=AssertionError('must preserve')), \
                    contextlib.redirect_stdout(io.StringIO()):
                experiment.prepare(plan, output)
                with patch.object(experiment, 'refine', side_effect=AssertionError('must skip completed stage')):
                    experiment.prepare(plan, output)
            training.assert_not_called()
            result = experiment.stage(plan, item, output).result
            doc = driver.load(result)
            self.assertEqual(doc['ablation']['recipe'], 'D0_floor')
            self.assertNotIn('stale', doc['ablation'])
            self.assertNotIn('reprojection_fit', doc)
            self.assertEqual(doc['depth_floor_fit']['config']['reprojection_weight'], 0.)
            for name, value in geometry.items():
                np.testing.assert_array_equal(doc['fit'][name if name != 'rotation' else 'rotation_columns'], value)
            self.assertTrue(doc['fit']['sampling_floor']['zero_floor_preserved'])
            for filename in doc['files'].values():
                self.assertTrue((result.parent / filename).is_file(), filename)
            self.assertTrue((result.parent / 'ellipsoid.ply').read_bytes().startswith(b'ply\n'))
            import torch
            from nag.model.background_image_ellipsoid_scene_node_3d import load_ellipsoid_geometry
            rendered = load_ellipsoid_geometry(result, scale_margin=0., device=torch.device('cpu'))
            np.testing.assert_array_equal(rendered.radii.numpy(), geometry['radii'].astype(np.float32))
            np.testing.assert_array_equal(rendered.center.numpy(), geometry['center'].astype(np.float32))
            cells = experiment.matrix(plan, output, require_prepared=True)
            self.assertTrue(all(c['reused_run'] for c in cells))
            self.assertEqual(len(list(result.parent.parent.glob('attempt_*'))), 1)
            (result.parent / 'ellipsoid.ply').write_text('unexpected overwrite')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                experiment.matrix(plan, output, require_prepared=True)

    def test_cohort_scope_restored_on_exception_and_pin_conflicts_rejected(self):
        original = driver.ABLATION_INTERVALS
        with self.assertRaises(RuntimeError):
            with experiment.ego_intervals():
                self.assertEqual(driver.ABLATION_INTERVALS, experiment.INTERVALS)
                raise RuntimeError('test')
        self.assertIs(driver.ABLATION_INTERVALS, original)
        pins = {'/file': 'first'}
        experiment.merge_pins(pins, {'/other': 'second'})
        with self.assertRaisesRegex(ValueError, 'Conflicting source'):
            experiment.merge_pins(pins, {'/file': 'different'})

    def test_training_and_summary_dispatch_use_existing_runner_with_all_seven_intervals(self):
        for summary in (False, True):
            with tempfile.TemporaryDirectory(prefix='autowork_depth_floor_test_') as d:
                prep, train = Path(d) / 'autowork_prep', Path(d) / 'autowork_train'
                def dispatch(args, plan, cells):
                    self.assertEqual(driver.ABLATION_INTERVALS, experiment.INTERVALS)
                    self.assertEqual(args.stage, 'summary' if summary else 'train')
                    self.assertEqual(args.intervals, list(experiment.BATCHES['part2of2']))
                    self.assertEqual(len(cells), 21)
                argv = ['train', '--dry-run', '--preparation-root', str(prep), '--training-root', str(train),
                        '--intervals', *experiment.BATCHES['part2of2']]
                if summary:
                    argv.append('--summary')
                original = driver.ABLATION_INTERVALS
                with patch.object(experiment, 'inventory', return_value=cohort()), \
                        patch.object(experiment, 'matrix', return_value=[{}] * 21) as matrix, \
                        patch.object(experiment, 'show'), \
                        patch.object(driver, 'train_or_summary', side_effect=dispatch) as training:
                    experiment.main(argv)
                training.assert_called_once()
                self.assertTrue(matrix.call_args.kwargs['require_prepared'])
                self.assertIs(driver.ABLATION_INTERVALS, original)
                self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_roots_and_cli_reject_overwrites_and_invalid_budgets(self):
        args = experiment.parse_args(['train'])
        self.assertEqual(args.max_hours, 8.)
        self.assertEqual(set().union(*map(set, experiment.BATCHES.values())), set(experiment.INTERVALS))
        for argv in (['prepare', '--preparation-root', str(driver.DEFAULT_PREP)],
                     ['train', '--training-root', str(driver.DEFAULT_TRAIN)],
                     ['train', '--training-root', str(experiment.DEFAULT_SOURCES[1] / 'autowork_child')],
                     ['prepare', '--intervals', 'W191_0_40'], ['train', '--max-hours', 'nan'],
                     ['train', '--max-hours', '0'], ['train', '--max-hours', '-1'],
                     ['train', '--preparation-root', '/tmp/autowork_same', '--training-root', '/tmp/autowork_same']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.parse_args(argv)


if __name__ == '__main__':
    unittest.main()
