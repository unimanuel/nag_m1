"""Six-interval raw PCA export/launcher tests; never fit, expand or train."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'reports'), str(REPO / 'nag')]
import paper_raw_pca_baselines as raw


def plan():
    return dict(intervals=[dict(label=label, pins={}, estimated_training_seconds=1200)
                          for label in raw.INTERVALS], recipes={raw.TREATMENT: raw.RECIPE},
        diagnostic_code={}, pins={}, exclusions=[dict(label='W191_40_128', max_camera_normalized_radius=1.029)])


def source_fixture(root):
    p = dict(cohort='pca_shared_safety_ego_baseline_v1', recipes={raw.source.TREATMENT: raw.source.RECIPE},
        intervals=[], pins={}, diagnostic_code={})
    for label in raw.source.INTERVALS:
        item = dict(label=label, pins={}, estimated_training_seconds=1200)
        p['intervals'].append(item)
        stage = root / label / 'PCA'; stage.mkdir(parents=True)
        directory = stage / 'attempt_001'; directory.mkdir()
        (stage / 'stage.json').write_text('{}')
        doc = dict(fit=dict(mode='pca_enclosing'), files=dict(front_points_ply='front_points.ply',
            camera_centers_ply='camera_centers.ply'))
        (directory / 'pca_original.json').write_text(json.dumps(doc))
        for name in ('pca_original.ply', 'front_points.ply', 'camera_centers.ply'):
            (directory / name).write_text('original ' + name)
        (directory / 'ellipsoid.json').write_text('{"expanded": true}')
    (root / 'manifest.json').write_text(json.dumps(p))
    return p


class RawPCABaselineTests(unittest.TestCase):
    def test_matrix_is_eighteen_raw_cells_without_exception_or_reuse(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            cells = raw.matrix(plan(), Path(d))
        self.assertEqual(len(cells), 18)
        self.assertEqual(set(c['label'] for c in cells), set(raw.INTERVALS))
        self.assertNotIn('W191_40_128', [c['label'] for c in cells])
        self.assertTrue(all(c['recipe'] == raw.TREATMENT and c['mode'] == 'regular' and
                            c['reused_run'] is None for c in cells))
        self.assertFalse(raw.RECIPE['shared_safety'])
        self.assertFalse(raw.RECIPE['minimal_expansion_exception'])

    def test_inventory_selects_original_assets_and_records_deferral(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            root = Path(d); p = source_fixture(root)
            def stage(p, item, root):
                return SimpleNamespace(result=root / item['label'] / 'PCA/attempt_001/ellipsoid.json')
            def diagnostics(doc, item):
                excluded = item['label'] == 'W191_40_128'
                return dict(outside_or_on_surface_camera_count=53 if excluded else 0,
                    missed_grid_rays=11615 if excluded else 0,
                    maximum_camera_normalized_radius=1.029 if excluded else .9)
            with patch.object(raw.driver, 'check_plan') as check, patch.object(raw.source, 'stage', side_effect=stage), \
                    patch.object(raw.source, 'geometry_diagnostics', side_effect=diagnostics), \
                    patch.object(raw.source, 'shared_safety_expansion', side_effect=AssertionError('Never expand')):
                result = raw.inventory(root)
            check.assert_called_once()
            self.assertEqual(tuple(i['label'] for i in result['intervals']), raw.INTERVALS)
            self.assertEqual([e['label'] for e in result['exclusions']], ['W191_40_128'])
            self.assertEqual(result['exclusions'][0]['outside_camera_count'], 53)
            self.assertFalse(result['exclusions'][0]['minimal_expansion_applied'])
            for item in result['intervals']:
                self.assertTrue(item['raw_pca_sources']['geometry'].endswith('/pca_original.json'))
                self.assertTrue(item['raw_pca_sources']['mesh'].endswith('/pca_original.ply'))
                for asset in item['raw_pca_sources'].values():
                    self.assertEqual(item['pins'][asset], raw.driver.sha(asset))
            self.assertEqual(raw.driver.load(root / 'manifest.json'), p)

    def test_inventory_fails_for_new_outside_case_instead_of_silently_repairing_or_skipping(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            root = Path(d); source_fixture(root)
            def stage(p, item, root):
                return SimpleNamespace(result=root / item['label'] / 'PCA/attempt_001/ellipsoid.json')
            def diagnostics(doc, item):
                return dict(outside_or_on_surface_camera_count=1 if item['label'] == 'W254_0_37' else 0,
                    missed_grid_rays=0, maximum_camera_normalized_radius=1.01)
            with patch.object(raw.driver, 'check_plan'), patch.object(raw.source, 'stage', side_effect=stage), \
                    patch.object(raw.source, 'geometry_diagnostics', side_effect=diagnostics), \
                    self.assertRaisesRegex(ValueError, 'W254_0_37'):
                raw.inventory(root)

    def test_another_source_recipe_is_rejected_before_any_actions(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            root = Path(d); (root / 'manifest.json').write_text('{"cohort": "another recipe"}')
            with patch.object(raw.driver, 'check_plan') as check, self.assertRaises(ValueError):
                raw.inventory(root)
            check.assert_not_called()

    def test_expanded_geometry_cannot_be_relabeled_raw(self):
        with self.assertRaises(ValueError):
            raw.original_diagnostics(dict(fit=dict(mode='pca_enclosing_shared_safety')), {})

    def test_prepare_preserves_original_geometry_mesh_and_small_radii_exactly(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            root = Path(d); src = root / 'source'; src.mkdir()
            poses = np.repeat(np.eye(4)[None], 3, axis=0); poses[:, 0, 3] = [0., .1, .2]
            pairs = root / 'pairs.npz'; np.savez(pairs, frames=np.arange(3), poses=poses)
            doc = dict(sequence='segment-1234567890', frame_slice=dict(start=0, stop=3, step=1),
                fit=dict(mode='pca_enclosing', uses_all_selected_points=True, contains_all_points=True,
                    center=[0., 0., 0.], radii=[12., 5., 2.], rotation_columns=np.eye(3).tolist(), uniform_scale=1.2),
                files=dict(front_points_ply='front_points.ply', camera_centers_ply='camera_centers.ply',
                    ellipsoid_mesh_ply='pca_original.ply'))
            original = src / 'pca_original.json'; original.write_text(json.dumps(doc))
            assets = dict(geometry=str(original))
            for field, name in [('mesh', 'pca_original.ply'), ('front_points', 'front_points.ply'),
                                ('camera_centers', 'camera_centers.ply')]:
                path = src / name; path.write_text('RAW ORIGINAL ' + name); assets[field] = str(path)
            pins = {p: raw.driver.sha(p) for p in assets.values()}
            item = dict(label='W123_0_2', segment='1234567890', first=0, stop=3, size=[32, 24],
                pairs=str(pairs), D2_runs=['/synthetic/main'], pins=pins, raw_pca_sources=assets)
            p = dict(intervals=[item], recipes={raw.TREATMENT: raw.RECIPE}, source_preparation_root=str(src), exclusions=[])
            target = root / 'autowork_raw'
            config = {'pinhole_camera_config': {'position': poses.tolist()}}
            with patch.object(raw.driver, 'check_plan'), patch.object(raw.driver, 'run_config', return_value=config), \
                    patch.object(raw.source, 'intrinsics', return_value=np.array([[25., 0., 16.], [0., 25., 12.], [0., 0., 1.]])), \
                    patch.object(raw.source, 'shared_safety_expansion', side_effect=AssertionError('Never expand')), \
                    patch.object(raw.driver.subprocess, 'run', side_effect=AssertionError('Never fit or train')), \
                    contextlib.redirect_stdout(io.StringIO()):
                raw.prepare(p, target); raw.prepare(p, target)
            result = raw.stage(p, item, target).result; saved = raw.driver.load(result)
            self.assertEqual(saved['fit'], doc['fit'])
            self.assertEqual(saved['fit']['radii'][2], 2.)
            self.assertEqual(result.with_suffix('.ply').read_bytes(), Path(assets['mesh']).read_bytes())
            self.assertFalse(saved['baseline']['safety_projection_applied'])
            self.assertFalse(saved['baseline']['minimal_expansion_applied'])
            self.assertTrue(saved['baseline']['geometry_arrays_preserved_exactly'])
            self.assertEqual(saved['files']['ellipsoid_mesh_ply'], 'ellipsoid.ply')
            self.assertEqual(len(raw.driver.load(target / 'matrix.json')['cells']), 3)
            self.assertEqual(raw.driver.load(original), doc)
            raw.driver.assert_pins(pins)
            result.with_suffix('.ply').write_text('changed')
            with self.assertRaises(ValueError):
                raw.stage(p, item, target)

    def test_prepare_dry_run_never_writes_or_copies(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            with patch.object(raw, 'inventory', return_value=plan()), patch.object(raw, 'prepare') as prepare, \
                    patch.object(raw.driver, 'train_or_summary') as launch, contextlib.redirect_stdout(io.StringIO()):
                raw.main(['prepare', '--dry-run', '--preparation-root', str(prep), '--training-root', str(train)])
            prepare.assert_not_called(); launch.assert_not_called()
            self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_missing_preparation_is_not_silently_generated_by_training(self):
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            with self.assertRaisesRegex(ValueError, 'Prepare raw PCA first'):
                raw.matrix(plan(), Path(d) / 'autowork_absent', True)

    def test_summary_keeps_scope_and_cannot_start_a_training(self):
        old = raw.driver.ABLATION_INTERVALS
        with tempfile.TemporaryDirectory(prefix='autowork_raw_pca_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            def summary(args, p, cells):
                self.assertEqual(args.stage, 'summary')
                self.assertEqual(raw.driver.ABLATION_INTERVALS, raw.INTERVALS)
                self.assertEqual(len(cells), 18)
            with patch.object(raw, 'inventory', return_value=plan()), patch.object(raw, 'matrix', return_value=[{}] * 18), \
                    patch.object(raw.driver, 'train_or_summary', side_effect=summary) as launch, \
                    patch.object(raw, 'prepare') as prepare, contextlib.redirect_stdout(io.StringIO()):
                raw.main(['train', '--summary', '--preparation-root', str(prep), '--training-root', str(train)])
            prepare.assert_not_called(); launch.assert_called_once()
        self.assertIs(raw.driver.ABLATION_INTERVALS, old)

    def test_cli_protects_sources_and_defers_the_exception(self):
        bad = [
            ['train', '--intervals', 'W191_40_128'], ['train', '--allow-outside-cameras'],
            ['train', '--minimal-expansion'], ['train', '--max-hours', 'nan'],
            ['prepare', '--preparation-root', str(raw.source.DEFAULT_PREP)],
            ['train', '--training-root', str(raw.source.DEFAULT_TRAIN)],
            ['prepare', '--source-preparation-root', '/tmp/autowork_source', '--preparation-root', '/tmp/autowork_source/autowork_nested'],
            ['train', '--preparation-root', '/tmp/autowork_one', '--training-root', '/tmp/autowork_one/autowork_nested'],
        ]
        for argv in bad:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                raw.parse_args(argv)
        defaults = raw.parse_args(['prepare'])
        self.assertIn('raw_pca', defaults.preparation_root.name)
        self.assertNotEqual(defaults.training_root, raw.source.DEFAULT_TRAIN)

    def test_batches_cover_only_the_six_included_intervals(self):
        labels = []
        for name, intervals in raw.BATCHES.items():
            text = (REPO / f'run_paper_raw_pca_{name}.sh').read_text()
            self.assertIn('--intervals ' + ' '.join(intervals), text)
            self.assertIn('"$@"', text)
            labels.extend(intervals)
        self.assertEqual(tuple(labels), raw.INTERVALS)
        self.assertEqual([len(v) * 3 for v in raw.BATCHES.values()], [6, 12])


if __name__ == '__main__':
    unittest.main()
