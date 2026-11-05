"""Original PCA construction and matched launcher tests, using synthetic inputs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'reports'), str(REPO / 'nag'),
               str(REPO / 'libs/nag-waymo-processing')]
import paper_pca_baselines as pca
import prepare_paper_pca_geometry as worker
from nag_waymo_processing.ellipsoid import (fit_pca_enclosing_ellipsoid, WAYMO_CAMERA_TO_NAG_CAMERA)


def plan():
    return dict(intervals=[dict(label=label, segment='1234567890', first=0, stop=3,
        pins={}, estimated_training_seconds=1200) for label in pca.INTERVALS],
        recipes={'PCA': pca.RECIPE}, diagnostic_code={})


def synthetic(root):
    cloud = root / 'clouds'; data = root / 'data/segment-1234567890'
    cloud_dir = cloud / data.name
    cloud_dir.mkdir(parents=True); data.mkdir(parents=True)
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = np.arange(3) * .2
    pose_path = data / 'poses.json'
    pose_path.write_text(json.dumps({'position': poses.tolist()}))
    points = np.random.default_rng(42).normal(size=(48, 3)) * [7., 3., 1.] + [0., 0., 20.]
    for frame, pose in enumerate(poses):
        xyz = np.vstack((points, [10000., 10000., 10000.]))
        np.savez(cloud_dir / f'{frame:03d}.npz', xyz=xyz,
                 front_camera_projected=np.r_[np.ones(48, bool), False], format_version=2,
                 front_camera_to_output=pose @ WAYMO_CAMERA_TO_NAG_CAMERA,
                 coordinate_frame='sequence')
    (cloud_dir / 'metadata.json').write_text(json.dumps(dict(sequence=data.name,
        format_version=2, front_camera_id=1, returns='first')))
    pairs = root / 'pairs.npz'; np.savez(pairs, poses=poses, frames=np.arange(3))
    item = dict(label='W123_0_2', segment='1234567890', first=0, stop=3,
        config='/synthetic/config.yaml', data_dir=str(data), pose_path=str(pose_path),
        pairs=str(pairs), size=[32, 24], D2_runs=['/synthetic/main'], pins={})
    item['pca_cloud_sha256'] = pca.cloud_pins(item, cloud)
    return cloud, item, poses, np.tile(points, (3, 1))


class PaperPCABaselineTests(unittest.TestCase):
    def test_matrix_has_twenty_one_new_regular_cells(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            cells = pca.matrix(plan(), Path(d))
        self.assertEqual(len(cells), 21)
        self.assertTrue(all(c['recipe'] == 'PCA' and c['mode'] == 'regular' and c['reused_run'] is None for c in cells))
        for label in pca.INTERVALS:
            self.assertEqual([c['repeat'] for c in cells if c['label'] == label], [1, 2, 3])

    def test_inventory_preserves_original_audits_and_pins_pca_sources(self):
        old = pca.driver.ABLATION_INTERVALS
        def inventory(catalog):
            self.assertEqual(pca.driver.ABLATION_INTERVALS, pca.INTERVALS)
            return plan()
        with patch.object(pca.driver, 'inventory', side_effect=inventory), \
                patch.object(pca, 'cloud_pins', return_value={'cloud': 'pin'}), \
                patch.object(pca, 'raw_environment', return_value={'python': 'processing'}):
            result = pca.inventory(Path('/catalog'), Path('/clouds'))
        self.assertIs(pca.driver.ABLATION_INTERVALS, old)
        self.assertEqual(result['recipes'], {'PCA': pca.RECIPE})
        self.assertEqual(result['processing_environment'], {'python': 'processing'})
        self.assertTrue(all(i['pca_cloud_sha256'] == {'cloud': 'pin'} for i in result['intervals']))
        for path in (Path(pca.__file__), pca.WORKER, *(REPO / name for name in pca.LAUNCHERS)):
            self.assertEqual(result['diagnostic_code'][str(path.resolve())], pca.driver.sha(path))

    def test_scope_restores_original_cohort_on_failure(self):
        old = pca.driver.ABLATION_INTERVALS
        with self.assertRaises(RuntimeError):
            with pca.ego_intervals():
                raise RuntimeError('failure')
        self.assertIs(pca.driver.ABLATION_INTERVALS, old)

    def test_processing_export_is_exact_unmodified_pca_of_all_front_points(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            root = Path(d); cloud, item, poses, points = synthetic(root)
            out = root / 'out'; out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                result = worker.export(item, cloud, out)
            doc = pca.driver.load(result); expected = fit_pca_enclosing_ellipsoid(points)
            for key, field in [('center', 'center'), ('radii', 'radii'), ('rotation_columns', 'rotation')]:
                np.testing.assert_allclose(doc['fit'][key], expected[field], rtol=1e-12, atol=1e-12)
            self.assertEqual(doc['point_count'], len(points))
            self.assertEqual(doc['fit']['mode'], 'pca_enclosing')
            self.assertLess(min(doc['fit']['radii']), 8.)  # no later minimum-radius repair
            self.assertTrue(doc['alignment']['pose_mismatch_correction_enabled'])
            self.assertTrue(doc['fit']['contains_all_points'])
            for name in ('ellipsoid.ply', 'front_points.ply', 'camera_centers.ply'):
                self.assertTrue((out / name).is_file())
            self.assertFalse(any(out.glob('pca_export_*')))
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                worker.export(item, cloud, out)

    def test_cloud_identity_and_hash_changes_are_not_ignored(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            root = Path(d); cloud, item, _, _ = synthetic(root)
            pins = pca.cloud_pins(item, cloud)
            path = cloud / 'segment-1234567890/000.npz'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                pca.driver.assert_pins(pins)
            metadata = cloud / 'segment-1234567890/metadata.json'
            doc = json.loads(metadata.read_text()); doc['returns'] = 'both'; metadata.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                pca.cloud_pins(item, cloud)

    def test_worker_runs_in_the_actual_processing_interpreter_on_synthetic_data(self):
        if not pca.RAW_PYTHON.is_file():
            self.skipTest('Historical processing interpreter is unavailable')
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            root = Path(d); cloud, item, _, points = synthetic(root)
            job = root / 'job.json'; job.write_text(json.dumps(item))
            out = root / 'out'; out.mkdir()
            subprocess.run([str(pca.RAW_PYTHON), str(pca.WORKER), '--job-json', str(job),
                '--cloud-root', str(cloud), '--output', str(out)], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            doc = pca.driver.load(out / 'ellipsoid.json')
            self.assertEqual(doc['point_count'], len(points))
            self.assertEqual(doc['fit']['mode'], 'pca_enclosing')
            self.assertTrue(doc['fit']['contains_all_points'])
            self.assertLess(min(doc['fit']['radii']), 8.)

    def test_diagnostics_report_outside_cameras_and_misses_without_repair(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            root = Path(d); cloud, item, poses, _ = synthetic(root)
            out = root / 'out'; out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                doc = pca.driver.load(worker.export(item, cloud, out))
            before = copy.deepcopy(doc)
            config = {'pinhole_camera_config': {'position': poses.tolist()}}
            with patch.object(pca.driver, 'run_config', return_value=config), \
                    patch.object(pca, 'intrinsics', return_value=np.array([[25., 0., 16.], [0., 25., 12.], [0., 0., 1.]])):
                result = pca.geometry_diagnostics(doc, item)
            self.assertEqual(doc, before)
            self.assertEqual(result['outside_or_on_surface_camera_count'], 3)
            self.assertGreater(result['missed_grid_rays'], 0)
            self.assertFalse(result['geometry_changed_for_safety'])
            self.assertEqual(result['grid_rays'], 3 * 73 * 49)
            wrong = copy.deepcopy(doc); wrong['fit']['mode'] = 'camera_centered_enclosing'
            with self.assertRaises(ValueError):
                pca.geometry_diagnostics(wrong, item)

    def test_original_outside_geometry_requires_explicit_training_acknowledgement(self):
        cells = [dict(label='W191_40_128', geometry='/geometry.json')]
        doc = {'baseline': {'diagnostics': {'outside_or_on_surface_camera_count': 53}}}
        args = SimpleNamespace(intervals=None, allow_outside_cameras=False)
        with patch.object(pca.driver, 'load', return_value=doc):
            with self.assertRaisesRegex(ValueError, '--allow-outside-cameras'):
                pca.acknowledge_outside_cameras(args, cells)
            args.allow_outside_cameras = True
            with contextlib.redirect_stdout(io.StringIO()):
                pca.acknowledge_outside_cameras(args, cells)
            args.allow_outside_cameras = False; args.intervals = ['W254_0_37']
            pca.acknowledge_outside_cameras(args, cells)

    def test_prepare_exports_and_validates_retry_without_training(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            root = Path(d); cloud, item, poses, _ = synthetic(root)
            p = dict(intervals=[item], cloud_root=str(cloud), processing_environment={}, recipes={'PCA': pca.RECIPE})
            target = root / 'autowork_prep'
            def execute(command, **kwargs):
                worker.export(item, cloud, Path(command[command.index('--output') + 1]))
            config = {'pinhole_camera_config': {'position': poses.tolist()}}
            with patch.object(pca.driver, 'check_plan'), patch.object(pca, 'raw_environment', return_value={}), \
                    patch.object(pca.driver, 'run_config', return_value=config), \
                    patch.object(pca, 'intrinsics', return_value=np.array([[25., 0., 16.], [0., 25., 12.], [0., 0., 1.]])), \
                    patch.object(pca.subprocess, 'run', side_effect=execute) as launched, \
                    contextlib.redirect_stdout(io.StringIO()):
                pca.prepare(p, target); pca.prepare(p, target)
            launched.assert_called_once()
            stage = pca.stage(p, item, target)
            document = pca.driver.load(stage.result)
            self.assertFalse(document['baseline']['training_launched'])
            self.assertEqual(document['baseline']['recipe'], pca.RECIPE)
            self.assertEqual(document['baseline']['diagnostics']['outside_or_on_surface_camera_count'], 3)
            self.assertEqual(len(pca.driver.load(target / 'matrix.json')['cells']), 3)
            stage.result.with_suffix('.ply').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                pca.stage(p, item, target)

    def test_prepare_dry_run_never_creates_paths_or_fits(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            with patch.object(pca, 'inventory', return_value=plan()), patch.object(pca, 'prepare') as prepare, \
                    patch.object(pca.driver, 'train_or_summary') as train_call, \
                    contextlib.redirect_stdout(io.StringIO()):
                pca.main(['prepare', '--dry-run', '--preparation-root', str(prep), '--training-root', str(train)])
            prepare.assert_not_called(); train_call.assert_not_called()
            self.assertFalse(prep.exists()); self.assertFalse(train.exists())

    def test_summary_never_requires_outside_camera_acknowledgement(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pca_test_') as d:
            prep = Path(d) / 'autowork_prep'; train = Path(d) / 'autowork_train'
            with patch.object(pca, 'inventory', return_value=plan()), patch.object(pca, 'matrix', return_value=[]), \
                    patch.object(pca, 'show'), patch.object(pca, 'acknowledge_outside_cameras') as ack, \
                    patch.object(pca.driver, 'train_or_summary') as summary:
                pca.main(['train', '--summary', '--preparation-root', str(prep), '--training-root', str(train)])
            ack.assert_not_called(); summary.assert_called_once()
            self.assertEqual(summary.call_args.args[0].stage, 'summary')

    def test_cli_protects_existing_cohorts_and_rejects_bad_options(self):
        bad = [
            ['prepare', '--preparation-root', str(pca.driver.DEFAULT_PREP)],
            ['train', '--training-root', str(pca.DEFAULT_TRAIN.with_name('autowork_remaining_ego_loss_ablations_v1'))],
            ['train', '--intervals', 'W203_60_107'], ['train', '--max-hours', 'nan'],
            ['train', '--preparation-root', '/tmp/autowork_same', '--training-root', '/tmp/autowork_same'],
        ]
        for args in bad:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pca.parse_args(args)

    def test_batches_cover_seven_intervals_and_twenty_one_runs(self):
        labels = []
        for name, intervals in pca.BATCHES.items():
            source = (REPO / f'run_paper_pca_baselines_{name}.sh').read_text()
            self.assertIn('--intervals ' + ' '.join(intervals), source)
            self.assertIn('"$@"', source)
            labels.extend(intervals)
        self.assertEqual(tuple(labels), pca.INTERVALS)
        self.assertEqual(len(set(labels)), 7)


if __name__ == '__main__':
    unittest.main()
