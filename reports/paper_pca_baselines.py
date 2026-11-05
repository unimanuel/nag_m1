#!/usr/bin/env python3
"""Matched original PCA-enclosure baseline for all seven ego intervals.

No camera-centering, clipping, margin, radius/ratio constraint or floor refinement is
applied. Cameras outside the original enclosure require explicit training
acknowledgement; diagnostic ray-grid misses remain visible, not repaired.
"""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess

import numpy as np

import paper_loss_ablations as driver
from preparation.pipeline import RAW_PYTHON, raw_environment
from diagnostics.parallax_core import intersect
from diagnostics.parallax_io import intrinsics

INTERVALS = ('W191_0_40', 'W191_40_128', 'W191_128_198',
             'W254_0_37', 'W254_37_81', 'W254_86_137', 'W254_139_195')
BATCHES = {'part1of2': INTERVALS[:3], 'part2of2': INTERVALS[3:]}
DEFAULT_CLOUD = Path('/mnt/d/CGProjectData/waymo/pointclouds')
DEFAULT_PREP = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_paper_pca_baselines_v1')
DEFAULT_TRAIN = driver.REPO / 'runs/autowork_paper_pca_baselines_v1'
WORKER = driver.REPO / 'reports/prepare_paper_pca_geometry.py'
LAUNCHERS = ('prepare_paper_pca_baselines.sh', 'run_paper_pca_baselines.sh',
             'run_paper_pca_baselines_part1of2.sh', 'run_paper_pca_baselines_part2of2.sh')
RECIPE = dict(name='original_pca_enclosing_v1', selection='all_FRONT_projected_points',
    correct_pose_mismatches=True, foreground_mask_filter=False, center='PCA_local_box_midpoint',
    radii='all_point_local_extrema_and_uniform_enclosing_scale', camera_expansion=False,
    minimum_radius=None, maximum_axis_ratio=None, sphere_clipping=False,
    mapping='legacy_regular', runtime_margin=0., diagnostic_grid_wh=[73, 49])


@contextmanager
def ego_intervals():
    original = driver.ABLATION_INTERVALS
    driver.ABLATION_INTERVALS = INTERVALS
    try:
        yield
    finally:
        driver.ABLATION_INTERVALS = original


def cloud_pins(item, cloud_root):
    root = Path(cloud_root) / ('segment-' + item['segment'])
    metadata = root / 'metadata.json'
    doc = driver.load(metadata)
    if (doc['sequence'] != 'segment-' + item['segment'] or doc['format_version'] != 2
            or doc['front_camera_id'] != 1 or doc['returns'] != 'first'):
        raise ValueError('Unexpected PCA point-cloud extraction metadata: ' + str(root))
    paths = [metadata] + [root / f'{frame:03d}.npz' for frame in range(item['first'], item['stop'])]
    return {str(path): driver.sha(path) for path in paths}


def inventory(catalog, cloud_root):
    with ego_intervals():
        plan = driver.inventory(catalog)
    if tuple(item['label'] for item in plan['intervals']) != INTERVALS:
        raise ValueError('Expected exactly the seven ego intervals')
    plan.update(cohort='original_pca_ego_baseline_v1', recipes={'PCA': RECIPE},
                cloud_root=str(Path(cloud_root).resolve()), processing_environment=raw_environment())
    source = driver.REPO / 'libs/nag-waymo-processing/nag_waymo_processing'
    for path in (Path(__file__).resolve(), WORKER, source / 'ellipsoid.py', source / 'point_cloud.py',
                 *(driver.REPO / name for name in LAUNCHERS)):
        plan['diagnostic_code'][str(path)] = driver.sha(path)
    for item in plan['intervals']:
        # Read hashes once per invocation. Clouds are not training inputs once
        # their derivative geometry has been exported and frozen.
        item['pca_cloud_sha256'] = cloud_pins(item, cloud_root)
    return plan


def stage_signature(plan, item, root):
    return dict(plan=driver.sha(Path(root) / 'manifest.json'), source=item['pins'],
                clouds=item['pca_cloud_sha256'], recipe=plan['recipes']['PCA'])


def stage(plan, item, root):
    return driver.Stage(Path(root) / item['label'], 'PCA', stage_signature(plan, item, root))


def matrix(plan, root, require_prepared=False):
    cells = []
    for item in plan['intervals']:
        pointer = Path(root) / item['label'] / 'PCA/stage.json'
        if pointer.exists():
            geometry = str(stage(plan, item, root).result)
        elif require_prepared:
            raise ValueError('Prepare PCA first: ' + item['label'])
        else:
            geometry = str(Path(root) / item['label'] / 'PCA/attempt_001/ellipsoid.json')
        for repeat in (1, 2, 3):
            cells.append(dict(id=f'autowork_{item["label"]}_PCA_regular_seed42_repeat{repeat:02d}',
                label=item['label'], recipe='PCA', repeat=repeat, seed=42, mode='regular', geometry=geometry,
                geometry_sha256=driver.sha(geometry) if Path(geometry).is_file() else None,
                reused_run=None, reuse_reason=None))
    return cells


def geometry_diagnostics(document, item):
    from reprojection_ellipsoid import _array_geometry
    fit = document['fit']
    if (document['sequence'] != 'segment-' + item['segment'] or
            tuple(document['frame_slice'][k] for k in ('start', 'stop', 'step')) != (item['first'], item['stop'], 1)
            or fit.get('mode') != 'pca_enclosing' or not fit['uses_all_selected_points']
            or not fit['contains_all_points']):
        raise ValueError('Not the intended original PCA enclosure')
    geometry = _array_geometry(dict(center=fit['center'], radii=fit['radii'], rotation=fit['rotation_columns']))
    with np.load(item['pairs'], allow_pickle=False) as arrays:
        if not np.array_equal(arrays['frames'], np.arange(item['first'], item['stop'])):
            raise ValueError('Camera cache has another interval')
        poses = arrays['poses']
    c = driver.run_config(item['D2_runs'][0])
    expected = np.asarray(c['pinhole_camera_config']['position'])[item['first']:item['stop']]
    if not np.allclose(poses, expected, rtol=0, atol=1e-6):
        raise ValueError('PCA camera diagnostics do not match training calibration')
    k = intrinsics(c['pinhole_camera_config'], item['size'])
    w, h = item['size']
    xx, yy = np.meshgrid(np.linspace(0, w - 1, 73), np.linspace(0, h - 1, 49))
    pixels = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    camera_directions = pixels @ np.linalg.inv(k).T
    rho = np.linalg.norm(((poses[:, :3, 3] - geometry['center']) @ geometry['rotation']) / geometry['radii'], axis=1)
    per_frame = []
    for frame, pose, normalized in zip(range(item['first'], item['stop']), poses, rho):
        origins = np.broadcast_to(pose[:3, 3], camera_directions.shape)
        _, distances, valid = intersect(origins, camera_directions @ pose[:3, :3].T, geometry)
        per_frame.append(dict(frame=frame, camera_normalized_radius=float(normalized),
            grid_rays=len(valid), missed_grid_rays=int((~valid).sum()),
            minimum_positive_hit_m=float(distances[valid].min()) if valid.any() else None))
    return dict(recipe='original_pca_enclosing_v1', geometry_changed_for_safety=False,
        camera_count=len(rho), outside_or_on_surface_camera_count=int((rho >= 1).sum()),
        maximum_camera_normalized_radius=float(rho.max()),
        conservative_camera_clearance_m=float(geometry['radii'].min() * (1 - rho.max())),
        grid_wh=[73, 49], grid_rays=sum(row['grid_rays'] for row in per_frame),
        missed_grid_rays=sum(row['missed_grid_rays'] for row in per_frame), per_frame=per_frame,
        note='Grid coverage is diagnostic, not an exhaustive all-pixel visibility or foreground-depth test.')


def prepare(plan, root):
    driver.freeze(root, plan, dry_run=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + '_preparation.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        driver.freeze(root, plan)
        for item in plan['intervals']:
            driver.check_plan(plan)
            driver.assert_pins(item['pca_cloud_sha256'])
            if raw_environment() != plan['processing_environment']:
                raise ValueError('PCA processing environment changed')
            current = stage(plan, item, root)
            if current.result:
                print('SKIP prepared PCA ' + item['label'], flush=True)
                continue
            out = current.attempt()
            job_path = out / 'job.json'
            driver.write_json(job_path, item)
            print('PCA ' + item['label'], flush=True)
            subprocess.run([str(RAW_PYTHON), str(WORKER), '--job-json', str(job_path),
                '--cloud-root', plan['cloud_root'], '--output', str(out)], cwd=driver.REPO, check=True)
            driver.assert_pins(item['pca_cloud_sha256'])
            document = driver.load(out / 'ellipsoid.json')
            diagnostics = geometry_diagnostics(document, item)
            driver.write_json(out / 'pca_diagnostics.json', diagnostics)
            document['baseline'] = dict(recipe=RECIPE, processing_environment=plan['processing_environment'],
                source_cloud_sha256=item['pca_cloud_sha256'], diagnostics=diagnostics,
                training_launched=False)
            driver.write_json(out / 'ellipsoid.json', document)
            current.complete(out / 'ellipsoid.json', out)
            print('INSPECT ' + str(out / 'ellipsoid.ply'), flush=True)
            print('CAMERAS outside/on=' + str(diagnostics['outside_or_on_surface_camera_count']) +
                  '; rho_max=' + str(diagnostics['maximum_camera_normalized_radius']) +
                  '; diagnostic grid misses=' + str(diagnostics['missed_grid_rays']), flush=True)
        driver.write_json(root / 'matrix.json', dict(cells=matrix(plan, root, True),
                          plan_sha256=driver.sha(root / 'manifest.json')))


def acknowledge_outside_cameras(args, cells):
    selected = [c for c in cells if not args.intervals or c['label'] in args.intervals]
    outside = {}
    for cell in selected:
        d = driver.load(cell['geometry'])['baseline']['diagnostics']
        if d['outside_or_on_surface_camera_count']:
            outside[cell['label']] = d['outside_or_on_surface_camera_count']
    if outside and not args.allow_outside_cameras:
        raise ValueError('Original PCA leaves cameras outside/on the surface: ' + str(outside) +
            '. Inspect the PLYs and pca_diagnostics.json. To deliberately train this unchanged historical recipe, '
            'pass --allow-outside-cameras; no geometry will be repaired.')
    if outside:
        print('ACKNOWLEDGED unchanged original PCA outside-camera cases: ' + str(outside), flush=True)


def show(plan, cells):
    for item in plan['intervals']:
        print(f'{item["label"]}: 3 new PCA repeats; buffered estimate '
              f'{3 * item["estimated_training_seconds"] / 3600:.2f} h', flush=True)
    print(f'TOTAL {len(cells)} new PCA trainings; regular only; no training launched by audit.', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('audit', 'prepare', 'train', 'summary'))
    p.add_argument('--catalog', type=Path, default=driver.DEFAULT_CATALOG)
    p.add_argument('--cloud-root', type=Path, default=DEFAULT_CLOUD)
    p.add_argument('--preparation-root', type=Path, default=DEFAULT_PREP)
    p.add_argument('--training-root', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--intervals', nargs='+', choices=INTERVALS)
    p.add_argument('--max-hours', type=float, default=8.)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--allow-outside-cameras', action='store_true')
    a = p.parse_args(argv)
    if a.summary:
        a.stage = 'summary'
    if a.intervals and a.stage not in ('train', 'summary'):
        p.error('--intervals selects training/summary; prepare all seven geometries first')
    if not np.isfinite(a.max_hours) or a.max_hours <= 0:
        p.error('Positive finite hours required')
    protected = (driver.DEFAULT_PREP, driver.DEFAULT_TRAIN,
        DEFAULT_PREP.with_name('autowork_remaining_ego_loss_ablations_v1'),
        DEFAULT_TRAIN.with_name('autowork_remaining_ego_loss_ablations_v1'))
    for root in (a.preparation_root, a.training_root):
        if not root.name.startswith('autowork_') or root.resolve() == driver.REPO.resolve():
            p.error('Use separate autowork_ namespaces')
        if any(root.resolve() == old.resolve() or old.resolve() in root.resolve().parents for old in protected):
            p.error('Keep completed ablation roots read-only')
    if a.preparation_root.resolve() == a.training_root.resolve():
        p.error('Preparation/training roots must be distinct')
    return a


def main(argv=None):
    args = parse_args(argv)
    plan = inventory(args.catalog, args.cloud_root)
    driver.freeze(args.preparation_root, plan, dry_run=True)
    cells = matrix(plan, args.preparation_root, require_prepared=args.stage in ('train', 'summary'))
    show(plan, cells)
    if args.stage == 'audit' or args.stage == 'prepare' and args.dry_run:
        return
    if args.stage == 'prepare':
        prepare(plan, args.preparation_root)
    else:
        if args.stage == 'train' and not args.dry_run:
            acknowledge_outside_cameras(args, cells)
        with ego_intervals():
            driver.train_or_summary(args, plan, cells)


if __name__ == '__main__':
    main()
