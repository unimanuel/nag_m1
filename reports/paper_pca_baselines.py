#!/usr/bin/env python3
"""Matched PCA enclosure plus the shared D0/D1/D2 safety expansion.

Keep the original PCA center and axes; expand radii only using the unchanged
shared safety projection. Export original/effective meshes in a new namespace.
No reprojection optimization, camera-centering, clipping or floor is applied.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess

import numpy as np

import paper_loss_ablations as driver
from preparation.pipeline import RAW_PYTHON, raw_environment
from diagnostics.parallax_core import intersect
from diagnostics.parallax_io import intrinsics
from reprojection_ellipsoid import (ReprojectionFitConfig, _array_geometry,
                                   _camera_safety, enforce_safety)

INTERVALS = ('W191_0_40', 'W191_40_128', 'W191_128_198',
             'W254_0_37', 'W254_37_81', 'W254_86_137', 'W254_139_195')
BATCHES = {'part1of2': INTERVALS[:3], 'part2of2': INTERVALS[3:]}
DEFAULT_CLOUD = Path('/mnt/d/CGProjectData/waymo/pointclouds')
DEFAULT_PREP = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_paper_pca_shared_safety_v1')
DEFAULT_TRAIN = driver.REPO / 'runs/autowork_paper_pca_shared_safety_v1'
ORIGINAL_PREP = DEFAULT_PREP.with_name('autowork_paper_pca_baselines_v1')
ORIGINAL_TRAIN = DEFAULT_TRAIN.with_name('autowork_paper_pca_baselines_v1')
WORKER = driver.REPO / 'reports/prepare_paper_pca_geometry.py'
LAUNCHERS = ('prepare_paper_pca_baselines.sh', 'run_paper_pca_baselines.sh',
             'run_paper_pca_baselines_part1of2.sh', 'run_paper_pca_baselines_part2of2.sh')
TREATMENT = 'PCA_shared_safety'
SAFETY_CONFIG = ReprojectionFitConfig()
SAFETY = dict(max_camera_normalized_radius=SAFETY_CONFIG.max_camera_normalized_radius,
    minimum_camera_surface_distance=SAFETY_CONFIG.minimum_camera_surface_distance,
    minimum_axis_radius=SAFETY_CONFIG.minimum_axis_radius, max_axis_ratio=SAFETY_CONFIG.max_axis_ratio)
RECIPE = dict(name='pca_enclosing_shared_safety_v1', selection='all_FRONT_projected_points',
    correct_pose_mismatches=True, foreground_mask_filter=False, center='PCA_local_box_midpoint',
    radii='PCA_all_point_enclosure_then_shared_radius_only_safety_expansion', camera_expansion=True,
    shared_safety=SAFETY, shared_safety_function='nag/reprojection_ellipsoid.py:enforce_safety',
    center_and_rotation_preserved=True, sphere_clipping=False,
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
    plan.update(cohort='pca_shared_safety_ego_baseline_v1', recipes={TREATMENT: RECIPE},
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
                clouds=item['pca_cloud_sha256'], recipe=plan['recipes'][TREATMENT])


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
            cells.append(dict(id=f'autowork_{item["label"]}_PCA_safety_regular_seed42_repeat{repeat:02d}',
                label=item['label'], recipe=TREATMENT, repeat=repeat, seed=42, mode='regular', geometry=geometry,
                geometry_sha256=driver.sha(geometry) if Path(geometry).is_file() else None,
                reused_run=None, reuse_reason=None))
    return cells


def shared_safety_expansion(document, poses):
    """Apply the exact common safety projection without fitting or recentering."""
    fit = document['fit']
    if fit.get('mode') != 'pca_enclosing' or not fit['contains_all_points']:
        raise ValueError('Shared PCA safety requires the original all-point enclosure')
    original = _array_geometry(dict(center=fit['center'], radii=fit['radii'], rotation=fit['rotation_columns']))
    effective = enforce_safety(original, poses, SAFETY_CONFIG)
    if (not np.array_equal(original['center'], effective['center']) or
            not np.array_equal(original['rotation'], effective['rotation']) or
            np.any(effective['radii'] < original['radii'])):
        raise ValueError('Shared PCA safety must preserve center/axes and only expand radii')
    before, after = _camera_safety(original, poses), _camera_safety(effective, poses)
    if (after['max_camera_normalized_radius'] > SAFETY_CONFIG.max_camera_normalized_radius + 1e-7 or
            after['conservative_camera_clearance_m'] < SAFETY_CONFIG.minimum_camera_surface_distance - 1e-7 or
            after['minimum_axis_radius_m'] < SAFETY_CONFIG.minimum_axis_radius - 1e-7 or
            after['maximum_axis_ratio'] > SAFETY_CONFIG.max_axis_ratio + 1e-7):
        raise ValueError('Expanded PCA geometry failed the shared safety checks')
    result = copy.deepcopy(document)
    target = result['fit']
    target.update(mode='pca_enclosing_shared_safety',
        method='Original PCA enclosure followed by shared radius-only safety expansion; no iterative fit',
        radii=effective['radii'].tolist(), diameters=(2 * effective['radii']).tolist(),
        point_coverage_fraction=1.0,
        shared_safety_radius_expansion_applied=bool(np.any(effective['radii'] > original['radii'])),
        max_camera_normalized_radius=after['max_camera_normalized_radius'],
        conservative_minimum_camera_surface_distance=after['conservative_camera_clearance_m'],
        actual_axis_ratio=after['maximum_axis_ratio'])
    # Minimum-radius/ratio repair can change axis proportions. Do not describe
    # the final radii as one uniform multiple of the original PCA half extents.
    for name in ('uniform_scale', 'base_half_extents', 'max_normalized_radius'):
        if name in target:
            target['pca_original_' + name] = target.pop(name)
    provenance = dict(settings=SAFETY, before=before, after=after,
        radius_factors=(effective['radii'] / original['radii']).tolist(),
        center_preserved_exactly=True, rotation_preserved_exactly=True,
        original_json='pca_original.json', original_mesh='pca_original.ply',
        all_point_enclosure_preserved_by_radius_expansion=True)
    return result, provenance


def geometry_diagnostics(document, item):
    fit = document['fit']
    if (document['sequence'] != 'segment-' + item['segment'] or
            tuple(document['frame_slice'][k] for k in ('start', 'stop', 'step')) != (item['first'], item['stop'], 1)
            or fit.get('mode') not in ('pca_enclosing', 'pca_enclosing_shared_safety') or not fit['uses_all_selected_points']
            or not fit['contains_all_points']):
        raise ValueError('Not the intended PCA enclosure')
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
    corrected = fit['mode'] == 'pca_enclosing_shared_safety'
    return dict(recipe=RECIPE['name'] if corrected else 'original_pca_enclosing_v1',
        geometry_changed_for_safety=bool(fit.get('shared_safety_radius_expansion_applied', False)),
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
            original = driver.load(out / 'ellipsoid.json')
            original_diagnostics = geometry_diagnostics(original, item)
            with np.load(item['pairs'], allow_pickle=False) as arrays:
                document, safety = shared_safety_expansion(original, arrays['poses'])
            diagnostics = geometry_diagnostics(document, item)
            if diagnostics['outside_or_on_surface_camera_count'] or diagnostics['missed_grid_rays']:
                raise ValueError('Shared-safety PCA left an outside camera or invalid diagnostic ray')
            original['files']['ellipsoid_mesh_ply'] = 'pca_original.ply'
            driver.write_json(out / 'pca_original.json', original)
            if (out / 'pca_original.ply').exists():
                raise FileExistsError('Refuse to overwrite original PCA inspection mesh')
            (out / 'ellipsoid.ply').replace(out / 'pca_original.ply')
            from export_background_geometry_meshes import mesh_surface, write_mesh_ply, ELLIPSOID_BLUE
            fit = document['fit']
            vertices, faces = mesh_surface(np.asarray(fit['center']), np.asarray(fit['radii']),
                                           np.asarray(fit['rotation_columns']), 48, 96)
            write_mesh_ply(out / 'ellipsoid.ply', vertices, faces, ELLIPSOID_BLUE)
            driver.write_json(out / 'pca_diagnostics.json', dict(original=original_diagnostics,
                              effective=diagnostics, shared_safety=safety))
            document['baseline'] = dict(recipe=RECIPE, processing_environment=plan['processing_environment'],
                source_cloud_sha256=item['pca_cloud_sha256'], diagnostics=diagnostics,
                original_diagnostics=original_diagnostics, shared_safety=safety,
                training_launched=False)
            driver.write_json(out / 'ellipsoid.json', document)
            current.complete(out / 'ellipsoid.json', out)
            print('INSPECT ' + str(out / 'ellipsoid.ply'), flush=True)
            print('CAMERAS outside/on=' + str(diagnostics['outside_or_on_surface_camera_count']) +
                  '; rho_max=' + str(diagnostics['maximum_camera_normalized_radius']) +
                  '; diagnostic grid misses=' + str(diagnostics['missed_grid_rays']), flush=True)
        driver.write_json(root / 'matrix.json', dict(cells=matrix(plan, root, True),
                          plan_sha256=driver.sha(root / 'manifest.json')))


def show(plan, cells):
    for item in plan['intervals']:
        print(f'{item["label"]}: 3 new PCA + shared-safety repeats; buffered estimate '
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
    a = p.parse_args(argv)
    if a.summary:
        a.stage = 'summary'
    if a.intervals and a.stage not in ('train', 'summary'):
        p.error('--intervals selects training/summary; prepare all seven geometries first')
    if not np.isfinite(a.max_hours) or a.max_hours <= 0:
        p.error('Positive finite hours required')
    protected = (driver.DEFAULT_PREP, driver.DEFAULT_TRAIN, ORIGINAL_PREP, ORIGINAL_TRAIN,
        DEFAULT_PREP.with_name('autowork_remaining_ego_loss_ablations_v1'),
        DEFAULT_TRAIN.with_name('autowork_remaining_ego_loss_ablations_v1'))
    for root in (a.preparation_root, a.training_root):
        if not root.name.startswith('autowork_') or root.resolve() == driver.REPO.resolve():
            p.error('Use separate autowork_ namespaces')
        if any(root.resolve() == old.resolve() or old.resolve() in root.resolve().parents for old in protected):
            p.error('Keep original PCA and completed ablation roots read-only')
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
        with ego_intervals():
            driver.train_or_summary(args, plan, cells)


if __name__ == '__main__':
    main()
