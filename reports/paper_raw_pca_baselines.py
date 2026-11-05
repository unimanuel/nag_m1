#!/usr/bin/env python3
"""Train untouched PCA enclosures on the six currently runnable ego intervals.

Reuse the original (not expanded) exports from completed PCA preparation.
W191 40-128 is explicitly deferred: NAG rejects its outside initial cameras.
No safety projection, minimal exception, fitting, or training runs in audit.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil

import numpy as np

import paper_pca_baselines as source

driver = source.driver
EXCLUDED = ('W191_40_128',)
INTERVALS = tuple(label for label in source.INTERVALS if label not in EXCLUDED)
BATCHES = {'part1of2': INTERVALS[:2], 'part2of2': INTERVALS[2:]}
DEFAULT_SOURCE = source.DEFAULT_PREP
DEFAULT_PREP = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_paper_raw_pca_v1')
DEFAULT_TRAIN = driver.REPO / 'runs/autowork_paper_raw_pca_v1'
TREATMENT = 'Raw_PCA'
LAUNCHERS = ('prepare_paper_raw_pca.sh', 'run_paper_raw_pca.sh',
             'run_paper_raw_pca_part1of2.sh', 'run_paper_raw_pca_part2of2.sh')
RECIPE = dict(name='raw_pca_enclosing_v1', selection='all_FRONT_projected_points',
    correct_pose_mismatches=True, foreground_mask_filter=False,
    center='original_PCA_local_box_midpoint', radii='original_all_point_extrema_and_uniform_enclosing_scale',
    shared_safety=False, camera_expansion=False, minimum_radius=None, maximum_axis_ratio=None,
    sphere_clipping=False, mapping='legacy_regular', runtime_margin=0.,
    excluded_intervals=list(EXCLUDED), minimal_expansion_exception=False)


@contextmanager
def raw_intervals():
    original = driver.ABLATION_INTERVALS
    driver.ABLATION_INTERVALS = INTERVALS
    try:
        yield
    finally:
        driver.ABLATION_INTERVALS = original


def original_diagnostics(document, item):
    if document['fit'].get('mode') != 'pca_enclosing':
        raise ValueError('Raw PCA requires the original export, not a safety-expanded fit')
    return source.geometry_diagnostics(document, item)


def inventory(source_root):
    source_root = Path(source_root).resolve()
    manifest = source_root / 'manifest.json'
    original_plan = driver.load(manifest)
    if (original_plan.get('cohort') != 'pca_shared_safety_ego_baseline_v1' or
            original_plan.get('recipes') != {source.TREATMENT: source.RECIPE} or
            tuple(i['label'] for i in original_plan['intervals']) != source.INTERVALS):
        raise ValueError('Requires the completed seven-interval PCA preparation with preserved originals')
    driver.check_plan(original_plan)
    plan = copy.deepcopy(original_plan)
    plan.update(cohort='raw_pca_six_ego_baseline_v1', recipes={TREATMENT: RECIPE}, intervals=[],
                source_preparation_root=str(source_root), exclusions=[])
    plan['pins'][str(manifest)] = driver.sha(manifest)
    for path in (Path(__file__).resolve(), *(driver.REPO / name for name in LAUNCHERS)):
        plan['diagnostic_code'][str(path)] = driver.sha(path)
    for original_item in original_plan['intervals']:
        item = copy.deepcopy(original_item)
        stage = source.stage(original_plan, original_item, source_root)
        if stage.result is None:
            raise ValueError('Missing original PCA export for ' + item['label'] + '; keep the completed preparation')
        directory = stage.result.parent
        geometry = directory / 'pca_original.json'
        mesh = directory / 'pca_original.ply'
        document = driver.load(geometry)
        diagnostics = original_diagnostics(document, item)
        stage_manifest = source_root / item['label'] / 'PCA/stage.json'
        plan['pins'][str(stage_manifest)] = driver.sha(stage_manifest)
        if item['label'] in EXCLUDED:
            plan['pins'][str(geometry)] = driver.sha(geometry)
            plan['exclusions'].append(dict(label=item['label'], reason='NAG initialization requires all camera radii < 1; raw PCA is deferred without any repair',
                geometry=str(geometry), max_camera_normalized_radius=diagnostics['maximum_camera_normalized_radius'],
                outside_camera_count=diagnostics['outside_or_on_surface_camera_count'],
                minimal_expansion_applied=False))
            continue
        if diagnostics['outside_or_on_surface_camera_count'] or diagnostics['missed_grid_rays']:
            raise ValueError('Raw PCA cannot initialize ' + item['label'] + '; do not silently repair or skip another interval')
        assets = dict(geometry=str(geometry), mesh=str(mesh),
            front_points=str(directory / document['files']['front_points_ply']),
            camera_centers=str(directory / document['files']['camera_centers_ply']))
        for path in assets.values():
            item['pins'][path] = driver.sha(path)
        item.update(raw_pca_sources=assets, raw_pca_diagnostics=diagnostics)
        plan['intervals'].append(item)
    if tuple(i['label'] for i in plan['intervals']) != INTERVALS:
        raise ValueError('Unexpected raw PCA training cohort')
    return plan


def stage(plan, item, root):
    signature = dict(plan=driver.sha(Path(root) / 'manifest.json'), source=item['pins'], recipe=RECIPE)
    return driver.Stage(Path(root) / item['label'], 'raw_PCA', signature)


def matrix(plan, root, require_prepared=False):
    cells = []
    for item in plan['intervals']:
        if (Path(root) / item['label'] / 'raw_PCA/stage.json').exists():
            geometry = str(stage(plan, item, root).result)
        elif require_prepared:
            raise ValueError('Prepare raw PCA first: ' + item['label'])
        else:
            geometry = str(Path(root) / item['label'] / 'raw_PCA/attempt_001/ellipsoid.json')
        for repeat in (1, 2, 3):
            cells.append(dict(id=f'autowork_{item["label"]}_raw_PCA_regular_seed42_repeat{repeat:02d}',
                label=item['label'], recipe=TREATMENT, repeat=repeat, seed=42, mode='regular',
                geometry=geometry, geometry_sha256=driver.sha(geometry) if Path(geometry).is_file() else None,
                reused_run=None, reuse_reason=None))
    return cells


def prepare(plan, root):
    driver.freeze(root, plan, dry_run=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + '_preparation.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        driver.freeze(root, plan)
        for item in plan['intervals']:
            driver.check_plan(plan)
            current = stage(plan, item, root)
            if current.result:
                print('SKIP prepared raw PCA ' + item['label'], flush=True)
                continue
            out = current.attempt()
            assets = item['raw_pca_sources']
            document = driver.load(assets['geometry'])
            original_fit = copy.deepcopy(document['fit'])
            diagnostics = original_diagnostics(document, item)
            if diagnostics['outside_or_on_surface_camera_count'] or diagnostics['missed_grid_rays']:
                raise ValueError('Raw PCA failed initialization/coverage checks; no expansion is allowed')
            for field, filename in [('mesh', 'ellipsoid.ply'), ('front_points', 'front_points.ply'),
                                    ('camera_centers', 'camera_centers.ply')]:
                shutil.copyfile(assets[field], out / filename)
            document['files'].update(ellipsoid_mesh_ply='ellipsoid.ply',
                front_points_ply='front_points.ply', camera_centers_ply='camera_centers.ply')
            document['baseline'] = dict(recipe=RECIPE, source_original_geometry=assets['geometry'],
                source_original_sha256=item['pins'][assets['geometry']],
                source_preparation_root=plan['source_preparation_root'], diagnostics=diagnostics,
                geometry_arrays_preserved_exactly=True, safety_projection_applied=False,
                minimal_expansion_applied=False, training_launched=False)
            if document['fit'] != original_fit:
                raise ValueError('Raw PCA fit metadata must remain exactly unchanged')
            driver.write_json(out / 'ellipsoid.json', document)
            driver.write_json(out / 'pca_diagnostics.json', diagnostics)
            current.complete(out / 'ellipsoid.json', out)
            print('INSPECT ' + str(out / 'ellipsoid.ply') +
                  '; rho_max=' + str(diagnostics['maximum_camera_normalized_radius']), flush=True)
        driver.write_json(root / 'matrix.json', dict(cells=matrix(plan, root, True),
            exclusions=plan['exclusions'], plan_sha256=driver.sha(root / 'manifest.json')))


def show(plan, cells):
    for excluded in plan['exclusions']:
        print(f'DEFERRED {excluded["label"]}: raw rho_max={excluded["max_camera_normalized_radius"]:.6f}; no training or repair', flush=True)
    for item in plan['intervals']:
        print(f'{item["label"]}: 3 new RAW PCA repeats; buffered estimate '
              f'{3 * item["estimated_training_seconds"] / 3600:.2f} h', flush=True)
    print(f'TOTAL {len(cells)} new raw-PCA trainings; no shared safety, no exception expansion.', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('audit', 'prepare', 'train', 'summary'))
    p.add_argument('--source-preparation-root', type=Path, default=DEFAULT_SOURCE)
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
        p.error('--intervals selects training/status; prepare all six raw exports first')
    if not np.isfinite(a.max_hours) or a.max_hours <= 0:
        p.error('Positive finite hours required')
    protected = (driver.DEFAULT_PREP, driver.DEFAULT_TRAIN, source.DEFAULT_PREP, source.DEFAULT_TRAIN,
        source.ORIGINAL_PREP, source.ORIGINAL_TRAIN, a.source_preparation_root,
        DEFAULT_PREP.with_name('autowork_remaining_ego_loss_ablations_v1'),
        DEFAULT_TRAIN.with_name('autowork_remaining_ego_loss_ablations_v1'))
    for root in (a.preparation_root, a.training_root):
        resolved = root.resolve()
        if not root.name.startswith('autowork_') or resolved == driver.REPO.resolve():
            p.error('Use distinct autowork_ output roots')
        if any(resolved == old.resolve() or old.resolve() in resolved.parents or resolved in old.resolve().parents
               for old in protected):
            p.error('Keep original/shared-safety sources and ablation roots read-only')
    prep, train = a.preparation_root.resolve(), a.training_root.resolve()
    if prep == train or prep in train.parents or train in prep.parents:
        p.error('Preparation/training roots must be separate, not nested')
    return a


def main(argv=None):
    args = parse_args(argv)
    plan = inventory(args.source_preparation_root)
    driver.freeze(args.preparation_root, plan, dry_run=True)
    cells = matrix(plan, args.preparation_root, require_prepared=args.stage in ('train', 'summary'))
    show(plan, cells)
    if args.stage == 'audit' or args.stage == 'prepare' and args.dry_run:
        return
    if args.stage == 'prepare':
        prepare(plan, args.preparation_root)
    else:
        with raw_intervals():
            driver.train_or_summary(args, plan, cells)


if __name__ == '__main__':
    main()
