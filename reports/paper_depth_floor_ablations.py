#!/usr/bin/env python3
"""Depth plus frozen legacy sampling floor, with no reprojection-error cost.

Initialize from completed D0 fits, preserve zero-floor originals exactly, and
reuse D0 trainings only for identical float32 effective geometry/settings.
Historical fitting kernels, source manifests and model code remain untouched.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from pathlib import Path
import re

import numpy as np

import paper_loss_ablations as driver
from preparation.legacy_floor import fit_sampling_floor
from preparation._legacy_floor_kernel import floor_statistics
from diagnostics.parallax_core import inside
from diagnostics.parallax_evaluate import batched_project
from texture_sampling_diagnostics import balanced_weights, sampling_metrics

INTERVALS = ('W191_0_40', 'W191_40_128', 'W191_128_198',
             'W254_0_37', 'W254_37_81', 'W254_86_137', 'W254_139_195')
BATCHES = {'part1of2': INTERVALS[:3], 'part2of2': INTERVALS[3:]}
DEFAULT_SOURCES = (driver.DEFAULT_TRAIN, driver.REPO / 'runs/autowork_remaining_ego_loss_ablations_v1')
DEFAULT_PREP = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_paper_depth_floor_v1')
DEFAULT_TRAIN = driver.REPO / 'runs/autowork_paper_depth_floor_v1'
TREATMENT = 'D0_floor'
CONFIG = driver.ReprojectionFitConfig(reprojection_weight=0.)
KERNEL = driver.REPO / 'nag/preparation/_legacy_floor_kernel.py'
KERNEL_SHA = '5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603'
LAUNCHERS = ('prepare_paper_depth_floor.sh', 'run_paper_depth_floor.sh',
             'run_paper_depth_floor_part1of2.sh', 'run_paper_depth_floor_part2of2.sh')
RECIPE = dict(name='depth_sampling_floor_ablation_v1', initialization='completed_matched_D0',
    config=asdict(CONFIG), floor_threshold=.5, floor_huber=.2, continuation=[5., 10., 20.],
    max_sampling_rays=4096, geometric_budget_fraction=.02, geometric_budget_epsilon=1e-6,
    preserve_exact_zero_full_ray_floor=True, kernel_sha256=KERNEL_SHA,
    mapping='legacy_regular', runtime_margin=0., sphere_clipping=False)


@contextmanager
def ego_intervals():
    original = driver.ABLATION_INTERVALS
    driver.ABLATION_INTERVALS = INTERVALS
    try:
        yield
    finally:
        driver.ABLATION_INTERVALS = original


def merge_pins(target, incoming):
    for path, digest in incoming.items():
        if path in target and target[path] != digest:
            raise ValueError('Conflicting source provenance: ' + path)
        target[path] = digest


def require_safe(metrics):
    if (not np.isfinite(metrics['weighted_data_loss']) or
            min(metrics['valid_pair_fraction'], metrics['valid_depth_fraction']) != 1 or
            metrics['max_camera_normalized_radius'] > .6500001 or
            metrics['conservative_camera_clearance_m'] < 5.9999999 or
            metrics['minimum_axis_radius_m'] < 7.9999999 or metrics['maximum_axis_ratio'] > 10.0000001):
        raise ValueError('Depth/floor geometry failed full-observation validity or common safety')


def initial_gate(item):
    """Read-only exact full-ray preservation test; never invokes optimization."""
    with np.load(item['pairs'], allow_pickle=False) as archive:
        pairs = dict(archive)
    if not np.array_equal(pairs['frames'], np.arange(item['first'], item['stop'])):
        raise ValueError('D0 observation frame order changed')
    geometry = driver._source_geometry(driver.load(item['D0_geometry']))
    pixels, depth = batched_project(pairs['source_points'], pairs['source_ids'], pairs['poses'], pairs['intrinsics'])
    valid = inside(pixels, depth, item['size'])
    pixels, ids = pixels[valid], pairs['source_ids'][valid]
    if len(ids) < 128:
        raise ValueError('Too few full-ray sampling observations')
    sphere = sampling_metrics(pixels, ids, pairs['poses'], pairs['intrinsics'],
                              item['sphere_reference'], item['canonical_mapping'])['texels_per_pixel_min']
    sigma = sampling_metrics(pixels, ids, pairs['poses'], pairs['intrinsics'],
                             geometry, item['canonical_mapping'])['texels_per_pixel_min']
    if not np.isfinite(sphere).all() or np.any(sphere <= 0):
        raise ValueError('Invalid sphere sampling reference')
    floor = floor_statistics(sigma, sphere, balanced_weights(ids, pixels, item['size']))
    if not np.isfinite(floor['loss']) or floor['loss'] >= 1e6:
        raise ValueError('Invalid source sampling measurements')
    full = driver.evaluate_geometry(pairs, geometry, np.flatnonzero(pairs['visibility'] == 1),
                                   np.arange(len(pairs['source_points'])), CONFIG)
    require_safe(full)
    return dict(full_ray_count=len(ids), floor=floor, full_metrics=full,
                preserve_expected=bool(floor['loss'] == 0.))


def inventory(training_roots):
    if driver.sha(KERNEL) != KERNEL_SHA:
        raise ValueError('Frozen legacy sampling-floor kernel changed')
    plan = dict(schema=1, cohort='depth_sampling_floor_ego_ablation_v1', intervals=[], pins={},
        diagnostic_code={}, context=driver.native_context(), recipes={TREATMENT: RECIPE})
    found = {}
    for root in map(Path, training_roots):
        manifest_path = root / 'manifest.json'
        manifest = driver.load(manifest_path)
        old = manifest['plan']
        if manifest.get('schema') != 1 or old['recipes']['D0'] != asdict(CONFIG):
            raise ValueError('Source is not the completed matched depth-only ablation')
        driver.check_plan(old)
        merge_pins(plan['pins'], old['pins'])
        merge_pins(plan['diagnostic_code'], old['diagnostic_code'])
        plan['pins'][str(manifest_path.resolve())] = driver.sha(manifest_path)
        for item in old['intervals']:
            label = item['label']
            if label not in INTERVALS:
                continue
            if label in found:
                raise ValueError('Duplicate D0 interval sources: ' + label)
            item = copy.deepcopy(item)
            cells = sorted([c for c in manifest['cells'] if c['label'] == label and c['recipe'] == 'D0'],
                           key=lambda c: c['repeat'])
            if [c['repeat'] for c in cells] != [1, 2, 3] or any(c['seed'] != 42 or c['mode'] != 'regular' for c in cells):
                raise ValueError('Requires three explicit regular seed-42 D0 repeats')
            geometry = Path(cells[0]['geometry'])
            if any(c['geometry'] != str(geometry) or c['geometry_sha256'] != driver.sha(geometry) for c in cells):
                raise ValueError('D0 repeat geometry mismatch')
            prep_root = geometry.parents[3]
            prep_manifest = prep_root / 'manifest.json'
            if driver.sha(prep_manifest) != manifest['preparation_manifest'] or driver.load(prep_manifest) != old:
                raise ValueError('D0 preparation/training manifests differ')
            prepared = driver.Stage(prep_root / label, 'D0', driver.d0_signature(old, item, prep_root))
            if prepared.result is None or prepared.result.resolve() != geometry.resolve():
                raise ValueError('D0 preparation stage is incomplete or points elsewhere')
            doc = driver.load(geometry)
            if (doc['reprojection_fit']['config'] != asdict(CONFIG) or
                    doc['fit'].get('regularizer_version') != 'ablation_depth_only_v1' or
                    doc['ablation']['kernel_source_sha256'] != driver.SOURCE_SHA256 or
                    doc['sequence'] != 'segment-' + item['segment'] or
                    tuple(doc['frame_slice'][k] for k in ('start', 'stop', 'step')) != (item['first'], item['stop'], 1) or
                    Path(doc['reprojection_fit']['prepared_observations']).resolve() != Path(item['prepared']).resolve()):
                raise ValueError('Unexpected D0 objective or source/cache identity')
            for path in (geometry, prep_manifest, prep_root / label / 'D0/stage.json'):
                item['pins'][str(path)] = driver.sha(path)
            runs, times = [], []
            for cell in cells:
                candidates = [p for p in (root / cell['id']).glob('attempt_*')
                              if p.is_dir() and re.fullmatch(r'attempt_\d+', p.name)]
                good = [p for p in candidates if driver.completed(p, item, cell)]
                if len(good) != 1:
                    raise ValueError('Expected exactly one complete matched D0 attempt: ' + cell['id'])
                run = good[0]; runs.append(str(run.resolve()))
                times.append(driver.duration_seconds(run / 'exit_000.txt'))
                paths = [run / name for name in ('training_config.yaml', 'exit_000.txt',
                                                 'initial/background_diagnostics/ellipsoid_geometry.txt')]
                paths += [run / 'tracker' / f'train_epoch_{prefix}{metric}.csv'
                          for metric in driver.stats.METRICS for prefix in ('', 'Mean')]
                for path in paths:
                    item['pins'][str(path)] = driver.sha(path)
            reference = driver.load(item['D2_geometry'])['fit']['sampling_floor']
            mapping = reference['canonical_mapping']
            if (mapping.get('mode') != 'legacy' or mapping.get('visible_uv', {}).get('enabled') or
                    mapping.get('u_offset', 0) != 0 or
                    not np.array_equal(mapping.get('rotation'), np.eye(3)) or
                    mapping['resolution_hw'] != item['normalized_training']['background_plane_kwargs']['color_mask_resolution']):
                raise ValueError('Reference must use the matched canonical regular mapping')
            item.update(D0_geometry=str(geometry), D0_runs=runs, sphere_reference=reference['sphere'],
                canonical_mapping=mapping, estimated_training_seconds=max(times) * 1.35)
            item['initial_depth_floor_gate'] = initial_gate(item)
            found[label] = item
    if set(found) != set(INTERVALS):
        raise ValueError('Missing D0 sources: ' + str(sorted(set(INTERVALS) - set(found))))
    plan['intervals'] = [found[label] for label in INTERVALS]
    for path in (Path(__file__).resolve(), KERNEL, driver.REPO / 'nag/preparation/legacy_floor.py',
                 *(driver.REPO / name for name in LAUNCHERS)):
        plan['diagnostic_code'][str(path)] = driver.sha(path)
    return plan


def stage(plan, item, root):
    signature = dict(plan=driver.sha(Path(root) / 'manifest.json'), source=item['pins'], recipe=RECIPE)
    return driver.Stage(Path(root) / item['label'], 'depth_floor', signature)


def matrix(plan, root, require_prepared=False):
    cells = []
    for item in plan['intervals']:
        if (Path(root) / item['label'] / 'depth_floor/stage.json').exists():
            geometry = str(stage(plan, item, root).result)
            same = driver.same_runtime(driver.load(geometry)['fit'], driver.load(item['D0_geometry'])['fit'])
        elif require_prepared:
            raise ValueError('Prepare depth + floor first: ' + item['label'])
        else:
            geometry = str(Path(root) / item['label'] / 'depth_floor/attempt_001/ellipsoid.json')
            same = item['initial_depth_floor_gate']['preserve_expected']
        for repeat in (1, 2, 3):
            cells.append(dict(id=f'autowork_{item["label"]}_D0_floor_regular_seed42_repeat{repeat:02d}',
                label=item['label'], recipe=TREATMENT, repeat=repeat, seed=42, mode='regular', geometry=geometry,
                geometry_sha256=driver.sha(geometry) if Path(geometry).is_file() else None,
                reused_run=item['D0_runs'][repeat - 1] if same else None,
                reuse_reason='Exact float32 D0/depth-floor runtime geometry equivalence' if same else None))
    return cells


def refine(pairs, geometry, sphere, mapping, size, config=CONFIG):
    if config.reprojection_weight != 0. or config.depth_weight != .15 or config.radius_excess_weight != 0.:
        raise ValueError('Depth + floor must have exactly zero reprojection and depth weight 0.15')
    return fit_sampling_floor(pairs, geometry, sphere, mapping, size, config,
        max_sampling=4096, floor_threshold=.5, preserve_zero_floor=True,
        compactness_weight=0., axis_multistart=False)


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
                print('SKIP prepared depth + floor ' + item['label'], flush=True)
                continue
            out = current.attempt()
            source = Path(item['D0_geometry']); original = driver.load(source)
            geometry = driver._source_geometry(original)
            with np.load(item['pairs'], allow_pickle=False) as archive:
                pairs = dict(archive)
            print('DEPTH + FLOOR ' + item['label'], flush=True)
            fit = refine(pairs, geometry, item['sphere_reference'], item['canonical_mapping'], item['size'])
            full = driver.evaluate_geometry(pairs, fit['geometry'], np.flatnonzero(pairs['visibility'] == 1),
                                           np.arange(len(pairs['source_points'])), CONFIG)
            require_safe(full)
            before = item['initial_depth_floor_gate']['full_metrics']['weighted_data_loss']
            if full['weighted_data_loss'] > before * 1.02 + 1e-6:
                raise ValueError('Full-observation depth-only degradation exceeds the 2% budget')
            if not np.array_equal(fit['geometry']['rotation'], geometry['rotation']):
                raise ValueError('Depth-floor refinement must keep the source rotation fixed')
            if item['initial_depth_floor_gate']['preserve_expected']:
                if (not fit['sampling_floor'].get('zero_floor_preserved') or
                        any(not np.array_equal(fit['geometry'][key], geometry[key]) for key in geometry)):
                    raise ValueError('A safe exact-zero-floor D0 geometry must be preserved exactly')
            kernel_version = fit['regularizer_version']
            fit['regularizer_version'] = 'ablation_depth_sampling_floor_v1'
            driver._copy_inspection_assets(source, out)
            driver._write_diagnostics(out, fit, item['label'], source, Path(item['prepared']))
            for old, new in [('reprojection_fit_diagnostics.json', 'depth_floor_fit_diagnostics.json'),
                             ('reprojection_fit_comparison.csv', 'depth_floor_fit_comparison.csv')]:
                (out / old).replace(out / new)
            doc = driver._document(original, fit, source, Path(item['prepared']), item['label'])
            doc.pop('ablation', None)
            doc.pop('baseline', None)
            doc['fit'].update(mode='depth_sampling_floor_proxy',
                method='Matched depth-only geometry plus frozen legacy sampling floor; reprojection cost exactly zero',
                sampling_floor=fit['sampling_floor'])
            doc['depth_floor_fit'] = doc.pop('reprojection_fit')
            doc['depth_floor_fit'].update(floor_kernel_version=kernel_version,
                optimization_target='0.15 * robust log-depth loss + sampling floor; pixel errors are diagnostic only')
            doc['point_selection'] = 'Cached FRONT-background source rays; original visibility/pair feasibility checks retained; no semantic static tracking'
            doc['ray_depth']['selection'] = 'Distinct cached source rays with depth weight 0.15 and zero reprojection weight'
            doc['files'].pop('reprojection_fit_diagnostics_json', None)
            doc['files'].pop('reprojection_fit_comparison_csv', None)
            doc['files'].update(depth_floor_diagnostics_json='depth_floor_fit_diagnostics.json',
                depth_floor_comparison_csv='depth_floor_fit_comparison.csv',
                sampling_floor_diagnostics_json='sampling_floor_diagnostics.json')
            same = driver.same_runtime(original['fit'], doc['fit'])
            doc['ablation'] = dict(recipe=TREATMENT, source_D0_geometry=str(source),
                source_D0_sha256=driver.sha(source), full_diagnostics=full,
                same_runtime_as_D0=same, preserve_expected=item['initial_depth_floor_gate']['preserve_expected'],
                kernel_sha256=KERNEL_SHA, training_launched=False)
            driver.write_json(out / 'ellipsoid.json', driver.jsonable(doc))
            driver.write_json(out / 'sampling_floor_diagnostics.json', driver.jsonable(fit['sampling_floor']))
            from export_background_geometry_meshes import mesh_surface, write_mesh_ply, ELLIPSOID_BLUE
            vertices, faces = mesh_surface(fit['geometry']['center'], fit['geometry']['radii'], fit['geometry']['rotation'], 48, 96)
            write_mesh_ply(out / 'ellipsoid.ply', vertices, faces, ELLIPSOID_BLUE)
            current.complete(out / 'ellipsoid.json', out)
            print('INSPECT ' + str(out / 'ellipsoid.ply') + ('; reuse D0 trainings' if same else '; three new trainings'), flush=True)
        driver.write_json(root / 'matrix.json', dict(cells=matrix(plan, root, True),
                          plan_sha256=driver.sha(root / 'manifest.json')))


def show(plan, cells):
    total = 0.
    for item in plan['intervals']:
        new = sum(c['reused_run'] is None for c in cells if c['label'] == item['label'])
        hours = new * item['estimated_training_seconds'] / 3600
        total += hours
        print(f'{item["label"]}: full-ray floor loss={item["initial_depth_floor_gate"]["floor"]["loss"]:.9g}; '
              f'{new} new / {3-new} reused; buffered estimate {hours:.2f} h', flush=True)
    print(f'TOTAL at most {sum(c["reused_run"] is None for c in cells)} new / {len(cells)} cells; '
          f'{total:.2f} h buffered; reprojection weight exactly zero.', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('audit', 'prepare', 'train', 'summary'))
    p.add_argument('--d0-training-roots', nargs='+', type=Path, default=DEFAULT_SOURCES)
    p.add_argument('--preparation-root', type=Path, default=DEFAULT_PREP)
    p.add_argument('--training-root', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--intervals', nargs='+', choices=INTERVALS)
    p.add_argument('--max-hours', type=float, default=8.)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    args = p.parse_args(argv)
    if args.summary:
        args.stage = 'summary'
    if args.intervals and args.stage not in ('train', 'summary'):
        p.error('--intervals selects training/status; prepare all seven intervals first')
    if not np.isfinite(args.max_hours) or args.max_hours <= 0:
        p.error('Positive finite hours required')
    protected = [*DEFAULT_SOURCES, *args.d0_training_roots, driver.DEFAULT_PREP,
        driver.DEFAULT_PREP.with_name('autowork_remaining_ego_loss_ablations_v1')]
    for name in ('autowork_paper_pca_baselines_v1', 'autowork_paper_pca_shared_safety_v1', 'autowork_paper_raw_pca_v1'):
        protected.extend([driver.REPO / 'runs' / name, DEFAULT_PREP.with_name(name)])
    for root in (args.preparation_root, args.training_root):
        resolved = root.resolve()
        if not root.name.startswith('autowork_') or resolved == driver.REPO.resolve():
            p.error('Use separate autowork_ output namespaces')
        if any(resolved == old.resolve() or old.resolve() in resolved.parents or resolved in old.resolve().parents for old in protected):
            p.error('Keep source and completed experiment roots read-only')
    prep, train = args.preparation_root.resolve(), args.training_root.resolve()
    if prep == train or prep in train.parents or train in prep.parents:
        p.error('Preparation/training roots must be separate, not nested')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = inventory(args.d0_training_roots)
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
