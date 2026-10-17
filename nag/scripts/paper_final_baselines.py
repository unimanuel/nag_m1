"""Matched non-ego paper baselines; audits and dry runs never launch training.

Historical final configs remain untouched. Ellipsoid fitting is deliberately
not inferred from an archival PCA fit or from an ego interval's geometry.
"""
import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag/scripts'))
from ego_repeatability import METRICS
from paper_plane_baselines import (PLANE_TYPE, SPHERE_TYPE, audit_yaml_family,
                                   config_differences, config, basic_completed)
from paper_sphere_baselines import sha, duration_seconds, code_snapshot
from diagnostics.parallax_io import diagnostic
from preparation.jobs import training_worker_override, assert_training_worker_policy

TEMPLATE = REPO / 'config/waymo/large-ego/segment-2547899409721197155_0_37_large-undistorted.yaml'
CONFIG_ROOT = REPO / 'config/waymo/autowork_paper-final'
SEGMENTS = ('W125', 'W141', 'W203', 'W324', 'W344', 'W952', 'W975')
TREATMENTS = ('plane', 'sphere_regular', 'sphere_visuv')
SUFFIX = {'plane': '_bgplane', 'sphere_regular': '', 'sphere_visuv': '_sph_visuv'}
UV = {'enabled': True, 'margin_uv': [.05, .05], 'interior_samples_per_axis': 9}


def intervals():
    result = {}
    for path in sorted((REPO / 'config/waymo/final').glob('*_large-undistorted.yaml')):
        c = config(path)
        label = c['name'].split('_large')[0]
        segment = c['data_path'].rsplit('segment-', 1)[1]
        _, first, last = label.split('_')
        frames = c['frame_indices_filter']
        if ((frames['start'], frames['stop'], frames.get('step') or 1)
                != (int(first), int(last) + 1, 1) or label in result
                or label.split('_')[0] != 'W' + segment[:3]
                or path.stem != f'segment-{segment}_{first}_{last}_large-undistorted'):
            raise ValueError(f'Invalid or duplicate final interval: {path}')
        result[label] = {'segment': segment, 'first': int(first), 'stop': int(last) + 1,
                         'stem': path.stem, 'original': str(path), 'data_path': c['data_path']}
    if len(result) != 25 or {k.split('_')[0] for k in result} != set(SEGMENTS):
        raise ValueError('Expected the established 25 final intervals in seven parent segments')
    return result


def expected_config(item, label, treatment):
    settings = copy.deepcopy(config(TEMPLATE))
    settings.update(data_path=item['data_path'],
                    frame_indices_filter={'__class__': 'slice', 'start': item['first'],
                                          'stop': item['stop'], 'step': None},
                    name=label + '_large-undistorted_paper_final' + SUFFIX[treatment],
                    save_intermediate_checkpoints=False)
    if treatment == 'plane':
        settings['background_plane_type'] = PLANE_TYPE
    elif treatment == 'sphere_visuv':
        settings['background_plane_kwargs']['sphere_visible_uv_texture'] = copy.deepcopy(UV)
    return settings


def yaml_audit(items):
    for label, item in items.items():
        family = {}
        for treatment in TREATMENTS:
            path = CONFIG_ROOT / (item['stem'] + SUFFIX[treatment] + '.yaml')
            actual = config(path)
            expected = expected_config(item, label, treatment)
            if actual != expected:
                raise ValueError(f'Unmatched paper config {path}: {config_differences(expected, actual)}')
            family[treatment] = actual
        audit_yaml_family(family['sphere_regular'], family['sphere_visuv'], family['plane'])


def data_audit(item, data_root):
    """Boxes are per object, NOT per frame. Optional mask filters may be absent."""
    root = data_root / ('segment-' + item['segment'])
    frames = set(range(item['first'], item['stop']))
    patterns = {'images_undistorted': r'(\d+)\.(png|jpg|jpeg)',
                'depth': r'(\d+)\.tiff', 'masks': r'img_(\d+)_ov_\d+\.png'}
    inputs, missing = [], {}
    for directory, pattern in patterns.items():
        folder = root / directory
        paths = [p for p in folder.iterdir()] if folder.is_dir() else []
        selected = [p for p in paths if (m := re.fullmatch(pattern, p.name)) and int(m[1]) in frames]
        found = {int(re.fullmatch(pattern, p.name)[1]) for p in selected}
        if frames - found:
            missing[directory] = sorted(frames - found)
        inputs.extend(selected)
    # Include all object-level boxes, calibration and mapping/filter evidence.
    boxes = sorted((root / 'boxes').glob('*.json'))
    if not boxes:
        missing['boxes_directory'] = [str(root / 'boxes')]
    inputs.extend(boxes)
    for name in ['waymo_pose_config_no_distortion.json', 'box_id_mapping.json']:
        path = root / name
        if not path.is_file():
            missing[name] = [str(path)]
        else:
            inputs.append(path)
    optional = root / 'mask_indices_filter.json'
    if optional.exists():
        inputs.append(optional)
    camera_path = root / 'waymo_pose_config_no_distortion.json'
    movement = None
    if camera_path.is_file():
        camera = json.loads(camera_path.read_text())
        poses = np.asarray(camera['position'], dtype=float)
        if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < item['stop']
                or not np.isfinite(poses).all() or np.any(camera.get('lens_distortion', [0] * 5))):
            missing['calibration'] = ['Invalid/incomplete undistorted camera poses']
        else:
            centers = poses[item['first']:item['stop'], :3, 3]
            movement = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    snapshot = [{'path': str(p.absolute()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for p in sorted(set(inputs))]
    return {'ready': not missing, 'missing': missing, 'camera_path_length_m': movement,
            'optional_mask_filter_present': optional.exists(), 'inputs': snapshot}


def historical_runs(items):
    """Read-only inventory; old metrics are never reused as matched results."""
    result = {label: [] for label in items}
    for root in sorted((REPO / 'runs').iterdir()):
        if not root.is_dir() or root.name.startswith('autowork_'):
            continue
        for path in sorted(root.iterdir()):
            match = re.search(r'W\d+_\d+_\d+(?=_)', path.name)
            if not match or match[0] not in items or not (path / 'training_config.yaml').is_file():
                continue
            label, item = match[0], items[match[0]]
            c = config(path / 'training_config.yaml')
            sl = c['frame_indices_filter']
            exact = (sl['start'], sl['stop'], sl.get('step') or 1) == (item['first'], item['stop'], 1)
            metrics = basic_completed(path, {'frame_count': item['stop'] - item['first']}) if exact else None
            try:
                seconds = duration_seconds(path / 'exit_000.txt') if metrics else None
            except (OSError, KeyError, ValueError):
                seconds = None
            bg = c.get('background_plane_kwargs', {})
            result[label].append({'path': str(path.resolve()), 'complete': metrics is not None,
                'class': c['background_plane_type'], 'sampler': c.get('sampler_kwargs'),
                'texture_resolution': bg.get('color_mask_resolution'),
                'visible_uv': bg.get('sphere_visible_uv_texture'), 'seconds': seconds,
                'matched_paper_result': False})
    return result


def training_context():
    data_override = os.environ.get('DATA_PATH')
    if data_override and (REPO / data_override).resolve() != (REPO / 'data/datasets').resolve():
        raise ValueError('DATA_PATH overrides the dataset root inspected by this paper launcher')
    current = code_snapshot()
    reference = REPO / 'runs/autowork_paper_sphere_baselines_part1of5/manifest.json'
    old = json.loads(reference.read_text())['context']
    from preparation.compatibility import allowed_preparation_change
    changed = [p for p, h in old['code'].items() if current.get(p) != h
               and not allowed_preparation_change(p, h, current.get(p))]
    versions = {k: importlib.metadata.version(k) for k in old['versions']}
    environment = {k: os.environ.get(k) for k in old['environment']}
    if changed or versions != old['versions'] or sys.executable != old['executable'] or environment != old['environment']:
        raise ValueError(f'Use the completed ego training code/environment/interpreter. Changed code: {changed}')
    for p in sorted(REPO.glob('run_paper_final_baselines*.sh')):
        current[p.name] = sha(p)
    files = [TEMPLATE] + sorted(CONFIG_ROOT.glob('*.yaml'))
    files += sorted((REPO / 'config/waymo/final').glob('*_large-undistorted.yaml'))
    if old.get('env_file'):
        env = old['env_file']
        if sha(env['path']) != env['sha256']:
            raise ValueError('.env changed from the completed ego sphere cohort')
        files.append(Path(env['path']))
    return {'schema': 1, 'code': current, 'versions': versions, 'executable': sys.executable,
            'environment': environment, 'files': {str(p.resolve()): sha(p) for p in files},
            'ego_reference_manifest': str(reference), 'ego_reference_manifest_sha256': sha(reference)}


def assert_context(context):
    current = training_context()
    if current != context:
        raise ValueError('Paper cohort code/configuration/environment changed; do not mix treatments')


def make_jobs(items, history):
    jobs = []
    configured_workers = config(TEMPLATE)['num_workers']
    for repeat in range(1, 4):
        for label, item in items.items():
            # Older sampler timing is only an estimate. Never promise a job cap.
            timings = [r['seconds'] for r in history[label] if r['complete'] and r['seconds']
                       and r['texture_resolution'] == [960, 1920]
                       and r['class'] in (SPHERE_TYPE, PLANE_TYPE)]
            seconds = max(timings or [max(1200, (item['stop'] - item['first']) * 60)])
            for treatment in (TREATMENTS if repeat % 2 else TREATMENTS[::-1]):
                cfg = str(CONFIG_ROOT / (item['stem'] + SUFFIX[treatment] + '.yaml'))
                override = training_worker_override(cfg, item['segment'], configured_workers)
                worker_tag = f'_workers{override["num_workers"]}' if override else ''
                jobs.append({'id': f'autowork_{label}_{treatment}_same_seed_seed42_repeat{repeat:02d}{worker_tag}',
                    'label': label, 'treatment': treatment, 'repeat': repeat, 'seed': 42,
                    'frame_count': item['stop'] - item['first'], 'first': item['first'], 'stop': item['stop'],
                    'segment': item['segment'], 'config': cfg,
                    'historical_seconds': seconds, 'buffered_seconds': seconds * 1.35, **override})
    return jobs


def worker_output_suffix(items):
    configured = config(TEMPLATE)['num_workers']
    counts = {training_worker_override(CONFIG_ROOT / (item['stem'] + '.yaml'),
              item['segment'], configured).get('num_workers', configured) for item in items.values()}
    if counts == {configured}:
        return ''
    return f'_{next(iter(counts))}workers' if len(counts) == 1 else '_mixedworkers'


def assert_job_worker_policy(job):
    assert_training_worker_policy(job, job['segment'], config(job['config'])['num_workers'])


def command(job, output):
    cmd = [sys.executable, str(REPO / 'nag/scripts/run_nag.py'), '--config-path', job['config'],
           '--background-plane-type', PLANE_TYPE if job['treatment'] == 'plane' else SPHERE_TYPE,
           '--seed', '42', '--name', job['id'], '--output-path', str(output)]
    if job['treatment'] != 'plane':
        cmd.append('--background-sphere-radius-from-visible-plane-centers')
    if 'num_workers' in job:
        cmd.extend(['--num-workers', str(job['num_workers'])])
    return cmd


def completed_job(path, job):
    metrics = basic_completed(path, job)
    if metrics is None:
        return None
    c, reference = config(path / 'training_config.yaml'), config(job['config'])
    if 'num_workers' in job:
        reference['num_workers'] = job['num_workers']
    reference['name'] = job['id']
    reference['used_runner_type'] = 'nag.run.nag_runner.NAGRunner'
    reference['object_rigid_control_points'] = job['frame_count']
    reference['plane_init_strategy_kwargs']['position_spline_fitting'] = reference['object_position_spline_approximation']
    start = 0
    for phase in reference['training_phases']:
        phase['start_at'] = start
        start += phase['length']
    def normalized(value):
        if isinstance(value, dict):
            if value.get('__class__') == 'type':
                return value['value']
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        return value
    # Compare every field that the loader does not resolve/expand at runtime.
    resolved = {'data_path', 'images_path', 'masks_path', 'depths_path', 'boxes_path',
                'boxes_object_id_mask_mapping_path', 'mask_ids_filter_path',
                'pinhole_camera_config', 'output_path', 'experiment_datetime', 'num_objects'}
    for key, value in reference.items():
        if key not in resolved and normalized(c.get(key)) != value:
            raise ValueError(f'Completed settings differ: {job["id"]}, {key}')
    def stored_path(value):
        while isinstance(value, dict):
            value = value['value']
        return Path(value)
    data_path = stored_path(c['data_path'])
    if (data_path.name != 'segment-' + job['segment'] or
            (REPO / data_path).resolve() != (REPO / 'data/datasets/waymo' / ('segment-' + job['segment'])).resolve()):
        raise ValueError(f'Completed settings refer to another segment: {path}')
    if c.get('background_ellipsoid_path') or c.get('background_sphere_radius_override') is not None:
        raise ValueError(f'Unexpected geometry override: {path}')
    for key, subpath in [('images_path', 'images_undistorted'), ('masks_path', 'masks'),
                         ('depths_path', 'depth'), ('boxes_path', 'boxes'),
                         ('boxes_object_id_mask_mapping_path', 'box_id_mapping.json')]:
        if stored_path(c[key]) != data_path / subpath:
            raise ValueError(f'Unexpected effective input path {key}: {path}')
    camera = c['pinhole_camera_config']
    raw_camera = json.loads((REPO / data_path / 'waymo_pose_config_no_distortion.json').read_text())
    if (not np.array_equal(np.asarray(camera['position']), np.asarray(raw_camera['position']))
            or np.any(camera.get('lens_distortion', [0] * 5))):
        raise ValueError(f'Unexpected effective calibration: {path}')
    for key in ['focal_length', 'resolution', 'center']:
        if key in raw_camera and camera.get(key) != raw_camera[key]:
            raise ValueError(f'Unexpected effective camera {key}: {path}')
    mask_filter = REPO / data_path / 'mask_indices_filter.json'
    if (mask_filter.is_file() and (c.get('mask_ids_filter_path') is None or
            stored_path(c['mask_ids_filter_path']) != data_path / 'mask_indices_filter.json')):
        raise ValueError(f'Unexpected effective mask filtering: {path}')
    if job['treatment'] != 'plane':
        geometry = diagnostic(path / 'initial/background_diagnostics/sphere_geometry.txt')
        if (geometry['radius_source'] != 'visible_plane_centers' or
                bool(geometry['visible_uv_texture']['enabled']) != (job['treatment'] == 'sphere_visuv')):
            raise ValueError(f'Wrong sphere radius/texture allocation policy: {path}')
    elif c.get('background_sphere_radius_from_visible_plane_centers'):
        raise ValueError(f'Plane has an active sphere radius policy: {path}')
    return metrics


def summarize(root, jobs):
    groups, count = {}, 0
    for job in jobs:
        good = [r for p in (root / job['id']).glob('attempt_*') if (r := completed_job(p, job)) is not None]
        if len(good) > 1:
            raise ValueError(f'Multiple completed attempts: {job["id"]}')
        if good:
            count += 1
            groups.setdefault((job['label'], job['treatment']), []).append(good[0])
    print(f'Completed {count}/{len(jobs)} matched final jobs')
    for (label, treatment), records in sorted(groups.items()):
        values = []
        for metric in METRICS:
            v = [r[metric] for r in records]
            sd = f'{statistics.stdev(v):.6f}' if len(v) > 1 else 'n/a'
            values.append(f'{metric}={statistics.mean(v):.6f} (SD {sd})')
        print(f'{label} {treatment}: n={len(records)}; ' + '; '.join(values))


def scheduling_seconds(root, job, jobs):
    """Use observed matched repeat durations once available, without rewriting provenance."""
    timings = []
    for candidate in jobs:
        if (candidate['label'], candidate['treatment']) != (job['label'], job['treatment']):
            continue
        for path in (root / candidate['id']).glob('attempt_*'):
            if (path / 'exit_000.txt').is_file() and completed_job(path, candidate) is not None:
                timings.append(duration_seconds(path / 'exit_000.txt'))
    return max(timings) * 1.35 if timings else job['buffered_seconds']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segment', choices=SEGMENTS)
    p.add_argument('--intervals', nargs='+', help='Explicit subset; never silently drop an unready interval')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--historical', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--max-hours', type=float, default=9.)
    p.add_argument('--allow-long-job', action='store_true', help='Explicitly allow one estimated-over-budget job per invocation')
    args = p.parse_args()
    args.data_root = REPO / 'data/datasets/waymo'
    if args.segment and args.intervals:
        p.error('Choose segment or explicit intervals, not both')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        p.error('max-hours must be positive and finite')
    all_items = intervals()
    if args.intervals and (len(set(args.intervals)) != len(args.intervals) or set(args.intervals) - set(all_items)):
        p.error('Unknown or duplicate final interval')
    if not args.audit_only and not args.segment and not args.intervals:
        p.error('Choose one segment or explicit intervals; no automatic whole-suite training')
    items = {k: v for k, v in all_items.items() if (not args.segment or k.split('_')[0] == args.segment)
             and (not args.intervals or k in args.intervals)}
    selection = args.segment or '__'.join(items)
    suffix = worker_output_suffix(items) if args.output_root is None else ''
    root = (args.output_root or REPO / ('runs/autowork_paper_final_baselines_' + selection + suffix)).resolve()
    manifest_path = root / 'manifest.json'
    if args.summary:
        summarize(root, json.loads(manifest_path.read_text())['jobs'])
        return
    yaml_audit(all_items)
    audits = {}
    for label, item in items.items():
        audit = audits[label] = data_audit(item, args.data_root)
        movement = audit['camera_path_length_m']
        configured = config(TEMPLATE)['num_workers']
        workers = training_worker_override(CONFIG_ROOT / (item['stem'] + '.yaml'),
                                           item['segment'], configured).get('num_workers', configured)
        print(f'{label}: {"READY" if audit["ready"] else "BLOCKED"}; workers={workers}; camera path {movement} m; '
              f'missing={audit["missing"]}', flush=True)
    history = historical_runs(items) if args.historical or not manifest_path.exists() else None
    if args.historical:
        for label, rows in history.items():
            print(f'{label}: {sum(r["complete"] for r in rows)} completed / {len(rows)} historical runs; none reused')
            for row in rows:
                print('  ' + json.dumps(row))
    if args.audit_only:
        training_context()
        print(f'75 YAMLs match the selected ego W254 0-37 reference; '
              f'{sum(a["ready"] for a in audits.values())}/{len(items)} intervals ready. '
              'Read-only audit: no files or trainings created.')
        return
    if not root.name.startswith('autowork_'):
        p.error('New output root must begin with autowork_')
    if not all(a['ready'] for a in audits.values()):
        raise ValueError('Selected intervals have missing inputs. Restore them or choose an explicit ready --intervals subset.')
    context = training_context()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['context'] != context or manifest['labels'] != list(items)
                or manifest['data'] != audits or manifest['data_root'] != str(args.data_root.resolve())):
            raise ValueError('Existing final cohort differs; preserve it and restore its inputs/settings')
        jobs = manifest['jobs']
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('Nonempty output without owned manifest; never overwrite it')
        jobs = make_jobs(items, history)
        manifest = {'schema': 1, 'labels': list(items), 'jobs': jobs, 'context': context,
                    'data': audits, 'data_root': str(args.data_root.resolve()),
                    'historical_inventory': history, 'purpose': 'fresh three seed-42 repeats; no archival metric reuse'}
    for job in jobs:
        assert_job_worker_policy(job)
    effective = {label: training_worker_override(CONFIG_ROOT / (item['stem'] + '.yaml'),
                 item['segment'], config(TEMPLATE)['num_workers']).get('num_workers', config(TEMPLATE)['num_workers'])
                 for label, item in items.items()}
    print('Effective training workers: ' + json.dumps(effective, sort_keys=True), flush=True)
    estimate = sum(j['buffered_seconds'] for j in jobs) / 3600
    print(f'{len(jobs)} jobs, full selection buffered historical estimate {estimate:.2f} h. '
          f'Invocation scheduling budget {args.max_hours} h; active jobs are never killed.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(command(job, root / job['id'] / 'attempt_001')))
        print('Dry run: no files or training created.')
        return
    lock_root = REPO / 'runs/autowork_paper_final_baselines_lock'
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / 'gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another final batch is active; run sequentially') from None
        cohort = lock_root / 'cohort.json'
        if cohort.exists() and json.loads(cohort.read_text()) != context:
            raise ValueError('Another final segment used different code/configs/environment')
        if not cohort.exists():
            cohort.write_text(json.dumps(context, indent=2) + '\n')
        root.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        elif json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Manifest changed while acquiring the final-batch lock')
        start, launched = time.monotonic(), 0
        for job in jobs:
            attempts = sorted((root / job['id']).glob('attempt_*'))
            good = [q for q in attempts if completed_job(q, job) is not None]
            if len(good) > 1:
                raise ValueError(f'Multiple completed attempts: {job["id"]}')
            if good:
                print('SKIP completed ' + job['id'], flush=True)
                continue
            elapsed = time.monotonic() - start
            cost = scheduling_seconds(root, job, jobs)
            if elapsed + cost > args.max_hours * 3600:
                if not (args.allow_long_job and launched == 0 and cost > args.max_hours * 3600):
                    print('BUDGET STOP: rerun this script to continue. If even one job exceeds the '
                          'estimate budget, inspect timings and explicitly use --allow-long-job.', flush=True)
                    break
                print('WARNING: explicitly starting one estimated-over-budget job.', flush=True)
            assert_context(context)
            assert_job_worker_policy(job)
            if data_audit(items[job['label']], args.data_root) != audits[job['label']]:
                raise ValueError('Training inputs changed during the batch')
            number = max([int(q.name.split('_')[-1]) for q in attempts] or [0]) + 1
            output = root / job['id'] / f'attempt_{number:03d}'
            print(f'Next job buffered estimate {cost / 3600:.2f} h; ' + shlex.join(command(job, output)), flush=True)
            subprocess.run(command(job, output), cwd=REPO, check=True)
            launched += 1
            if completed_job(output, job) is None:
                raise RuntimeError(f'Training did not produce all intended epoch-80 outputs: {output}')
        summarize(root, jobs)


if __name__ == '__main__':
    main()
