"""Matched plane repeats, with a read-only audit of the three background families."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
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
from ego_repeatability import completed, METRICS
from paper_sphere_baselines import (sha, duration_seconds, completed_sphere,
                                   check_training_compatibility, code_snapshot as sphere_snapshot,
                                   SPHERE_TYPE)

PLANE_TYPE = 'nag.model.view_dependent_background_image_plane_scene_node_3d.ViewDependentBackgroundImagePlaneSceneNode3D'
ELLIPSOID_TYPE = 'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D'
BATCHES = {
    'part1of2': ['W191_0_40', 'W191_40_128', 'W191_128_198', 'W254_0_37'],
    'part2of2': ['W254_37_81', 'W254_86_137', 'W254_139_195'],
}
LABELS = [label for group in BATCHES.values() for label in group]
BUFFER = 1.35
LOCK_ROOT = REPO / 'runs/autowork_paper_plane_baselines_lock'
UV_SETTINGS = {'enabled': True, 'margin_uv': [.05, .05], 'interior_samples_per_axis': 9}
# Only known, intentional background selection and run/storage metadata may differ.
SOURCE_DIFFERENCES = {
    '/background_plane_type', '/background_ellipsoid_path',
    '/background_ellipsoid_scale_margin', '/background_sphere_radius_from_visible_plane_centers',
    '/name', '/output_path/value', '/experiment_datetime/value', '/save_intermediate_checkpoints',
}


def config_differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((config_differences(a.get(k), b.get(k), prefix + '/' + k)
                    for k in sorted(set(a) | set(b))), [])
    return [] if a == b else [(prefix, a, b)]


def audit_yaml_family(regular, visible, plane):
    """Deep equality outside the precise treatment/name changes, not a short key list."""
    if regular.get('background_plane_type') != SPHERE_TYPE or visible.get('background_plane_type') != SPHERE_TYPE:
        raise ValueError('Regular/Visible-UV reference must use the sphere class')
    if plane.get('background_plane_type') != PLANE_TYPE:
        raise ValueError('Plane YAML must use the view-dependent plane class')
    if plane.get('background_plane_kwargs', {}).get('sphere_visible_uv_texture') is not None:
        raise ValueError('Plane YAML must not contain a Visible-UV treatment')
    differences = config_differences(regular, plane)
    if any(path not in {'/background_plane_type', '/name'} for path, _, _ in differences):
        raise ValueError(f'Unmatched plane YAML: {differences}')
    if regular['background_plane_kwargs'].get('sphere_visible_uv_texture') is not None:
        raise ValueError('Regular sphere must not have a Visible-UV override')
    if visible['background_plane_kwargs'].get('sphere_visible_uv_texture') != UV_SETTINGS:
        raise ValueError('Unexpected Visible-UV settings')
    normalized = copy.deepcopy(visible)
    normalized['background_plane_kwargs'].pop('sphere_visible_uv_texture')
    uv_differences = config_differences(regular, normalized)
    if any(path != '/name' for path, _, _ in uv_differences):
        raise ValueError(f'Unmatched Visible-UV YAML: {uv_differences}')
    if regular['background_plane_kwargs'].get('color_mask_resolution') != [960, 1920]:
        raise ValueError('All backgrounds must use the established 960x1920 nominal texture budget')
    for settings in (regular, visible, plane):
        if (settings.get('seed') != 42 or settings.get('max_epochs') != 80
                or settings.get('deterministic', False) or settings.get('checkpoint_path')
                or settings.get('checkpoint_tracker_path') or settings.get('save_intermediate_checkpoints', False)):
            raise ValueError('Requires fresh seed-42 nondeterministic epoch-80, final-only training')
    return {'plane_vs_regular_differences': [p for p, _, _ in differences],
            'visible_uv_settings': UV_SETTINGS, 'nominal_texture_resolution_hw': [960, 1920]}


def audit_effective_settings(settings, reference, family):
    expected = {'plane': PLANE_TYPE, 'sphere': SPHERE_TYPE, 'ellipsoid': ELLIPSOID_TYPE}[family]
    if (settings.get('background_plane_type') != expected or settings.get('seed') != 42
            or settings.get('max_epochs') != 80 or settings.get('deterministic', False)
            or settings.get('checkpoint_path') or settings.get('checkpoint_tracker_path')):
        raise ValueError(f'Incorrect effective {family} training settings')
    if family == 'ellipsoid':
        if not settings.get('background_ellipsoid_path') or settings.get('background_ellipsoid_scale_margin') != 0 \
                or settings.get('background_ellipsoid_texture_orientation') != 'legacy':
            raise ValueError('Reference must be the fixed legacy-pole zero-margin ellipsoid')
    elif family == 'sphere':
        if not settings.get('background_sphere_radius_from_visible_plane_centers'):
            raise ValueError('Reference sphere must use visible foreground plane centers')
    else:
        if (settings.get('save_intermediate_checkpoints', False)
                or settings.get('background_sphere_radius_from_visible_plane_centers')
                or settings.get('background_ellipsoid_path')
                or settings.get('background_sphere_radius_override') is not None):
            raise ValueError('Plane training must be final-only and have no curved-geometry override')
    unexpected = [d for d in config_differences(settings, reference) if d[0] not in SOURCE_DIFFERENCES]
    if unexpected:
        raise ValueError(f'Unmatched effective {family} settings: {unexpected}')


def code_snapshot():
    result = sphere_snapshot()
    for name in ['run_paper_plane_baselines.sh', 'run_paper_plane_baselines_part1.sh',
                 'run_paper_plane_baselines_part2.sh']:
        result[name] = sha(REPO / name)
    return result


def successful_attempt(root, job, validator):
    matches = [(p, validator(p, job)) for p in sorted((root / job['id']).glob('attempt_*'))]
    matches = [(p, v) for p, v in matches if v is not None]
    if len(matches) != 1:
        raise ValueError(f'Expected one completed source attempt: {job["id"]}, got {len(matches)}')
    return matches[0]


def basic_completed(path, job):
    result = completed(path, job['frame_count'])
    if result is None or not (path / 'exit_000.txt').read_text().startswith('Exit code: 0.'):
        return None
    return result


def config(path):
    return yaml.safe_load(Path(path).read_text())['NAGConfig']


def load_audit(current):
    files, sphere_jobs, settings, timings, contexts = set(), {}, {}, {}, []
    def record_source(path):
        files.update([path / 'training_config.yaml', path / 'exit_000.txt'])
        files.update(path / 'tracker' / f'train_epoch_Mean{m}.csv' for m in METRICS)

    for part in range(1, 6):
        root = REPO / f'runs/autowork_paper_sphere_baselines_part{part}of5'
        manifest_path = root / 'manifest.json'
        files.add(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        old = manifest['context']
        contexts.append(old)
        changed = [p for p, h in old['code'].items() if current.get(p) != h]
        if changed:
            raise ValueError(f'Training code differs from completed sphere cohort: {changed}')
        for job in manifest['jobs']:
            key = job['label'], job['mode'], job['repeat']
            if key in sphere_jobs:
                raise ValueError(f'Duplicate sphere job: {key}')
            if sha(job['config']) != job['config_sha256'] or sha(job['reference_geometry']) != job['reference_geometry_sha256']:
                raise ValueError(f'Sphere input changed: {key}')
            path, _ = successful_attempt(root, job, completed_sphere)
            c = config(path / 'training_config.yaml')
            audit_effective_settings(c, c, 'sphere')
            sphere_jobs[key] = job
            settings[key] = c
            if job['mode'] == 'regular':
                timings.setdefault(job['label'], []).append({'seconds': duration_seconds(path / 'exit_000.txt'),
                                                            'path': str(path), 'kind': 'fresh_regular_sphere'})
            record_source(path)
            files.update([Path(job['config']), Path(job['reference_geometry'])])
            files.add(path / 'initial/background_diagnostics/sphere_geometry.txt')
    expected = {(label, mode, repeat) for label in LABELS for mode in ['regular', 'visuv'] for repeat in range(1, 4)}
    if set(sphere_jobs) != expected or any(c != contexts[0] for c in contexts):
        raise ValueError('Sphere cohort coverage or shared signature mismatch')
    # Check repeat and layout consistency as well as same-mode cross-family pairs.
    # The only removed Visible-UV setting is independently validated in the YAML audit.
    for (label, mode, repeat), c in settings.items():
        canonical = copy.deepcopy(c)
        if mode == 'visuv':
            if canonical['background_plane_kwargs'].get('sphere_visible_uv_texture') != UV_SETTINGS:
                raise ValueError(f'Unapproved effective Visible-UV settings: {label} repeat {repeat}')
            canonical['background_plane_kwargs'].pop('sphere_visible_uv_texture')
        audit_effective_settings(canonical, settings[label, 'regular', 1], 'sphere')
    previous = contexts[0]
    versions = {k: importlib.metadata.version(k) for k in previous['versions']}
    if versions != previous['versions'] or sys.executable != previous['executable']:
        raise ValueError('Use the same nag1 interpreter and package versions as the completed sphere cohort')
    env = {k: os.environ.get(k) for k in previous['environment']}
    if env != previous['environment']:
        raise ValueError('Training environment options differ from the completed sphere cohort')
    if previous.get('env_file'):
        item = previous['env_file']
        if sha(item['path']) != item['sha256']:
            raise ValueError('.env differs from the completed sphere cohort')
        files.add(Path(item['path']))

    seen = set()
    for part in range(1, 5):
        root = REPO / f'runs/2026-09-25_ego_poles_repeatability_part{part}of4'
        manifest_path = root / 'manifest.json'
        files.add(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        check_training_compatibility(manifest['code'], current)
        if manifest['versions'] != versions or manifest['executable'] != sys.executable:
            raise ValueError('Ellipsoid cohort interpreter/package mismatch')
        for job in manifest['jobs']:
            if job['orientation'] != 'legacy':
                continue
            key = job['label'], job['mode'], job['repeat']
            if key in seen or key not in expected or job['config_sha256'] != sphere_jobs[key]['config_sha256']:
                raise ValueError(f'Unmatched ellipsoid source: {key}')
            seen.add(key)
            if sha(job['geometry']) != job['geometry_sha256']:
                raise ValueError(f'Ellipsoid geometry changed: {key}')
            path, _ = successful_attempt(root, job, basic_completed)
            audit_effective_settings(config(path / 'training_config.yaml'), settings[key], 'ellipsoid')
            record_source(path)
            files.add(Path(job['geometry']))
    if seen != expected:
        raise ValueError('Ellipsoid comparison must contain all 42 matched legacy-pole repeats')

    references, yaml_audit = {}, {}
    for label in LABELS:
        regular_path = Path(sphere_jobs[label, 'regular', 1]['config'])
        visible_path = Path(sphere_jobs[label, 'visuv', 1]['config'])
        plane_path = regular_path.with_name(regular_path.stem + '_bgplane.yaml')
        regular, visible, plane = [config(p) for p in [regular_path, visible_path, plane_path]]
        yaml_audit[label] = audit_yaml_family(regular, visible, plane)
        _, first, last = label.split('_')
        if plane['frame_indices_filter']['start'] != int(first) or plane['frame_indices_filter']['stop'] != int(last) + 1:
            raise ValueError(f'Incorrect plane interval: {label}')
        files.add(plane_path)
        references[label] = {'config': str(plane_path), 'config_sha256': sha(plane_path),
                             'frame_count': int(last) - int(first) + 1,
                             'reference_settings': settings[label, 'regular', 1]}

    # Historical plane outputs supply timing only, never baseline metrics or settings.
    # Include both older bbox variants for conservative runtime sizing, not recipe reuse.
    for name in ['2026-08-19_ellipsoid_non_center', '2026-06-11-13-15-combi']:
        for path in sorted((REPO / 'runs' / name).glob('*bgplane*')):
            match = re.search(r'W(?:191|254)_\d+_\d+(?=_)', path.name)
            if not match or match[0] not in LABELS or not (path / 'training_config.yaml').is_file():
                continue
            label = match[0]
            if basic_completed(path, references[label]) is None:
                continue
            if config(path / 'training_config.yaml').get('background_plane_type') != PLANE_TYPE:
                continue
            timings[label].append({'seconds': duration_seconds(path / 'exit_000.txt'),
                                   'path': str(path), 'kind': 'historical_plane_timing_only'})
            files.update([path / 'exit_000.txt', path / 'training_config.yaml'])
    selected = {}
    for label, reference in references.items():
        selected[label] = max(timings[label], key=lambda x: x['seconds'])
        reference['seconds'] = selected[label]['seconds']
    context = {'schema': 1, 'purpose': '21 fresh matched plane repeats; no Visible-UV plane treatment',
               'code': current, 'versions': versions, 'executable': sys.executable,
               'python': sys.version, 'environment': env,
               'source_files': {str(p): sha(p) for p in sorted(files)},
               'audit': {'yaml_count': 21, 'completed_sphere_count': 42, 'completed_ellipsoid_count': 42,
                         'families': yaml_audit, 'timing_candidates': timings, 'selected_timings': selected}}
    return references, context


def make_jobs(batch, references):
    return [{'id': f'autowork_{label}_plane_same_seed_seed42_repeat{repeat:02d}',
             'label': label, 'mode': 'regular', 'orientation': 'planar', 'group': 'same_seed',
             'seed': 42, 'repeat': repeat, **references[label],
             'buffered_seconds': BUFFER * references[label]['seconds']}
            for repeat in range(1, 4) for label in BATCHES[batch]]


def command(job, output):
    return [sys.executable, str(REPO / 'nag/scripts/run_nag.py'), '--config-path', job['config'],
            '--background-plane-type', PLANE_TYPE, '--seed', '42', '--name', job['id'], '--output-path', str(output)]


def completed_plane(path, job):
    result = basic_completed(path, job)
    if result is not None:
        audit_effective_settings(config(path / 'training_config.yaml'), job['reference_settings'], 'plane')
    return result


def summarize(root, jobs):
    groups = {}
    count = 0
    for job in jobs:
        good = [v for p in (root / job['id']).glob('attempt_*') if (v := completed_plane(p, job)) is not None]
        if len(good) > 1:
            raise ValueError(f'Multiple complete plane attempts: {job["id"]}')
        if good:
            count += 1
            groups.setdefault(job['label'], []).append(good[0])
    print(f'Completed {count}/{len(jobs)} plane jobs')
    for label, records in sorted(groups.items()):
        print(f'{label}: n={len(records)}')
        for metric in METRICS:
            values = [r[metric] for r in records]
            sd = f'{statistics.stdev(values):.6f}' if len(values) > 1 else 'n/a'
            print(f'  {metric}: mean={statistics.mean(values):.6f}, sample_sd={sd}')


def assert_signature(context):
    if code_snapshot() != context['code']:
        raise ValueError('Code changed during the plane cohort; restore it before continuing')
    for path, digest in context['source_files'].items():
        if sha(path) != digest:
            raise ValueError(f'Comparison input/configuration changed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=list(BATCHES))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--max-hours', type=float, default=10.)
    args = parser.parse_args()
    if args.summary and args.audit_only:
        parser.error('Choose summary or audit-only, not both')
    if not args.batch and not args.audit_only:
        parser.error('--batch is required except with --audit-only')
    if not np.isfinite(args.max_hours) or not 0 < args.max_hours <= 10:
        parser.error('max-hours must be in (0, 10]')
    root = (args.output_root or REPO / f'runs/autowork_paper_plane_baselines_{args.batch}').resolve()
    manifest_path = root / 'manifest.json'
    if args.summary:
        summarize(root, json.loads(manifest_path.read_text())['jobs'])
        return
    if not root.name.startswith('autowork_'):
        parser.error('new output folder must begin with autowork_')
    started = time.monotonic()
    references, context = load_audit(code_snapshot())
    print('AUDIT PASSED: all 21 YAMLs and 42 sphere + 42 legacy ellipsoid completed settings match '
          'outside intended background/layout and run/storage metadata differences.', flush=True)
    if args.audit_only:
        for label in LABELS:
            print(f'  {label}: plane = regular except class/name; Visible-UV = regular plus approved allocation')
        print('Audit only: no files or training created.')
        return
    jobs = make_jobs(args.batch, references)
    estimate = sum(j['buffered_seconds'] for j in jobs)
    if estimate >= 10 * 3600:
        raise ValueError('Conservative full-batch estimate is not below ten hours')
    manifest = {'schema': 1, 'batch': args.batch, 'jobs': jobs, 'context': context,
                'buffer_factor': BUFFER, 'buffered_hours': estimate / 3600}
    if root.exists() and any(root.iterdir()):
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing output has a different/missing manifest; never overwrite it')
    cohort_path = LOCK_ROOT / 'cohort.json'
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != context:
        raise ValueError('Inputs/environment differ from the other plane batch; restore them')
    print(f'{args.batch}: {len(jobs)} fresh jobs; conservative historical {estimate/BUFFER/3600:.2f} h; '
          f'with 35% allowance {estimate/3600:.2f} h; output {root}', flush=True)
    print('Ten hours is a scheduling estimate, not a hard limit on an active training. '
          'If slower, rerun the same wrapper to finish remaining jobs.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(command(job, root / job['id'] / 'attempt_001')))
        print('Dry run: no files or training created.')
        return
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOCK_ROOT / 'gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another plane batch is active; run the two parts sequentially') from None
        if cohort_path.exists() and json.loads(cohort_path.read_text()) != context:
            raise ValueError('Shared plane cohort changed while acquiring lock')
        if not cohort_path.exists():
            cohort_path.write_text(json.dumps(context, indent=2) + '\n')
        root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Plane batch manifest changed while acquiring lock')
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for index, job in enumerate(jobs, 1):
            attempts = sorted((root / job['id']).glob('attempt_*'))
            good = [p for p in attempts if completed_plane(p, job) is not None]
            if len(good) > 1:
                raise ValueError(f'Multiple completed attempts: {job["id"]}')
            if good:
                print(f'[{index}/{len(jobs)}] SKIP completed {job["id"]}', flush=True)
                continue
            if time.monotonic() - started + job['buffered_seconds'] > args.max_hours * 3600:
                print('INVOCATION BUDGET REACHED: no new training started; rerun the same script to finish.', flush=True)
                summarize(root, jobs)
                return
            assert_signature(context)
            number = max([int(p.name.split('_')[-1]) for p in attempts] or [0]) + 1
            output = root / job['id'] / f'attempt_{number:03d}'
            print(f'[{index}/{len(jobs)}] {shlex.join(command(job, output))}', flush=True)
            subprocess.run(command(job, output), cwd=REPO, check=True)
            if completed_plane(output, job) is None:
                raise RuntimeError(f'Training returned without complete intended outputs: {output}')
        summarize(root, jobs)


if __name__ == '__main__':
    main()
