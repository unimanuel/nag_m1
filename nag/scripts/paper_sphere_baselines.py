"""Fresh matched sphere baselines: three seed-42 repeats, both UV layouts.

Five sequential batches stay below six hours at buffered historical rates.
No ellipsoid fitting, upright changes, plane training, or checkpoint resume.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag/scripts'))
from ego_repeatability import completed, summarize
from diagnostics.parallax_io import diagnostic

BATCHES = {
    'part1of5': ['W191_0_40', 'W191_128_198'],
    'part2of5': ['W254_0_37', 'W254_86_137'],
    'part3of5': ['W191_40_128'],
    'part4of5': ['W254_37_81'],
    'part5of5': ['W254_139_195'],
}
BUFFER = 1.35
REPEATS = 3
MODES = ('regular', 'visuv')
SPHERE_TYPE = 'nag.model.view_dependent_background_image_sphere_scene_node_3d.ViewDependentBackgroundImageSphereSceneNode3D'
REFERENCES = REPO / 'reports/2026-09-28_upright_comparison.sources.json'
LOCK_ROOT = REPO / 'runs/autowork_paper_sphere_baselines_lock'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration_seconds(path):
    fields = dict(line.split(': ', 1) for line in Path(path).read_text().splitlines()
                  if ': ' in line)
    elapsed = (datetime.fromisoformat(fields['End-Date']) -
               datetime.fromisoformat(fields['Start-Date'])).total_seconds()
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f'Invalid historical duration: {path}')
    return elapsed


def validate_config(settings, label, mode, reference_hash, config):
    if sha(config) != reference_hash:
        raise ValueError(f'{label} {mode}: YAML differs from the completed ellipsoid repeats')
    _, first, last = label.split('_')
    frames = settings['frame_indices_filter']
    if (frames['start'], frames['stop'], frames.get('step') or 1) != (int(first), int(last) + 1, 1):
        raise ValueError(f'{label}: incorrect frame interval')
    if (settings.get('max_epochs') != 80 or settings.get('deterministic', False)
            or settings.get('checkpoint_path') or settings.get('checkpoint_tracker_path')
            or settings.get('save_intermediate_checkpoints', False)):
        raise ValueError(f'{label}: requires fresh nondeterministic epoch-80, final-only training')
    if settings['background_plane_type'] != SPHERE_TYPE:
        raise ValueError(f'{label}: not a sphere YAML')
    uv = settings['background_plane_kwargs'].get('sphere_visible_uv_texture', {}).get('enabled', False)
    if bool(uv) != (mode == 'visuv'):
        raise ValueError(f'{label}: incorrect UV allocation')
    if settings.get('background_sphere_radius_override') is not None:
        raise ValueError(f'{label}: unexpected radius override')


def make_jobs(batch, references):
    jobs = []
    for repeat in range(1, REPEATS + 1):
        for label in BATCHES[batch]:
            # Reverse layout order on repeat two, matching the paired-study idea.
            for mode in (MODES if repeat % 2 else MODES[::-1]):
                ref = references[label, mode]
                jobs.append({'id': f'autowork_{label}_sphere_{mode}_same_seed_seed42_repeat{repeat:02d}',
                             'label': label, 'mode': mode, 'group': 'same_seed',
                             'seed': 42, 'repeat': repeat, 'frame_count': ref['frame_count'],
                             'config': ref['config'], 'config_sha256': ref['config_sha256'],
                             'reference_run': ref['path'], 'reference_geometry': ref['geometry'],
                             'reference_geometry_sha256': ref['geometry_sha256'],
                             'historical_seconds': ref['seconds'],
                             'buffered_seconds': BUFFER * ref['seconds']})
    return jobs


def code_snapshot():
    paths = [p for base in [REPO / 'nag', REPO / 'tools/tools'] for p in base.rglob('*.py')]
    paths += [REPO / 'run_paper_sphere_baselines.sh']
    paths += [REPO / f'run_paper_sphere_baselines_part{i}.sh' for i in range(1, 6)]
    from preparation.compatibility import reviewed_validation_snapshot
    return reviewed_validation_snapshot({str(p.relative_to(REPO)): sha(p) for p in sorted(paths)})


def check_training_compatibility(old, current):
    """Reject changed model/training code except the audited checkpoint policy."""
    changes = {p for p, h in old.items() if current.get(p) != h}
    storage = {'nag/callbacks/nag_callback.py', 'nag/config/nag_config.py'}
    ignored = {'nag/sampling_floor_ellipsoid.py'}  # fitter is not used in sphere training
    unexpected = changes - storage - ignored
    if unexpected:
        raise ValueError(f'Code differs from ellipsoid repeats: {sorted(unexpected)}')
    for path in changes & storage:
        # These exact revisions only disable intermediate checkpoint writes.
        expected = subprocess.check_output(['git', 'show', f'3701bda:{path}'], cwd=REPO)
        before = subprocess.check_output(['git', 'show', f'3701bda^:{path}'], cwd=REPO)
        raw = (REPO / path).read_bytes()
        restored = before
        if path == 'nag/callbacks/nag_callback.py' and hashlib.sha256(before).hexdigest() != old[path]:
            # The WSL worktree was mixed CRLF/LF. Applying the checkpoint patch
            # changed its two context lines to LF as well as adding three lines.
            # Reverse exactly that byte-level patch and verify the old manifest
            # hash; do not waive provenance merely because normalized text fits.
            insertion = (b'        if prefix != "final-" and not getattr(model.config, "save_intermediate_checkpoints", False):\n'
                         b'            logging.debug("Skipping intermediate checkpoint; only the final checkpoint is enabled.")\n'
                         b'            return\n')
            if raw.count(insertion) != 1:
                raise ValueError(f'Unexpected checkpoint policy patch: {path}')
            restored = raw.replace(insertion, b'', 1)
            for line in [b'    def save_model(self, trainer: pl.Trainer, model: NAGModel, prefix: str = ""):\n',
                         b'        checkpoint_dir = os.path.join(model.config.output_path, "checkpoints")\n']:
                if restored.count(line) != 1:
                    raise ValueError(f'Unexpected checkpoint patch context: {path}')
                restored = restored.replace(line, line[:-1] + b'\r\n', 1)
        if (hashlib.sha256(raw).hexdigest() != current[path]
                or raw.replace(b'\r\n', b'\n') != expected
                or restored.replace(b'\r\n', b'\n') != before
                or hashlib.sha256(restored).hexdigest() != old[path]):
            raise ValueError(f'Unreviewed change to training configuration/callback: {path}')
    return {'changed_since_ellipsoid_repeats': sorted(changes),
            'storage_only_commit': '3701bda', 'unused_fitter_changes_excluded': sorted(changes & ignored)}


def load_references(current):
    records = json.loads(REFERENCES.read_text())['runs']
    references, repeat_manifests, compatibility = {}, [], []
    ellipsoid_jobs = []
    for part in range(1, 5):
        path = REPO / f'runs/2026-09-25_ego_poles_repeatability_part{part}of4/manifest.json'
        manifest = json.loads(path.read_text())
        repeat_manifests.append({'path': str(path), 'sha256': sha(path)})
        compatibility.append(check_training_compatibility(manifest['code'], current))
        expected_versions = {name: importlib.metadata.version(name) for name in manifest['versions']}
        if manifest['versions'] != expected_versions or manifest['executable'] != sys.executable:
            raise ValueError('Use the same nag1 interpreter and package versions as the ellipsoid repeats')
        ellipsoid_jobs += [j for j in manifest['jobs'] if j['orientation'] == 'legacy' and j['repeat'] == 1]
    for record in records:
        if record['family'] != 'sphere':
            continue
        label, mode = record['sequence'], record['mode']
        key = label, mode
        if key in references:
            raise ValueError(f'Ambiguous sphere reference: {key}')
        path = Path(record['path'])
        _, first, last = label.split('_')
        count = int(last) - int(first) + 1
        if (completed(path, count) is None
                or not (path / 'exit_000.txt').read_text().startswith('Exit code: 0.')):
            raise ValueError(f'Incomplete historical sphere reference: {path}')
        matches = [j for j in ellipsoid_jobs if (j['label'], j['mode']) == key]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous paired ellipsoid YAML: {key}')
        config = Path(matches[0]['config'])
        settings = yaml.safe_load(config.read_text())['NAGConfig']
        validate_config(settings, label, mode, matches[0]['config_sha256'], config)
        geom_path = path / 'initial/background_diagnostics/sphere_geometry.txt'
        geom = diagnostic(geom_path)
        if geom['radius_source'] != 'visible_plane_centers' or \
                bool(geom['visible_uv_texture']['enabled']) != (mode == 'visuv'):
            raise ValueError(f'Unexpected historical sphere geometry: {path}')
        references[key] = {'path': str(path), 'frame_count': count,
                           'config': str(config), 'config_sha256': sha(config),
                           'geometry': str(geom_path), 'geometry_sha256': sha(geom_path),
                           'seconds': duration_seconds(path / 'exit_000.txt')}
    expected = {(label, mode) for labels in BATCHES.values() for label in labels for mode in MODES}
    if set(references) != expected:
        raise ValueError('Sphere reference coverage mismatch')
    return references, repeat_manifests, compatibility


def command(job, output):
    return [sys.executable, str(REPO / 'nag/scripts/run_nag.py'),
            '--config-path', job['config'], '--background-plane-type', SPHERE_TYPE,
            '--background-sphere-radius-from-visible-plane-centers',
            '--seed', '42', '--name', job['id'], '--output-path', str(output)]


def completed_sphere(path, job):
    result = completed(path, job['frame_count'])
    if result is None:
        return None
    if not (path / 'exit_000.txt').read_text().startswith('Exit code: 0.'):
        return None
    actual = diagnostic(path / 'initial/background_diagnostics/sphere_geometry.txt')
    expected = diagnostic(Path(job['reference_geometry']))
    if (actual['radius_source'] != 'visible_plane_centers'
            or abs(float(actual['radius']) - float(expected['radius'])) > .001
            or not np.allclose(actual['center'], expected['center'], atol=.001, rtol=0)
            or actual['color_mask_resolution'] != expected['color_mask_resolution']
            or bool(actual['visible_uv_texture']['enabled']) != (job['mode'] == 'visuv')):
        raise ValueError(f'Completed sphere geometry differs from the intended reference: {path}')
    settings = yaml.safe_load((path / 'training_config.yaml').read_text())['NAGConfig']
    if settings['seed'] != 42 or settings['max_epochs'] != 80 or settings['checkpoint_path'] \
            or settings.get('save_intermediate_checkpoints', False):
        raise ValueError(f'Completed run is not the intended final-only fresh training: {path}')
    return result


def budget_allows(elapsed, job, max_seconds):
    return elapsed + job['buffered_seconds'] <= max_seconds


def assert_signature(context):
    if code_snapshot() != context['code']:
        raise ValueError('Code changed during the batch; restore it before continuing')
    if sha(REFERENCES) != context['reference_inventory_sha256']:
        raise ValueError('Historical source inventory changed')
    for item in context['repeat_manifests']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Paired ellipsoid manifest changed')
    for ref in context['references']:
        for path_key, hash_key in [('config', 'config_sha256'), ('geometry', 'geometry_sha256')]:
            if sha(ref[path_key]) != ref[hash_key]:
                raise ValueError(f'Input changed during the batch: {ref[path_key]}')
    env_file = context.get('env_file')
    if env_file and sha(env_file['path']) != env_file['sha256']:
        raise ValueError('.env changed during the batch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, choices=list(BATCHES))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--max-hours', type=float, default=6.,
                        help='Stop scheduling if the buffered next job exceeds this invocation budget')
    args = parser.parse_args()
    if not np.isfinite(args.max_hours) or not 0 < args.max_hours <= 6:
        parser.error('max-hours must be in (0, 6]')
    root = (args.output_root or REPO / f'runs/autowork_paper_sphere_baselines_{args.batch}').resolve()
    manifest_path = root / 'manifest.json'
    if args.summary:
        manifest = json.loads(manifest_path.read_text())
        # Summarizer expects attempt_*; our training layout uses that convention.
        summarize(root, manifest['jobs'])
        return
    if not root.name.startswith('autowork_'):
        parser.error('new output folder must begin with autowork_')
    started = time.monotonic()
    code = code_snapshot()
    references, repeat_manifests, compatibility = load_references(code)
    jobs = make_jobs(args.batch, references)
    estimate = sum(j['buffered_seconds'] for j in jobs)
    if estimate >= 6 * 3600:
        raise ValueError('Buffered full-batch estimate is not below six hours')
    context = {'schema': 1, 'purpose': 'matched regular/Visible-UV sphere repeats, no fitting',
               'seed': 42, 'repeats': REPEATS, 'code': code,
               'versions': {k: importlib.metadata.version(k) for k in ['torch', 'numpy', 'lightning']},
               'python': sys.version, 'executable': sys.executable,
               'environment': {k: os.environ.get(k) for k in
                               ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'DATA_PATH']},
               'reference_inventory_sha256': sha(REFERENCES),
               'references': [references[k] for k in sorted(references)],
               'repeat_manifests': repeat_manifests, 'training_compatibility': compatibility}
    if (REPO / '.env').is_file():
        context['env_file'] = {'path': str(REPO / '.env'), 'sha256': sha(REPO / '.env')}
    manifest = {'schema': 1, 'batch': args.batch, 'jobs': jobs, 'context': context,
                'buffer_factor': BUFFER, 'buffered_hours': estimate / 3600}
    if root.exists() and any(root.iterdir()):
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing output has different or missing manifest; never overwrite it')
    cohort_path = LOCK_ROOT / 'cohort.json'
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != context:
        raise ValueError('Inputs/environment differ from another sphere batch; restore them before continuing')
    print(f'{args.batch}: {len(jobs)} fresh jobs; historical {estimate/BUFFER/3600:.2f} h; '
          f'with 35% allowance {estimate/3600:.2f} h; output {root}', flush=True)
    print('Six hours is a scheduling estimate, not a guaranteed limit on an active training. '
          'A slowed batch stops between jobs; rerun the same wrapper to finish.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(command(job, root / job['id'] / 'attempt_001')))
        print('Dry run: no files, fitting or training created.')
        return
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOCK_ROOT / 'gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another sphere batch is active; run the five parts sequentially') from None
        if cohort_path.exists() and json.loads(cohort_path.read_text()) != context:
            raise ValueError('Shared cohort signature changed while acquiring lock')
        if not cohort_path.exists():
            cohort_path.write_text(json.dumps(context, indent=2) + '\n')
        root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Batch signature changed while acquiring lock')
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for index, job in enumerate(jobs, 1):
            attempts = sorted((root / job['id']).glob('attempt_*'))
            good = [p for p in attempts if completed_sphere(p, job) is not None]
            if len(good) > 1:
                raise ValueError(f'Multiple completed attempts: {job["id"]}')
            if good:
                print(f'[{index}/{len(jobs)}] SKIP completed {job["id"]}', flush=True)
                continue
            if not budget_allows(time.monotonic() - started, job, args.max_hours * 3600):
                print('INVOCATION BUDGET REACHED: no new job started. Rerun the same script '
                      'to finish remaining jobs; active runs were not interrupted.', flush=True)
                summarize(root, jobs)
                return
            assert_signature(context)
            number = max([int(p.name.split('_')[-1]) for p in attempts] or [0]) + 1
            output = root / job['id'] / f'attempt_{number:03d}'
            print(f'[{index}/{len(jobs)}] {shlex.join(command(job, output))}', flush=True)
            subprocess.run(command(job, output), cwd=REPO, check=True)
            if completed_sphere(output, job) is None:
                raise RuntimeError(f'Training returned without complete intended outputs: {output}')
        summarize(root, jobs)


if __name__ == '__main__':
    main()
