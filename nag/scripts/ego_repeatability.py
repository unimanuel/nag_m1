"""Fresh same-seed repeats of the validated seven-interval sampling-floor recipe."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_io import SEQUENCES

V2 = {'W191_40_128', 'W191_128_198', 'W254_86_137'}
METRICS = ('MPSNR', 'SSIM', 'LPIPS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(folder, frame_count):
    """Only a successful exit plus all final frames and final metrics counts."""
    if not (folder / 'exit_000.txt').is_file():
        return None
    frames = sorted(int(p.name.split('_')[0]) for p in (folder / 'final/complete').glob('*_t.png'))
    if frames != list(range(frame_count)):
        return None
    values = {}
    try:
        for metric in METRICS:
            with (folder / 'tracker' / f'train_epoch_Mean{metric}.csv').open(newline='') as f:
                row = list(csv.DictReader(f, delimiter=';'))[-1]
            value = float(row['value'])
            if float(row['step']) != 80 or not math.isfinite(value):
                return None
            values[metric] = value
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return values


def plan(labels, modes, repeats, extra_seeds, root):
    jobs = []
    rounds = [('same_seed', 42, n) for n in range(1, repeats + 1)]
    rounds += [('different_seed', seed, 1) for seed in extra_seeds]
    # Round-robin across intervals makes long-running coverage less front-loaded.
    for group, seed, repeat in rounds:
        for label in labels:
            segment, first, stop = SEQUENCES[label]
            version = 2 if label in V2 else 1
            geometry = root / f'segment-{segment}' / f'frames_{first:03d}_{stop-1:03d}_reprojection_depth_sampling_floor_v{version}' / 'ellipsoid.json'
            doc = json.loads(geometry.read_text())
            if (doc['sequence'] != f'segment-{segment}' or
                    tuple(doc['frame_slice'][k] for k in ('start', 'stop', 'step')) != (first, stop, 1) or
                    doc['fit']['regularizer_version'] != f'sampling_floor_v{version}'):
                raise ValueError(f'{label}: geometry does not match the intended cohort')
            for mode in modes:
                suffix = '_sph_visuv' if mode == 'visuv' else ''
                config = REPO / 'config/waymo/large-ego' / f'segment-{segment}_{first}_{stop-1}_large-undistorted{suffix}.yaml'
                cfg = yaml.safe_load(config.read_text())['NAGConfig']
                if cfg.get('deterministic', False) is not False or cfg.get('checkpoint_path') or cfg.get('max_epochs') != 80:
                    raise ValueError(f'{config}: requires fresh epoch-80 training with deterministic=False')
                job_id = f'{label}_{mode}_{group}_seed{seed}_repeat{repeat:02d}'
                jobs.append(dict(id=job_id, label=label, mode=mode, group=group,
                    seed=seed, repeat=repeat, frame_count=stop-first,
                    config=str(config), config_sha256=sha(config),
                    geometry=str(geometry), geometry_sha256=sha(geometry)))
    return jobs


def command(job, output):
    return [sys.executable, str(REPO / 'nag/scripts/run_nag.py'),
        '--config-path', job['config'], '--background-plane-type',
        'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D',
        '--background-ellipsoid-path', job['geometry'],
        '--background-ellipsoid-scale-margin', '0',
        '--background-ellipsoid-texture-orientation', 'legacy',
        '--seed', str(job['seed']), '--name', job['id'], '--output-path', str(output)]


def summarize(root, jobs):
    groups = {}
    done = 0
    for job in jobs:
        matches = [(p, completed(p, job['frame_count'])) for p in sorted((root / job['id']).glob('attempt_*'))]
        good = [(p, v) for p, v in matches if v is not None]
        if len(good) > 1:
            raise ValueError(f'Multiple completed attempts for {job["id"]}; inspect before analysis')
        if good:
            done += 1
            groups.setdefault((job['label'], job['mode'], job['group']), []).append(good[0][1])
    print(f'Completed {done}/{len(jobs)} jobs')
    for key, values in sorted(groups.items()):
        print(' / '.join(key), f'n={len(values)}')
        for metric in METRICS:
            v = [r[metric] for r in values]
            sd = f'{statistics.stdev(v):.6f}' if len(v) > 1 else 'n/a'
            print(f'  {metric}: mean={statistics.mean(v):.6f}, sample_sd={sd}, min={min(v):.6f}, max={max(v):.6f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    p.add_argument('--modes', nargs='+', choices=['regular', 'visuv'], default=['regular', 'visuv'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--extra-seeds', nargs='*', type=int, default=[])
    p.add_argument('--ellipsoid-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    p.add_argument('--output-root', type=Path, default=REPO / 'runs/2026-09-25_ego_repeatability')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    root = args.output_root.resolve()
    if args.summary:
        summarize(root, json.loads((root / 'manifest.json').read_text())['jobs'])
        return
    if args.repeats < 2 or 42 in args.extra_seeds or any(s < 0 or s >= 2**32 for s in args.extra_seeds):
        p.error('Use at least two repeats; extra seeds must be unsigned 32-bit values other than 42')
    if any(len(v) != len(set(v)) for v in [args.extra_seeds, args.sequences, args.modes]):
        p.error('Duplicate sequences, modes or seeds are not allowed')
    jobs = plan(args.sequences, args.modes, args.repeats, args.extra_seeds, args.ellipsoid_root.resolve())
    code = {str(f.relative_to(REPO)): sha(f) for base in [REPO / 'nag', REPO / 'tools/tools'] for f in sorted(base.rglob('*.py'))}
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'lightning']}
    manifest = dict(schema=1, jobs=jobs, code=code, versions=versions, python=sys.version,
        executable=sys.executable, environment={k: os.environ.get(k) for k in
            ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'DATA_PATH']})
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Batch inputs, code, environment or job selection changed. Restore them or choose a new --output-root.')
    print(f'{len(jobs)} fresh jobs planned; output: {root}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(command(job, root / job['id'] / 'attempt_001')))
        print('Dry run only: no files written; use --summary for completion status.')
        return
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Batch manifest changed before acquiring lock')
        if not path.exists():
            path.write_text(json.dumps(manifest, indent=2) + '\n')
        for index, job in enumerate(jobs, 1):
            attempts = sorted((root / job['id']).glob('attempt_*'))
            if any(completed(d, job['frame_count']) is not None for d in attempts):
                print(f'[{index}/{len(jobs)}] SKIP completed {job["id"]}', flush=True)
                continue
            # Keep failed/partial runs. A retry is a fresh training, never an overwrite.
            number = max([int(d.name.split('_')[-1]) for d in attempts] or [0]) + 1
            output = root / job['id'] / f'attempt_{number:03d}'
            if sha(job['config']) != job['config_sha256'] or sha(job['geometry']) != job['geometry_sha256']:
                raise ValueError('Input changed during batch; stopping')
            if any(sha(REPO / name) != digest for name, digest in code.items()):
                raise ValueError('Training code changed during batch; stopping')
            print(f'[{index}/{len(jobs)}] {shlex.join(command(job, output))}', flush=True)
            subprocess.run(command(job, output), cwd=REPO, check=True)
            if completed(output, job['frame_count']) is None:
                raise RuntimeError(f'Training returned without complete epoch-80 outputs: {output}')
        summarize(root, jobs)


if __name__ == '__main__':
    main()
