#!/usr/bin/env python3
"""Fit and train a world-Y-aware sampling-floor cohort without overwriting legacy fits."""
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))

from diagnostics.parallax_io import SEQUENCES, inventory, write_json
from diagnostics.parallax_observations import check_prepared
from fit_reprojection_ellipsoid import (_copy_inspection_assets, _document, _source_geometry,
    _validate_contract, _write_diagnostics, _write_mesh)
from reprojection_ellipsoid import ReprojectionFitConfig
from sampling_floor_ellipsoid import fit_sampling_floor, world_y_regular_mapping


VERSION = 'sampling_floor_world_y_v2'
LEGACY_V2 = {'W191_40_128', 'W191_128_198', 'W254_86_137'}
METRICS = ('MPSNR', 'SSIM', 'LPIPS')
POLE_REPEAT_ROOTS = tuple(
    REPO / 'runs' / f'2026-09-25_ego_poles_repeatability_part{number}of4'
    for number in range(1, 5)
)


def paths(root, label):
    segment, first, stop = SEQUENCES[label]
    parent = root / f'segment-{segment}'
    prefix = f'frames_{first:03d}_{stop-1:03d}_reprojection_depth'
    return parent / (prefix + '_v1') / 'ellipsoid.json', parent / (prefix + '_' + VERSION)


def legacy_floor_path(root, label):
    segment, first, stop = SEQUENCES[label]
    version = 2 if label in LEGACY_V2 else 1
    return root / f'segment-{segment}' / (
        f'frames_{first:03d}_{stop-1:03d}_reprojection_depth_sampling_floor_v{version}'
    ) / 'ellipsoid.json'


def _same_geometry(a, b):
    # Runtime diagnostics retain float32 effective geometry, whereas the JSON
    # fit keeps float64 values.  This is a provenance comparison, not a demand
    # for byte-for-byte serialization equality.
    return all(np.allclose(_source_geometry(a)[field], _source_geometry(b)[field],
                           rtol=1e-6, atol=1e-4)
               for field in ('center', 'radii', 'rotation'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _complete(folder, frame_count):
    if not (folder / 'exit_000.txt').is_file():
        return None
    frames = sorted(int(path.name.split('_')[0])
                    for path in (folder / 'final/complete').glob('*_t.png'))
    if frames != list(range(frame_count)):
        return None
    values = {}
    try:
        for metric in METRICS:
            with (folder / 'tracker' / f'train_epoch_Mean{metric}.csv').open(newline='') as file:
                row = list(csv.DictReader(file, delimiter=';'))[-1]
            value = float(row['value'])
            if float(row['step']) != 80 or not math.isfinite(value):
                return None
            values[metric] = value
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return values


def _existing_upright_repeats(label, mode, geometry, config_path):
    """Return validated prior upright repeats if exactly the same treatment exists."""
    expected = []
    for root in POLE_REPEAT_ROOTS:
        manifest = root / 'manifest.json'
        if not manifest.is_file():
            continue
        for job in json.loads(manifest.read_text())['jobs']:
            if (job['label'], job['mode'], job['orientation'], job['seed']) != (label, mode, 'world_y', 42):
                continue
            if job['config_sha256'] != _sha(config_path):
                continue
            saved = Path(job['geometry'])
            if not saved.is_file() or not _same_geometry(geometry, json.loads(saved.read_text())):
                continue
            runs = [(path, _complete(path, job['frame_count']))
                    for path in (root / job['id']).glob('attempt_*')]
            expected.extend(str(path.resolve()) for path, value in runs if value is not None)
    return sorted(expected) if len(expected) == 3 else None


def _fit_mapping(template):
    def mapping_for_geometry(geometry):
        return world_y_regular_mapping(geometry, template)
    return mapping_for_geometry


def prepare(args):
    inv = inventory(REPO, args.sequences, REPO / 'data/datasets/waymo',
                    ['2026-09-12_reprojection_depth_v1'])
    jobs = []
    for label in args.sequences:
        source, out = paths(args.ellipsoid_root, label)
        document = json.loads(source.read_text())
        pointer = args.prepared_root / label / 'prepared.json'
        prepared = check_prepared(pointer, REPO)
        _validate_contract(label, prepared, document)
        if prepared['sequence'] != label or not (Path(prepared['directory']) / 'pairs.npz').is_file():
            raise ValueError(f'{label}: mismatched or missing prepared observation cache')
        if document['fit'].get('regularizer_version', 'original') != 'original':
            raise ValueError(f'{label}: must start from the original reprojection-depth v1 fit')
        if document['reprojection_fit']['config'].get('radius_excess_weight', 0) != 0:
            raise ValueError(f'{label}: source has a radius-scale regularizer')
        for asset in ('front_points.ply', 'background_points.ply', 'camera_centers.ply'):
            if not (source.parent / asset).is_file():
                raise FileNotFoundError(source.parent / asset)
        runs = [run for run in inv['runs'] if run['sequence'] == label and run['mode'] == 'regular'
                and run['cohort'] == 'benchmark']
        sphere = [run for run in runs if run['family'] == 'sphere']
        original_runs = [run for run in runs if run['family'] == 'reprojection_v1']
        if len(sphere) != 1 or len(original_runs) != 1:
            raise ValueError(f'{label}: requires one regular sphere and original reprojection reference')
        sphere, original_run = sphere[0], original_runs[0]
        if prepared['size'] != original_run['size'] or prepared['frames'] != original_run['frames']:
            raise ValueError(f'{label}: cached calibration size/frame mismatch')
        for field in ('u_offset', 'y_flip', 'resolution_hw'):
            if not np.array_equal(original_run['mapping'][field], sphere['mapping'][field]):
                raise ValueError(f'{label}: original ellipsoid and sphere differ in {field}')
        if original_run['mapping'].get('visible_uv', {}).get('enabled'):
            raise ValueError(f'{label}: original reference must use regular UV')
        if not _same_geometry(document, {'fit': {
                'center': original_run['geometry']['center'], 'radii': original_run['geometry']['radii'],
                'rotation_columns': original_run['geometry']['rotation']}}):
            raise ValueError(f'{label}: source JSON does not match the trained original reprojection geometry')
        if out.exists():
            raise FileExistsError(f'{out}: output already exists; use a separate root rather than overwrite')
        cfg = replace(ReprojectionFitConfig(**document['reprojection_fit']['config']),
                      max_iterations=args.max_iterations)
        jobs.append((label, source, out, document, pointer, prepared, sphere, original_run, cfg))
        print(f'{label}: original reprojection fit -> {out / "ellipsoid.ply"}', flush=True)
    if args.dry_run:
        print('Preparation preflight passed. No fitting or files written.')
        return
    for label, source, out, document, pointer, prepared, sphere, original_run, cfg in jobs:
        with np.load(Path(prepared['directory']) / 'pairs.npz', allow_pickle=False) as archive:
            pairs = dict(archive)
        print(f'[fit] {label}', flush=True)
        template = original_run['mapping']
        fit = fit_sampling_floor(
            pairs, _source_geometry(document), sphere['geometry'], template, original_run['size'], cfg,
            preserve_zero_floor=True, mapping_for_geometry=_fit_mapping(template),
            regularizer_version=VERSION,
        )
        out.mkdir(parents=True)
        _copy_inspection_assets(source, out)
        _write_mesh(out / 'ellipsoid.ply', fit['geometry'])
        _write_diagnostics(out, fit, label, source, pointer)
        result = _document(document, fit, source, pointer, label)
        result['fit']['method'] = (
            'Original reprojection-depth fit refined with a fixed-seam world-Y regular '
            'texture sampling floor; the normalized texture basis is recomputed for each candidate'
        )
        if fit['sampling_floor'].get('zero_floor_preserved'):
            result['fit']['method'] = (
                'Exact original reprojection-depth geometry retained after full cached-source '
                'world-Y zero-floor and safety checks; optimization skipped'
            )
        result['fit']['sampling_floor'] = fit['sampling_floor']
        result['files']['sampling_floor_diagnostics_json'] = 'sampling_floor_diagnostics.json'
        result['reprojection_fit']['source_geometry_role'] = 'original reprojection-v1 fit, not legacy sampling-floor geometry'
        result['reprojection_fit']['texture_orientation_policy'] = (
            'world_y is fitted in canonical regular UV with a fixed world-Z longitude reference; '
            'NAG training uses world_y explicitly'
        )
        report = fit['sampling_floor']
        report['provenance'] = {
            'source_sha256': _sha(source), 'prepared_key': prepared['key'],
            'sphere_run': sphere['path'], 'original_run': original_run['path'],
            'code_sha256': _sha(REPO / 'nag/sampling_floor_ellipsoid.py'),
            'runtime_texture_basis_sha256': _sha(REPO / 'nag/model/ellipsoid_texture.py'),
            'texture_orientation': 'world_y', 'automatic_seam': False,
        }
        write_json(out / 'sampling_floor_diagnostics.json', report)
        write_json(out / 'ellipsoid.json', result)
        radii = np.asarray(fit['geometry']['radii'])
        major = int(np.argmax(radii))
        print(f'{label}: radii={radii.round(4).tolist()} m; major-axis world XYZ direction='
              f'{np.asarray(fit["geometry"]["rotation"])[:, major].round(4).tolist()}', flush=True)
        print(f'  world-Y full sampling: {report["full_sampling"]}; retained original={report["retained_original"]}', flush=True)
        print(f'  inspect {out / "ellipsoid.ply"}', flush=True)
    print('Preparation complete. Inspect the new PLYs before starting the separate training script.')


def train(args):
    jobs, reused = [], []
    for label in args.sequences:
        source, folder = paths(args.ellipsoid_root, label)
        geometry = folder / 'ellipsoid.json'
        document = json.loads(geometry.read_text())
        report = json.loads((folder / 'sampling_floor_diagnostics.json').read_text())
        segment, first, stop = SEQUENCES[label]
        if document['fit']['regularizer_version'] != VERSION or document['sequence'] != f'segment-{segment}':
            raise ValueError(f'{label}: wrong upright-aware geometry document')
        if document['frame_slice'] != json.loads(source.read_text())['frame_slice']:
            raise ValueError(f'{label}: output/source frame slice mismatch')
        if report['full_final']['weighted_data_loss'] > report['full_original']['weighted_data_loss'] * 1.02 + 1e-6:
            raise ValueError(f'{label}: 2% reprojection/depth budget exceeded')
        if report['final_mapping']['mode'] != 'world_y' or report['final_mapping']['u_offset'] != 0:
            raise ValueError(f'{label}: not a fixed-seam world-Y fit')
        for mode in args.modes:
            suffix = '_sph_visuv' if mode == 'visuv' else ''
            config = REPO / 'config/waymo/large-ego' / f'segment-{segment}_{first}_{stop-1}_large-undistorted{suffix}.yaml'
            cfg = yaml.safe_load(config.read_text())['NAGConfig']
            if cfg.get('deterministic', False) is not False or cfg.get('checkpoint_path') or cfg.get('max_epochs') != 80:
                raise ValueError(f'{config}: requires fresh nondeterministic epoch-80 training')
            existing = _existing_upright_repeats(label, mode, document, config)
            if existing and not args.retrain_identical:
                reused.append(dict(label=label, mode=mode, paths=existing))
                continue
            job_id = f'{label}_{mode}_seed42_upright_sampling_floor_v2'
            jobs.append(dict(id=job_id, label=label, mode=mode, frame_count=stop-first,
                             config=str(config), config_sha256=_sha(config), geometry=str(geometry),
                             geometry_sha256=_sha(geometry)))
    root = args.output_root.resolve()
    signature = {str(path.relative_to(REPO)): _sha(path) for path in [
        Path(__file__), REPO / 'nag/sampling_floor_ellipsoid.py', REPO / 'nag/model/ellipsoid_texture.py',
        REPO / 'nag/model/background_image_ellipsoid_scene_node_3d.py',
        REPO / 'nag/scripts/run_nag.py',
    ]}
    manifest = dict(schema=1, treatment=VERSION, texture_orientation='world_y', seed=42,
                    jobs=jobs, reused=reused, code=signature,
                    versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'lightning')})
    if root.exists():
        manifest_path = root / 'manifest.json'
        if not manifest_path.is_file():
            if any(root.iterdir()):
                raise ValueError(f'{root} exists without a manifest; choose a new --output-root')
        elif json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Training manifest differs. Use the original inputs or a new --output-root.')
    print(f'{len(jobs)} new training jobs; {len(reused)} validated prior upright treatments reused.', flush=True)
    for item in reused:
        print(f'  reuse {item["label"]} {item["mode"]}: ' + ', '.join(item['paths']), flush=True)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        if not (root / 'manifest.json').exists():
            write_json(root / 'manifest.json', manifest)
    for job in jobs:
        attempts = sorted((root / job['id']).glob('attempt_*')) if root.exists() else []
        if any(_complete(path, job['frame_count']) is not None for path in attempts):
            print(f'[skip completed] {job["id"]}', flush=True)
            continue
        number = max([int(path.name.split('_')[-1]) for path in attempts] or [0]) + 1
        output = root / job['id'] / f'attempt_{number:03d}'
        command = [sys.executable, str(REPO / 'nag/scripts/run_nag.py'), '--config-path', job['config'],
                   '--background-plane-type',
                   'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D',
                   '--background-ellipsoid-path', job['geometry'], '--background-ellipsoid-scale-margin', '0',
                   '--background-ellipsoid-texture-orientation', 'world_y', '--seed', '42',
                   '--name', job['id'], '--output-path', str(output)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        if _sha(job['config']) != job['config_sha256'] or _sha(job['geometry']) != job['geometry_sha256']:
            raise ValueError('Training input changed during batch; stopping')
        subprocess.run(command, cwd=REPO, check=True)
        if _complete(output, job['frame_count']) is None:
            raise RuntimeError(f'Training returned without complete epoch-80 output: {output}')
    if args.dry_run:
        print('Dry run only: no manifest or training output written.')
        return
    print(f'Training complete or reused; output manifest: {root / "manifest.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train'))
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument('--modes', nargs='+', choices=('regular', 'visuv'), default=('regular', 'visuv'))
    parser.add_argument('--prepared-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--ellipsoid-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    parser.add_argument('--output-root', type=Path, default=REPO / 'runs/2026-09-28_upright_sampling_floor')
    parser.add_argument('--max-iterations', type=int, default=200)
    parser.add_argument('--retrain-identical', action='store_true',
                        help='Train all 14 jobs even where three completed world-Y repeats are exactly reusable.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_iterations <= 0:
        parser.error('--max-iterations must be positive')
    if len(args.sequences) != len(set(args.sequences)) or len(args.modes) != len(set(args.modes)):
        parser.error('Duplicate sequences or modes are not allowed')
    (prepare if args.stage == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
