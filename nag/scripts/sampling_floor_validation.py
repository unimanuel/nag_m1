#!/usr/bin/env python3
"""Prepare inspectable sampling-floor ellipsoids, or train their ten controlled jobs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_io import SEQUENCES, inventory, write_json
from diagnostics.parallax_observations import check_prepared
from reprojection_ellipsoid import ReprojectionFitConfig
from sampling_floor_ellipsoid import fit_sampling_floor
from fit_reprojection_ellipsoid import (_source_geometry, _validate_contract, _copy_inspection_assets,
                                      _write_mesh, _write_diagnostics, _document)

LABELS = ['W191_0_40', 'W254_0_37', 'W254_37_81', 'W254_86_137', 'W254_139_195']
VERSION = 'sampling_floor_v1'
REVISION_LABELS = ['W191_40_128', 'W191_128_198', 'W254_86_137']


def paths(root, label, version=VERSION):
    segment, first, stop = SEQUENCES[label]
    parent = root / f'segment-{segment}'
    prefix = f'frames_{first:03d}_{stop-1:03d}_reprojection_depth'
    return parent / (prefix + '_v1') / 'ellipsoid.json', parent / (prefix + '_' + version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train'])
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES))
    parser.add_argument('--preserve-zero-floor', action='store_true', help='Version 2: preserve feasible zero-penalty originals exactly')
    parser.add_argument('--retrain-unchanged', action='store_true', help='Train even if version 2 preserved the original geometry exactly')
    parser.add_argument('--prepared-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--ellipsoid-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['regular', 'visuv'], default=['regular', 'visuv'])
    parser.add_argument('--max-iterations', type=int, default=200)
    args = parser.parse_args()
    version = 'sampling_floor_v2' if args.preserve_zero_floor else VERSION
    args.sequences = args.sequences or (REVISION_LABELS if args.preserve_zero_floor else LABELS)
    if args.max_iterations <= 0:
        parser.error('--max-iterations must be positive')
    if args.stage == 'train':
        commands = []
        reuse_inventory = None
        for label in args.sequences:
            source, out = paths(args.ellipsoid_root, label, version)
            document = json.loads((out / 'ellipsoid.json').read_text())
            if document['fit']['regularizer_version'] != version:
                raise ValueError(f'{label}: wrong regularizer version')
            segment, first, stop = SEQUENCES[label]
            if document['sequence'] != f'segment-{segment}' or document['frame_slice'] != json.loads(source.read_text())['frame_slice']:
                raise ValueError(f'{label}: geometry sequence mismatch')
            report = json.loads((out / 'sampling_floor_diagnostics.json').read_text())
            if report['full_final']['weighted_data_loss'] > report['full_original']['weighted_data_loss'] * 1.02 + 1e-6:
                raise ValueError(f'{label}: geometric budget exceeded')
            if report.get('zero_floor_preserved') and not args.retrain_unchanged:
                original = json.loads(source.read_text())
                if any(not np.array_equal(_source_geometry(document)[k], _source_geometry(original)[k])
                       for k in ('center', 'radii', 'rotation')):
                    raise ValueError(f'{label}: claimed preserved geometry differs from original')
                if reuse_inventory is None:
                    reuse_inventory = inventory(REPO, args.sequences, REPO / 'data/datasets/waymo',
                                                ['2026-09-12_reprojection_depth_v1'])
                for mode in args.modes:
                    matches = [r for r in reuse_inventory['runs'] if r['sequence'] == label and
                               r['family'] == 'reprojection_v1' and r['mode'] == mode and r['cohort'] == 'benchmark']
                    if len(matches) != 1:
                        raise ValueError(f'{label} {mode}: no unambiguous completed original; use --retrain-unchanged')
                    if any(not np.allclose(_source_geometry(original)[k], matches[0]['geometry'][k], rtol=1e-6, atol=1e-4)
                           for k in ('center', 'radii', 'rotation')):
                        raise ValueError(f'{label} {mode}: completed original geometry mismatch')
                    print(f'  reuse {mode}: {matches[0]["path"]}', flush=True)
                print(f'{label}: exact original geometry preserved; reuse original reprojection-v1 results '
                      f'(not a new training run). Use --retrain-unchanged for a repeat.', flush=True)
                continue
            for mode in args.modes:
                suffix = '_sph_visuv' if mode == 'visuv' else ''
                config = REPO / 'config/waymo/large-ego' / f'segment-{segment}_{first}_{stop-1}_large-undistorted{suffix}.yaml'
                if not config.is_file():
                    raise FileNotFoundError(config)
                name = f'{label}_large-undistorted_ellipsoid_reprojection_depth_{version}' + ('_visuv' if mode == 'visuv' else '')
                commands.append([sys.executable, str(REPO / 'nag/scripts/run_nag.py'),
                    '--config-path', str(config), '--background-plane-type',
                    'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D',
                    '--background-ellipsoid-path', str(out / 'ellipsoid.json'),
                    '--background-ellipsoid-scale-margin', '0',
                    '--background-ellipsoid-texture-orientation', 'legacy', '--name', name])
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)
        print(f'{len(commands)} controlled jobs; dry_run={args.dry_run}')
        return

    inv = inventory(REPO, args.sequences, REPO / 'data/datasets/waymo', ['2026-09-12_reprojection_depth_v1'])
    jobs = []
    for label in args.sequences:
        source, out = paths(args.ellipsoid_root, label, version)
        document = json.loads(source.read_text())
        pointer = args.prepared_root / label / 'prepared.json'
        prepared = check_prepared(pointer, REPO)
        _validate_contract(label, prepared, document)
        if prepared['sequence'] != label or not (Path(prepared['directory']) / 'pairs.npz').is_file():
            raise ValueError(f'{label}: mismatched or missing prepared observation cache')
        for asset in ('front_points.ply', 'background_points.ply', 'camera_centers.ply'):
            if not (source.parent / asset).is_file():
                raise FileNotFoundError(source.parent / asset)
        if (document['fit'].get('regularizer_version', 'original') != 'original' or
                document['reprojection_fit']['config'].get('radius_excess_weight', 0) != 0):
            raise ValueError(f'{label}: requires original reprojection-v1 initializer')
        runs = [r for r in inv['runs'] if r['sequence'] == label and r['mode'] == 'regular' and r['cohort'] == 'benchmark']
        sphere = next(r for r in runs if r['family'] == 'sphere')
        original_run = next(r for r in runs if r['family'] == 'reprojection_v1')
        if prepared['size'] != original_run['size'] or prepared['frames'] != original_run['frames']:
            raise ValueError(f'{label}: observation size or frame interval differs from training')
        for field in ('rotation', 'u_offset', 'y_flip', 'resolution_hw'):
            if not np.array_equal(original_run['mapping'][field], sphere['mapping'][field]):
                raise ValueError(f'{label}: canonical sphere and ellipsoid mapping differ in {field}')
        for name in ('center', 'radii', 'rotation'):
            if not np.allclose(_source_geometry(document)[name], original_run['geometry'][name], atol=1e-4):
                raise ValueError(f'{label}: source fit does not match trained original geometry')
        cfg = replace(ReprojectionFitConfig(**document['reprojection_fit']['config']), max_iterations=args.max_iterations)
        if out.exists():
            raise FileExistsError(f'{out}: output exists; preserve it or choose another --ellipsoid-root')
        jobs.append((label, source, out, document, pointer, prepared, sphere, original_run, cfg))
        print(f'{label}: original fit -> {out}/ellipsoid.ply', flush=True)
    if args.dry_run:
        print('Preparation preflight passed. No fitting or files written.')
        return
    for label, source, out, document, pointer, prepared, sphere, original_run, cfg in jobs:
        with np.load(Path(prepared['directory']) / 'pairs.npz', allow_pickle=False) as archive:
            pairs = dict(archive)
        print(f'[fit] {label}', flush=True)
        fit = fit_sampling_floor(pairs, _source_geometry(document), sphere['geometry'],
                                 original_run['mapping'], original_run['size'], cfg,
                                 preserve_zero_floor=args.preserve_zero_floor)
        out.mkdir(parents=True)
        _copy_inspection_assets(source, out)
        _write_mesh(out / 'ellipsoid.ply', fit['geometry'])
        _write_diagnostics(out, fit, label, source, pointer)
        result = _document(document, fit, source, pointer, label)
        result['fit']['method'] = 'Original reprojection-depth fit refined with canonical regular texture sampling floor; constrained SLSQP continuation'
        if fit['sampling_floor'].get('zero_floor_preserved'):
            result['fit']['method'] = 'Exact original reprojection-depth geometry retained after full cached-source zero-floor and safety checks; optimization skipped'
        result['fit']['sampling_floor'] = fit['sampling_floor']
        result['files']['sampling_floor_diagnostics_json'] = 'sampling_floor_diagnostics.json'
        result['reprojection_fit']['source_geometry_role'] = 'original reprojection-v1 fit, not raw-v4'
        report = fit['sampling_floor']
        report['provenance'] = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'prepared_key': prepared['key'], 'sphere_run': sphere['path'], 'original_run': original_run['path'],
            'code_sha256': hashlib.sha256((REPO / 'nag/sampling_floor_ellipsoid.py').read_bytes()).hexdigest()}
        write_json(out / 'sampling_floor_diagnostics.json', report)
        write_json(out / 'ellipsoid.json', result)
        radii = np.asarray(fit['geometry']['radii'])
        major = int(np.argmax(radii))
        print(f'{label}: radii={radii.round(4).tolist()} m; major-axis world XYZ direction='
              f'{np.asarray(fit["geometry"]["rotation"])[:, major].round(4).tolist()}', flush=True)
        print(f'  full sampling: {report["full_sampling"]}; retained original={report["retained_original"]}', flush=True)
        print(f'  inspect {out / "ellipsoid.ply"}', flush=True)
    print('Preparation complete. Inspect the PLYs before starting the separate training script.')


if __name__ == '__main__':
    main()
