#!/usr/bin/env python3
"""Prepare axis-neutral upright optimizer experiments from cached observations.

This is an experimental pre-training fitter. It does not train or overwrite
the established upright/legacy ellipsoid folders.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))

from diagnostics.parallax_io import SEQUENCES, write_json
from diagnostics.parallax_observations import check_prepared
from fit_reprojection_ellipsoid import (_copy_inspection_assets, _document, _source_geometry,
    _validate_contract, _write_diagnostics, _write_mesh)
from reprojection_ellipsoid import ReprojectionFitConfig
from sampling_floor_ellipsoid import fit_sampling_floor, world_y_regular_mapping


def version(weight):
    if not np.isfinite(weight) or weight <= 0 or weight >= 10:
        raise ValueError('compactness weight must be finite and in (0, 10)')
    units = round(weight * 1000)
    if not np.isclose(weight, units / 1000, atol=1e-12, rtol=0):
        raise ValueError('compactness weight must have at most three decimal places')
    return f'upright_compact_w{units:04d}_v1'


def strong_floor_version(threshold):
    if not np.isfinite(threshold) or not .5 < threshold <= 1:
        raise ValueError('strong-floor threshold must be finite and in (0.5, 1]')
    units = round(threshold * 1000)
    if not np.isclose(threshold, units / 1000, atol=1e-12, rtol=0):
        raise ValueError('strong-floor threshold must have at most three decimal places')
    return f'upright_strong_floor_t{units:04d}_v1'


def paths(root, label, treatment):
    segment, first, stop = SEQUENCES[label]
    parent = root / f'segment-{segment}'
    prefix = f'frames_{first:03d}_{stop-1:03d}_reprojection_depth'
    source = parent / (prefix + '_v1') / 'ellipsoid.json'
    previous = parent / (prefix + '_sampling_floor_world_y_v2') / 'sampling_floor_diagnostics.json'
    output = parent / f'autowork_frames_{first:03d}_{stop-1:03d}_{treatment}'
    return source, previous, output


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    treatment = (version(args.compactness_weight) if args.recipe == 'compact'
                 else strong_floor_version(args.floor_threshold))
    jobs = []
    for label in args.sequences:
        source, previous, output = paths(args.ellipsoid_root, label, treatment)
        pointer = args.prepared_root / label / 'prepared.json'
        prepared = check_prepared(pointer, REPO)
        document = json.loads(source.read_text())
        old = json.loads(previous.read_text())
        _validate_contract(label, prepared, document)
        if old['provenance']['source_sha256'] != sha(source) or \
                old['provenance']['prepared_key'] != prepared['key']:
            raise ValueError(f'{label}: historical upright fit provenance does not match current inputs')
        if document['fit'].get('regularizer_version', 'original') != 'original':
            raise ValueError(f'{label}: expected original reprojection-depth geometry')
        if old['canonical_mapping'].get('visible_uv', {}).get('enabled'):
            raise ValueError(f'{label}: canonical mapping must be regular, not Visible-UV')
        if output.exists():
            raise FileExistsError(f'{output}: never overwrite an existing experiment')
        config = replace(ReprojectionFitConfig(**document['reprojection_fit']['config']),
                         max_iterations=args.max_iterations)
        jobs.append((label, source, previous, output, pointer, prepared, document, old, config))
        print(f'{label}: {output}', flush=True)
    if args.dry_run:
        print(f'Preflight passed for {len(jobs)} sequences; no files written.')
        return
    for label, source, previous, output, pointer, prepared, document, old, config in jobs:
        with np.load(Path(prepared['directory']) / 'pairs.npz', allow_pickle=False) as cache:
            pairs = dict(cache)
        template = old['canonical_mapping']
        fit = fit_sampling_floor(
            pairs, _source_geometry(document), old['sphere'], template, prepared['size'],
            config, preserve_zero_floor=True,
            mapping_for_geometry=lambda geometry: world_y_regular_mapping(geometry, template),
            regularizer_version=treatment,
            compactness_weight=args.compactness_weight if args.recipe == 'compact' else 0.,
            axis_multistart=True,
            floor_threshold=.5 if args.recipe == 'compact' else args.floor_threshold,
            preserve_floor_threshold=.5,
        )
        report = fit['sampling_floor']
        if report['full_final']['weighted_data_loss'] > \
                report['full_original']['weighted_data_loss'] * 1.02 + 1e-6:
            raise RuntimeError(f'{label}: full-observation data budget exceeded')
        output.mkdir()
        _copy_inspection_assets(source, output)
        _write_mesh(output / 'ellipsoid.ply', fit['geometry'])
        _write_diagnostics(output, fit, label, source, pointer)
        result = _document(document, fit, source, pointer, label)
        result['fit']['method'] = (
            'World-Y regular-UV sampling floor with generic axis-contraction starts, '
            + ('an axis-neutral log-volume term' if args.recipe == 'compact'
               else 'a stronger sphere-relative floor on initially deficient fits')
            + ', original 2% data budget, and existing safety limits'
        )
        if report.get('zero_floor_preserved'):
            result['fit']['method'] = (
                'Original reprojection-depth geometry retained because it passed the '
                f'{report["preserve_floor_threshold"]:g}x-sphere reference-density floor; '
                'no new geometry optimization was run'
            )
        result['fit']['sampling_floor'] = report
        result['files']['sampling_floor_diagnostics_json'] = 'sampling_floor_diagnostics.json'
        result['reprojection_fit']['texture_orientation_policy'] = (
            'fixed-seam world_y mapping recomputed for every candidate; no Visible-UV fitting'
        )
        report['provenance'] = {
            'source_sha256': sha(source), 'previous_upright_report_sha256': sha(previous),
            'prepared_key': prepared['key'], 'fitter_sha256': sha(REPO / 'nag/sampling_floor_ellipsoid.py'),
            'driver_sha256': sha(Path(__file__)), 'texture_orientation': 'world_y',
            'training_performed': False, 'recipe': args.recipe,
        }
        write_json(output / 'sampling_floor_diagnostics.json', report)
        write_json(output / 'ellipsoid.json', result)
        radii = np.asarray(fit['geometry']['radii'])
        direction = np.asarray(fit['geometry']['rotation'])[:, int(np.argmax(radii))]
        print(f'{label}: radii={radii.round(3).tolist()}, major world direction='
              f'{direction.round(3).tolist()}, floor={report["full_sampling"]["final"]["loss"]:.6g}, '
              f'data={report["full_final"]["weighted_data_loss"]:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES),
                        default=list(SEQUENCES))
    parser.add_argument('--compactness-weight', type=float, default=.1)
    parser.add_argument('--recipe', choices=('compact', 'strong_floor'), default='compact')
    parser.add_argument('--floor-threshold', type=float, default=.8)
    parser.add_argument('--max-iterations', type=int, default=200)
    parser.add_argument('--prepared-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--ellipsoid-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_iterations <= 0 or len(args.sequences) != len(set(args.sequences)):
        parser.error('max-iterations must be positive and sequences unique')
    if args.recipe == 'compact':
        version(args.compactness_weight)
    else:
        strong_floor_version(args.floor_threshold)
    prepare(args)


if __name__ == '__main__':
    main()
