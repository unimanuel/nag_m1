#!/usr/bin/env python3
"""Measure saved experimental upright geometries on matched cached observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))

from diagnostics.parallax_core import inside
from diagnostics.parallax_evaluate import batched_project, evaluate_pairs
from diagnostics.parallax_io import SEQUENCES, write_csv, write_json
from diagnostics.parallax_observations import check_prepared
from fit_reprojection_ellipsoid import _source_geometry
from reprojection_ellipsoid import ReprojectionFitConfig, evaluate_geometry
from sampling_floor_ellipsoid import floor_statistics, world_y_regular_mapping
from texture_sampling_diagnostics import balanced_weights, sampling_metrics
from autowork_upright_compactness import paths, strong_floor_version, version


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_candidates(root, label, weights, strong_floor_threshold=.8):
    segment, first, stop = SEQUENCES[label]
    parent = root / f'segment-{segment}'
    prefix = f'frames_{first:03d}_{stop-1:03d}_reprojection_depth'
    baseline = [('original_v1', parent / (prefix + '_v1') / 'ellipsoid.json')]
    legacy = [(name, parent / (prefix + name) / 'ellipsoid.json')
              for name in ('_sampling_floor_v2', '_sampling_floor_v1')]
    baseline.append(('legacy_floor', next(path for _, path in legacy if path.is_file())))
    baseline.append(('upright_floor_v2', parent / (prefix + '_sampling_floor_world_y_v2') / 'ellipsoid.json'))
    for weight in weights:
        _, _, output = paths(root, label, version(weight))
        if (output / 'ellipsoid.json').is_file():
            baseline.append((f'compact_{weight:g}', output / 'ellipsoid.json'))
    _, _, strong = paths(root, label, strong_floor_version(strong_floor_threshold))
    if (strong / 'ellipsoid.json').is_file():
        baseline.append((f'strong_floor_{strong_floor_threshold:g}', strong / 'ellipsoid.json'))
    return baseline


def finite_summary(values, mask):
    selected = np.asarray(values)[mask]
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return {'median': None, 'p90': None}
    return {'median': float(np.median(selected)), 'p90': float(np.quantile(selected, .9))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument('--weights', nargs='+', type=float, default=[.1, .15, .2])
    parser.add_argument('--strong-floor-threshold', type=float, default=.8)
    parser.add_argument('--prepared-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--ellipsoid-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    parser.add_argument('--output', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_upright_optimizer_diagnostics_v2'))
    args = parser.parse_args()
    if len(args.sequences) != len(set(args.sequences)) or args.output.exists():
        parser.error('sequences must be unique and output folder must not already exist')
    for weight in args.weights:
        version(weight)
    strong_floor_version(args.strong_floor_threshold)
    plans = {}
    for label in args.sequences:
        meta = check_prepared(args.prepared_root / label / 'prepared.json', REPO)
        candidates = path_candidates(args.ellipsoid_root, label, args.weights,
                                     args.strong_floor_threshold)
        if not all(path.is_file() for _, path in candidates):
            raise FileNotFoundError(f'{label}: missing geometry document')
        original = json.loads(candidates[0][1].read_text())
        cfg = ReprojectionFitConfig(**original['reprojection_fit']['config'])
        _, upright_report_path, _ = paths(args.ellipsoid_root, label, version(args.weights[0]))
        report = json.loads(upright_report_path.read_text())
        plans[label] = (meta, candidates, cfg, report)
        print(f'{label}: {", ".join(name for name, _ in candidates)}', flush=True)
    args.output.mkdir()
    rows, provenance = [], {}
    for label in args.sequences:
        meta, candidates, cfg, old = plans[label]
        with np.load(Path(meta['directory']) / 'pairs.npz', allow_pickle=False) as archive:
            pairs = dict(archive)
        size = meta['size']
        pixels, depth = batched_project(pairs['source_points'], pairs['source_ids'],
                                        pairs['poses'], pairs['intrinsics'])
        source_valid = inside(pixels, depth, size)
        pixels, ids = pixels[source_valid], pairs['source_ids'][source_valid]
        weights = balanced_weights(ids, pixels, size)
        template = old['canonical_mapping']
        sphere = old['sphere']
        sphere_map = world_y_regular_mapping(sphere, template)
        reference = sampling_metrics(pixels, ids, pairs['poses'], pairs['intrinsics'],
                                     sphere, sphere_map)['texels_per_pixel_min']
        supported = pairs['visibility'] == 1
        pair_ids = np.flatnonzero(supported)
        depth_ids = np.arange(len(pairs['source_points']))
        initial = None
        provenance[label] = {'prepared_key': meta['key'],
                             'upright_report_sha256': sha(paths(args.ellipsoid_root, label,
                                                                 version(args.weights[0]))[1]),
                             'documents': {name: {'path': str(path), 'sha256': sha(path)}
                                           for name, path in candidates}}
        for name, path in candidates:
            geometry = _source_geometry(json.loads(path.read_text()))
            mapping = world_y_regular_mapping(geometry, template)
            projected = evaluate_pairs(pairs, geometry, mapping, size)
            sampled = sampling_metrics(pixels, ids, pairs['poses'], pairs['intrinsics'],
                                       geometry, mapping)
            floor = floor_statistics(sampled['texels_per_pixel_min'], reference, weights)
            stronger_floor = floor_statistics(sampled['texels_per_pixel_min'], reference,
                                              weights, threshold=args.strong_floor_threshold)
            data = evaluate_geometry(pairs, geometry, pair_ids, depth_ids, cfg)
            if initial is None:
                initial = data['weighted_data_loss']
            radii = np.asarray(geometry['radii'])
            major = int(np.argmax(radii))
            row = {
                'sequence': label, 'candidate': name, 'path': str(path),
                'radius_0_m': float(radii[0]), 'radius_1_m': float(radii[1]),
                'radius_2_m': float(radii[2]), 'axis_1_over_axis_0': float(radii[1] / radii[0]),
                'major_axis_index': major,
                'major_world_x': float(geometry['rotation'][0, major]),
                'major_world_y': float(geometry['rotation'][1, major]),
                'major_world_z': float(geometry['rotation'][2, major]),
                'volume_m3': float(4 * np.pi / 3 * np.prod(radii)),
                'data_loss': data['weighted_data_loss'],
                'data_loss_percent_vs_original': 100 * (data['weighted_data_loss'] / initial - 1),
                'within_2pct_data_budget': bool(data['weighted_data_loss'] <= initial * 1.02 + 1e-6),
                'sampling_floor_loss': floor['loss'],
                'sampling_below_floor_fraction': floor['below_floor_fraction'],
                'sampling_ratio_median': floor['ratio_median'],
                'stronger_sampling_floor_loss': stronger_floor['loss'],
                'stronger_sampling_below_floor_fraction': stronger_floor['below_floor_fraction'],
                'supported_pair_valid_fraction': float(np.mean(projected['valid'][supported])),
                'source_sampling_valid_fraction': float(np.mean(sampled['valid'])),
                'max_camera_normalized_radius': data['max_camera_normalized_radius'],
                'camera_clearance_m': data['conservative_camera_clearance_m'],
            }
            for field in ('pixel_error', 'camera_angle_deg', 'surface_angle_deg', 'uv_distance'):
                for stat, value in finite_summary(projected[field], supported).items():
                    row[f'{field}_{stat}'] = value
            for stat, value in finite_summary(sampled['texels_per_pixel_min'],
                                              np.ones(len(pixels), dtype=bool)).items():
                row[f'texels_per_pixel_min_{stat}'] = value
            rows.append(row)
            print(f'{label} {name}: radii={radii.round(2).tolist()}, floor={floor["loss"]:.5g}, '
                  f'data+={row["data_loss_percent_vs_original"]:.3f}%, '
                  f'pixel p90={row["pixel_error_p90"]:.2f}', flush=True)
    write_csv(args.output / 'metrics.csv', rows)
    write_json(args.output / 'manifest.json', {
        'schema': 1, 'status': 'complete', 'scope': 'geometry-only; no training or new learned flow',
        'sequences': args.sequences, 'weights': args.weights,
        'strong_floor_threshold': args.strong_floor_threshold, 'provenance': provenance,
        'code_sha256': sha(Path(__file__)),
        'interpretation': 'Saved-source background LiDAR samples are sparse and not guaranteed static. '
            'UV differences are required geometric corrections, not measured learned network flow. '
            'Untrained geometry cannot be assigned a reconstruction-quality score.',
    })
    print(f'Wrote {len(rows)} geometry-only rows to {args.output}', flush=True)


if __name__ == '__main__':
    main()
