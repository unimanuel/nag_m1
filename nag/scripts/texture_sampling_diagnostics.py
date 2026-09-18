#!/usr/bin/env python3
"""Compare saved geometry texture correspondence and sampling; no fitting/training."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_io import SEQUENCES, inventory, write_csv, write_json
from diagnostics.parallax_observations import check_prepared
from diagnostics.parallax_evaluate import batched_project, evaluate_pairs
from diagnostics.parallax_core import inside
from texture_sampling_diagnostics import balanced_weights, sampling_metrics, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument('--prepared-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--output', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_v1'))
    parser.add_argument('--data-root', type=Path, default=REPO / 'data/datasets/waymo')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--difference-step', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.difference_step <= 2:
        parser.error('--difference-step must be in (0, 2] pixels')
    inv = inventory(REPO, args.sequences, args.data_root,
                    ['2026-09-12_reprojection_depth_v1', '2026-09-13_reprojection_depth_radius_scale'])
    inv['runs'] = [r for r in inv['runs'] if r['cohort'] == 'benchmark']
    metadata = {}
    for label in args.sequences:
        meta = check_prepared(args.prepared_root / label / 'prepared.json', REPO)
        if meta['scope'] != 'full_interval':
            raise ValueError(f'{label}: full-interval observation cache required')
        metadata[label] = meta
        runs = [r for r in inv['runs'] if r['sequence'] == label]
        print(label + ': ' + ', '.join(r['id'] for r in runs), flush=True)
        if not any(r['family'] == 'reprojection_v1' for r in runs):
            raise ValueError(f'{label}: original reprojection runs missing')
        if label in ('W191_0_40', 'W254_0_37', 'W254_139_195') and not any(
                r['family'] == 'reprojection_new_radius_scale_v1' for r in runs):
            raise ValueError(f'{label}: scale-limited runs missing')
    if args.dry_run:
        print(f'{len(inv["runs"])} runs; output {args.output}; no files written')
        return
    rows, groups = [], []
    for label in args.sequences:
        meta = metadata[label]
        with np.load(Path(meta['directory']) / 'pairs.npz', allow_pickle=False) as cache:
            pairs = dict(cache)
        runs = [r for r in inv['runs'] if r['sequence'] == label]
        size = runs[0]['size']
        if size != meta['size'] or runs[0]['frames'] != meta['frames']:
            raise ValueError(f'{label}: cached calibration size/frame mismatch')
        pixels, z = batched_project(pairs['source_points'], pairs['source_ids'], pairs['poses'], pairs['intrinsics'])
        source_valid = inside(pixels, z, size)
        pixels, ids = pixels[source_valid], pairs['source_ids'][source_valid]
        weights = balanced_weights(ids, pixels, size)
        supported = pairs['visibility'] == 1
        pair_pixels, _ = batched_project(pairs['points'][supported], pairs['source'][supported], pairs['poses'], pairs['intrinsics'])
        pair_weights = balanced_weights(pairs['source'][supported], pair_pixels, size)
        measured = []
        for run in runs:
            sampling = sampling_metrics(pixels, ids, pairs['poses'], pairs['intrinsics'],
                                        run['geometry'], run['mapping'], args.difference_step)
            correspondence = evaluate_pairs(pairs, run['geometry'], run['mapping'], size)
            row = {k: run[k] for k in ('sequence', 'family', 'mode', 'path')}
            row.update(run['metrics'])
            for name, values in sampling.items():
                if name == 'valid':
                    continue
                for key, value in statistics(values, weights).items():
                    row[name + '_' + key] = value
            for name in ('pixel_error', 'camera_angle_deg', 'surface_angle_deg', 'texture_texel_distance'):
                for key, value in statistics(correspondence[name][supported], pair_weights).items():
                    row[name + '_' + key] = value
            row['supported_pair_count'] = int(supported.sum())
            row['sampling_source_count'] = len(ids)
            row['tile_covered_fraction'] = float(correspondence['tile_covered'][supported].mean())
            for threshold in (.25, .5, 1.):
                valid = sampling['valid']
                row[f'min_sampling_below_{threshold}_fraction'] = (
                    float(np.average(sampling['texels_per_pixel_min'][valid] < threshold, weights=weights[valid]))
                    if valid.any() else None)
            rows.append(row)
            measured.append((run, row, sampling, correspondence))
            cells = np.clip((pixels / np.asarray(size) * [12, 8]).astype(int), [0, 0], [11, 7])
            for group, keys in [('frame', ids), ('image_cell', cells[:, 1] * 12 + cells[:, 0])]:
                for key in np.unique(keys):
                    take = keys == key
                    record = {'sequence': label, 'family': run['family'], 'mode': run['mode'],
                              'group': group, 'bin': int(key)}
                    for name, values in sampling.items():
                        if name != 'valid':
                            record.update({name + '_' + k: v for k, v in statistics(values[take], weights[take]).items()})
                    groups.append(record)
            out = args.output / label / run['id']
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'samples.npz', pixels=pixels, source_ids=ids, weights=weights,
                                pair_source=pairs['source'][supported], pair_target=pairs['target'][supported],
                                pair_points=pairs['points'][supported], pair_weights=pair_weights,
                                **sampling, **{f'pair_{k}': correspondence[k][supported] for k in correspondence})
            write_json(out / 'summary.json', row)
            print(f'[measured] {label} {run["id"]}', flush=True)
        # Also compare identical finite observations across geometries within each
        # layout. Otherwise a small visible tile can appear better by dropping hard rays.
        for mode in ('regular', 'visuv'):
            cohort = [entry for entry in measured if entry[0]['mode'] == mode]
            if not cohort:
                continue
            common_sampling = np.logical_and.reduce([entry[2]['valid'] for entry in cohort])
            common_pairs = np.logical_and.reduce([
                np.isfinite(entry[3]['texture_texel_distance'][supported]) for entry in cohort])
            for run, row, sampling, correspondence in cohort:
                row['common_sampling_fraction'] = float(common_sampling.mean())
                row['common_pair_fraction'] = float(common_pairs.mean())
                for name, values in sampling.items():
                    if name != 'valid':
                        row['common_' + name + '_balanced_mean'] = (
                            float(np.average(values[common_sampling], weights=weights[common_sampling]))
                            if common_sampling.any() else None)
                for name in ('pixel_error', 'camera_angle_deg', 'surface_angle_deg', 'texture_texel_distance'):
                    row['common_' + name + '_balanced_mean'] = (
                        float(np.average(correspondence[name][supported][common_pairs], weights=pair_weights[common_pairs]))
                        if common_pairs.any() else None)
                write_json(args.output / label / run['id'] / 'summary.json', row)
        write_csv(args.output / 'metrics.csv', rows)
        write_csv(args.output / 'sampling_by_region.csv', groups)
    # Within-sequence, within-layout sphere differences avoid raw cross-scene correlations.
    deltas = []
    metric_names = [k for k in rows[0] if k.endswith('_balanced_mean')]
    for row in rows:
        baseline = next((r for r in rows if r['sequence'] == row['sequence'] and
                         r['mode'] == row['mode'] and r['family'] == 'sphere'), None)
        if baseline is None or row['family'] == 'sphere':
            continue
        delta = {k: row[k] for k in ('sequence', 'family', 'mode')}
        for key in metric_names + ['MPSNR', 'SSIM', 'LPIPS']:
            delta[key] = row[key] - baseline[key] if row[key] is not None and baseline[key] is not None else None
        deltas.append(delta)
    write_csv(args.output / 'deltas_vs_sphere.csv', deltas)
    correlations = []
    for mode in ('regular', 'visuv'):
        for metric in metric_names:
            for quality in ('MPSNR', 'SSIM', 'LPIPS'):
                selected = [d for d in deltas if d['mode'] == mode and d[metric] is not None]
                x, y = [d[metric] for d in selected], [d[quality] for d in selected]
                rho = float(spearmanr(x, y).statistic) if len(x) >= 5 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
                correlations.append({'mode': mode, 'diagnostic_delta': metric, 'quality_delta': quality,
                                     'count': len(x), 'spearman_rho': rho})
    write_csv(args.output / 'exploratory_correlations.csv', correlations)
    write_json(args.output / 'manifest.json', {'inventory': inv, 'observations': metadata,
               'difference_step_pixels': args.difference_step,
               'code': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), REPO / 'nag/texture_sampling_diagnostics.py']},
               'policy': 'Saved effective geometry and texture layout; shared initial cameras; no trained flow. '
               'Source rays are background-mask selected, not guaranteed semantically static. '
               'Correspondences use target-LiDAR-supported pairs. Sampling is measured at source LiDAR pixels. '
               'Balanced means equalize occupied cells within frames; invalid values are excluded and counted. '
               'Sub-texel thresholds are descriptive, not proven quality cutoffs. '
               'Correlations are exploratory, dependent observations; no significance claims.'})
    print(f'Completed: {args.output}/metrics.csv', flush=True)


if __name__ == '__main__':
    main()
