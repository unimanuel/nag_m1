#!/usr/bin/env python3
"""Compare fixed legacy-chart texture correction fields on completed ego geometries."""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import html
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))
from diagnostics.parallax_core import inside, intrinsics
from diagnostics.parallax_evaluate import batched_project, evaluate_pairs
from diagnostics.parallax_io import SEQUENCES, diagnostic, inventory, write_csv, write_json
from diagnostics.parallax_observations import check_prepared
from ego_repeatability import completed
from texture_sampling_diagnostics import sampling_metrics, statistics
from texture_flow_diagnostics import (NeighborhoodConfig, common_masks, correspondence_fields,
                                      local_field, neighborhoods, pair_weights)

FAMILIES = ('sphere', 'ar3', 'raw_v4', 'moderate_y', 'reprojection_v1', 'legacy_floor',
            'reprojection_new_radius_scale_v1')
REQUIRED = FAMILIES[:-1]
QUALITY = ('MPSNR', 'SSIM', 'LPIPS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_floor_runs(repo, labels, modes):
    """Use all three completed legacy repeats, with their actual effective layouts."""
    groups = {}
    for number in range(1, 5):
        root = repo / 'runs' / f'2026-09-25_ego_poles_repeatability_part{number}of4'
        path = root / 'manifest.json'
        if not path.is_file():
            continue
        for job in json.loads(path.read_text())['jobs']:
            if job['label'] not in labels or job['mode'] not in modes or job['orientation'] != 'legacy':
                continue
            if job['seed'] != 42 or job['group'] != 'same_seed':
                raise ValueError(f'Unexpected repeat treatment: {job["id"]}')
            attempts = [(p, completed(p, job['frame_count']))
                        for p in sorted((root / job['id']).glob('attempt_*'))]
            good = [(p, v) for p, v in attempts if v is not None]
            if len(good) != 1:
                raise ValueError(f'Expected one complete attempt: {root / job["id"]}')
            run, quality = good[0]
            cfgpath = run / 'training_config.yaml'
            cfg = yaml.safe_load(cfgpath.read_text())['NAGConfig']
            gp = run / 'initial/background_diagnostics/ellipsoid_geometry.txt'
            d = diagnostic(gp)
            t = d['texture_orientation']
            geometry = dict(center=d['center'], radii=d['effective_radii'], rotation=d['rotation_columns'])
            mapping = dict(mode=t['mode'], rotation=t.get('normalized_local_to_texture_columns', np.eye(3).tolist()),
                           u_offset=t.get('u_offset', 0),
                           y_flip=cfg.get('background_plane_kwargs', {}).get('sphere_uv_y_flip', False),
                           resolution_hw=d['color_mask_resolution'], visible_uv=d['visible_uv_texture'])
            if (cfg['seed'] != 42 or mapping['mode'] != 'legacy' or
                    mapping['visible_uv']['enabled'] != (job['mode'] == 'visuv')):
                raise ValueError(f'Saved treatment mismatch: {run}')
            _, first, stop = SEQUENCES[job['label']]
            sl = cfg['frame_indices_filter']
            if ((sl['start'], sl['stop'], sl.get('step') or 1) != (first, stop, 1)
                    or job['frame_count'] != stop-first):
                raise ValueError(f'Wrong repeat interval: {run}')
            sources = [path, cfgpath, gp] + [run / 'tracker' / f'train_epoch_Mean{m}.csv' for m in QUALITY]
            groups.setdefault((job['label'], job['mode']), []).append(
                dict(path=str(run.resolve()), geometry=geometry, mapping=mapping, metrics=quality,
                     camera=cfg['pinhole_camera_config'], cfg=cfg, repeat=job['repeat'],
                     sources={str(p): sha(p) for p in sources}))
    output = []
    for label in labels:
        for mode in modes:
            repeats = groups.get((label, mode), [])
            if len(repeats) != 3 or {r['repeat'] for r in repeats} != {1, 2, 3}:
                raise ValueError(f'{label} {mode}: three legacy floor repeats required')
            repeats.sort(key=lambda r: r['repeat'])
            first = repeats[0]
            for r in repeats[1:]:
                if (r['geometry'] != first['geometry'] or r['mapping'] != first['mapping'] or
                        r['camera'] != first['camera']):
                    raise ValueError(f'Repeat geometry/layout/calibration differs: {label} {mode}')
            metrics = {m: float(np.mean([r['metrics'][m] for r in repeats])) for m in QUALITY}
            output.append(dict(sequence=label, family='legacy_floor', mode=mode,
                id=f'legacy_floor_{mode}_benchmark', cohort='benchmark', path=first['path'],
                geometry=first['geometry'], mapping=first['mapping'], metrics=metrics,
                quality_n=3, quality_sd={m: float(np.std([r['metrics'][m] for r in repeats], ddof=1)) for m in QUALITY},
                quality_paths=[r['path'] for r in repeats], sources=[r['sources'] for r in repeats],
                camera=first['camera'], cfg=first['cfg']))
    return output


def select_runs(args):
    inv = inventory(REPO, args.sequences, args.data_root,
                    ['2026-09-12_reprojection_depth_v1', '2026-09-13_reprojection_depth_radius_scale'])
    runs = [r for r in inv['runs'] if r['cohort'] == 'benchmark' and r['mode'] in args.modes
            and r['family'] in FAMILIES]
    for r in runs:
        r['quality_n'] = 1
        r['quality_sd'] = dict.fromkeys(QUALITY)
        r['quality_paths'] = [r['path']]
        cfg = yaml.safe_load((Path(r['path']) / 'training_config.yaml').read_text())['NAGConfig']
        r['camera'], r['cfg'] = cfg['pinhole_camera_config'], cfg
        for m in QUALITY:
            with (Path(r['path']) / 'tracker' / f'train_epoch_Mean{m}.csv').open() as f:
                last = list(csv.DictReader(f, delimiter=';'))[-1]
            if float(last['step']) != 80 or not np.isfinite(float(last['value'])):
                raise ValueError(f'Expected finite final epoch-80 quality: {r["path"]}')
    runs.extend(load_floor_runs(REPO, args.sequences, args.modes))
    for label in args.sequences:
        for mode in args.modes:
            cohort = [r for r in runs if r['sequence'] == label and r['mode'] == mode]
            for family in REQUIRED:
                if sum(r['family'] == family for r in cohort) != 1:
                    raise ValueError(f'{label} {mode}: missing/ambiguous {family}')
            for r in cohort:
                mapping = r['mapping']
                if (mapping['mode'] != 'legacy' or mapping.get('u_offset', 0) != 0 or
                        not np.allclose(mapping['rotation'], np.eye(3), atol=1e-7)):
                    raise ValueError(f'Nonlegacy chart cannot enter this comparison: {r["path"]}')
                g = r['geometry']
                rot = np.asarray(g['rotation'])
                if (not np.isfinite(g['center']).all() or not np.isfinite(g['radii']).all()
                        or min(g['radii']) <= 0 or not np.allclose(rot.T@rot, np.eye(3), atol=1e-5)
                        or np.linalg.det(rot) < 0):
                    raise ValueError(f'Invalid geometry: {r["path"]}')
            if len({r['mapping']['y_flip'] for r in cohort}) != 1:
                raise ValueError(f'Mixed latitude convention: {label} {mode}')
            if mode == 'regular' and len({tuple(r['mapping']['resolution_hw']) for r in cohort}) != 1:
                raise ValueError(f'Mixed regular texture budgets: {label}')
    return runs


def load_pairs(meta, runs, label):
    with np.load(Path(meta['directory']) / 'pairs.npz', allow_pickle=False) as f:
        pairs = dict(f)
    _, first, stop = SEQUENCES[label]
    if meta['scope'] != 'full_interval' or meta['frames'] != list(range(first, stop)):
        raise ValueError(f'{label}: full-interval observation cache required')
    for r in runs:
        cfg, camera = r['cfg'], r['camera']
        if (cfg.get('normalize_camera') or cfg.get('position_spline_approximation') or
                np.any(camera.get('lens_distortion', [0]*5)) or
                not np.allclose(np.asarray(camera['position'])[first:stop], pairs['poses'], atol=1e-6, rtol=0) or
                not np.allclose(intrinsics(camera, meta['size']), pairs['intrinsics'], atol=1e-6, rtol=0)):
            raise ValueError(f'{label}: saved run/cache calibration mismatch: {r["path"]}')
        if r.get('size', meta['size']) != meta['size']:
            raise ValueError(f'{label}: render/cache size mismatch')
    take = pairs['visibility'] == 1
    for key in ('points', 'source', 'target', 'visibility', 'baseline', 'heading'):
        pairs[key] = pairs[key][take]
    pixels, zi = batched_project(pairs['points'], pairs['source'], pairs['poses'], pairs['intrinsics'])
    target_pixels, zj = batched_project(pairs['points'], pairs['target'], pairs['poses'], pairs['intrinsics'])
    if not len(pixels) or not (inside(pixels, zi, meta['size']) & inside(target_pixels, zj, meta['size'])).all():
        raise ValueError(f'{label}: supported observations empty or not inside both images')
    return pairs, pixels, zi, zj


def metric_arrays(fields, errors, sampling, uv_fit, tex_fit):
    values = dict(uv_displacement=np.where(errors['valid'], np.linalg.norm(fields['delta_uv'], axis=1), np.nan),
                  texel_displacement=np.where(errors['valid'], np.linalg.norm(fields['delta_texels'], axis=1), np.nan),
                  pixel_error=errors['pixel_error'], camera_angle_deg=errors['camera_angle_deg'],
                  surface_angle_deg=errors['surface_angle_deg'],
                  depth_relative_error=fields['depth_relative_error'],
                  seam_crossing=np.where(errors['valid'], fields['seam_crossing'].astype(float), np.nan),
                  near_seam=np.where(errors['valid'], fields['near_seam'].astype(float), np.nan),
                  pair_valid=errors['valid'], texture_valid=errors['valid'] & fields['tile_covered'],
                  sampling_valid=sampling['valid'])
    for prefix, fit in [('uv', uv_fit), ('texel', tex_fit)]:
        for key, array in fit.items():
            values[prefix+'_'+key] = array
    for key, array in sampling.items():
        if key != 'valid':
            values[key] = array
    values['depth_relative_error'] = np.where(errors['valid'], values['depth_relative_error'], np.nan)
    return values


def mask_key(name):
    if name in ('texels_per_pixel_min', 'texels_per_pixel_max', 'texel_area_per_pixel', 'sampling_condition',
                'sampling_vs_sphere_min_ratio', 'sampling_below_half_sphere'):
        return 'sampling_valid'
    if name.startswith('uv_'):
        if 'transport' in name:
            return 'uv_chart_fit_valid'
        if name in ('uv_gradient', 'uv_affine_residual'):
            return 'uv_image_fit_valid'
        return 'pair_valid'
    if name.startswith('texel_'):
        if 'transport' in name:
            return 'texel_chart_fit_valid'
        if name in ('texel_gradient', 'texel_affine_residual'):
            return 'texel_image_fit_valid'
        return 'texture_valid'
    return 'pair_valid'


def measurements(values, masks, weights, take=None):
    take = np.ones(len(weights), dtype=bool) if take is None else take
    out = {'pair_count': int(take.sum())}
    for key, mask in masks.items():
        out['common_'+key+'_fraction'] = float(np.average(mask[take], weights=weights[take])) if take.any() else None
    for key, array in values.items():
        if array.dtype == bool or key.endswith('support_count'):
            continue
        for prefix, mask in [('', np.ones(len(weights), dtype=bool)), ('common_', masks[mask_key(key)])]:
            selected = take & mask
            stats = statistics(array[selected], weights[selected])
            out.update({prefix+key+'_'+k: v for k, v in stats.items()})
    return out


def group_masks(pairs, pixels, size):
    yield 'all', 'all', np.ones(len(pixels), dtype=bool)
    for name, key, bounds in [('baseline_m', 'baseline', (0, .1, 1, 5, 15, np.inf)),
                              ('heading_deg', 'heading', (0, 5, 20, 45, 181))]:
        for low, high in zip(bounds[:-1], bounds[1:]):
            yield name, f'{low:g}-{high:g}', (pairs[key] >= low) & (pairs[key] < high)
    for i in np.unique(pairs['source']):
        yield 'source_frame', str(pairs['frames'][i]), pairs['source'] == i
    cells = np.clip((pixels/np.asarray(size)*[12, 8]).astype(int), [0, 0], [11, 7])
    for i in np.unique(cells[:, 1]*12+cells[:, 0]):
        yield 'image_cell', str(i), (cells[:, 1]*12+cells[:, 0]) == i


def plot_choices(pairs, count):
    candidates = []
    for i, j in np.unique(np.column_stack((pairs['source'], pairs['target'])), axis=0):
        mask = (pairs['source'] == i) & (pairs['target'] == j)
        if mask.sum() >= 12:
            candidates.append((float(pairs['heading'][mask][0]), float(pairs['baseline'][mask][0]), int(i), int(j)))
    if not candidates or not count:
        return []
    selected = [max(candidates)]
    by_baseline = sorted(candidates, key=lambda row: (row[1], row[2], row[3]))
    for fraction in np.linspace(.9, .1, max(count, 2)):
        item = by_baseline[int(round(fraction*(len(by_baseline)-1)))]
        if item not in selected:
            selected.append(item)
    return [(row[2], row[3]) for row in selected[:count]]


def make_plots(out, pairs, pixels, candidates, size, choices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = []
    for i, j in choices:
        take = (pairs['source'] == i) & (pairs['target'] == j)
        xy = pixels[take]
        # Shared colour limits across candidates, not per-image auto-normalization.
        keys = ('uv_displacement', 'texel_gradient', 'texel_affine_residual')
        limits = {}
        for key in keys:
            arrays = np.concatenate([entry['values'][key][take] for entry in candidates])
            arrays = arrays[np.isfinite(arrays)]
            limits[key] = max(float(np.quantile(arrays, .95)), 1e-10) if len(arrays) else 1.
        vector_scale = max(limits['uv_displacement'], 1e-10)/40
        for entry in candidates:
            r, values, fields = entry['run'], entry['values'], entry['fields']
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for ax in axes.flat:
                ax.scatter(xy[:, 0], xy[:, 1], s=5, color='#b8b8b8')
                ax.set(xlim=(0, size[0]), ylim=(size[1], 0), xlabel='Source image x (px)', ylabel='y (px)')
            delta = fields['delta_uv'][take]
            finite = values['pair_valid'][take]
            axes[0, 0].quiver(xy[finite, 0], xy[finite, 1], delta[finite, 0], delta[finite, 1],
                              angles='uv', scale_units='xy', scale=vector_scale, color='#1556a8', width=.003)
            axes[0, 0].set_title('Signed global UV displacement (arrows scaled; UV axes)')
            for ax, key, title in zip([axes[0, 1], axes[1, 0], axes[1, 1]], keys,
                    ['Global UV displacement norm', 'Local variation (texels / image px)', 'Local non-affine residual (texels)']):
                good = np.isfinite(values[key][take])
                artist = ax.scatter(xy[good, 0], xy[good, 1], c=values[key][take][good],
                                    s=15, cmap='viridis', vmin=0, vmax=limits[key])
                ax.set_title(title)
                fig.colorbar(artist, ax=ax)
            fig.suptitle(f'{r["sequence"]} | {r["family"]} {r["mode"]} | '
                         f'{pairs["frames"][i]} -> {pairs["frames"][j]} | '
                         f'baseline {pairs["baseline"][take][0]:.2f} m | heading {pairs["heading"][take][0]:.1f} deg\n'
                         'Sparse LiDAR samples only; gray = unsupported/invalid local measurement; blank = no observation')
            path = out / r['id'] / f'field_{pairs["frames"][i]:03d}_{pairs["frames"][j]:03d}.png'
            fig.savefig(path, dpi=120)
            plt.close(fig)
            plots.append(str(path))
    return plots


def comparison_outputs(output, rows, regions, plots, manifest):
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'by_region.csv', regions)
    deltas = []
    metrics = [key for key in rows[0] if key.startswith('common_') and key.endswith('_balanced_mean')]
    for r in rows:
        baseline = next(b for b in rows if b['sequence'] == r['sequence'] and b['mode'] == r['mode'] and b['family'] == 'sphere')
        if r['family'] == 'sphere':
            continue
        row = {key: r[key] for key in ('sequence', 'family', 'mode')}
        for key in metrics + list(QUALITY):
            row[key] = r[key]-baseline[key] if r.get(key) is not None and baseline.get(key) is not None else None
        deltas.append(row)
    write_csv(output / 'deltas_vs_sphere.csv', deltas)
    correlations = []
    for mode in sorted({r['mode'] for r in rows}):
        for metric in metrics:
            for quality in QUALITY:
                selected = [r for r in deltas if r['mode'] == mode and r.get(metric) is not None and r.get(quality) is not None]
                x, y = [r[metric] for r in selected], [r[quality] for r in selected]
                rho = float(spearmanr(x, y).statistic) if len(x) >= 5 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
                correlations.append(dict(mode=mode, diagnostic_delta=metric, quality_delta=quality,
                                         count=len(x), spearman_rho=rho))
    write_csv(output / 'exploratory_correlations.csv', correlations)
    manifest['plots'] = plots
    write_json(output / 'manifest.json', manifest)
    columns = ('sequence', 'family', 'mode', 'quality_n', 'MPSNR', 'SSIM', 'LPIPS',
               'common_uv_gradient_balanced_mean', 'common_texel_gradient_balanced_mean',
               'common_uv_transport_deviation_balanced_mean', 'common_texel_affine_residual_balanced_mean',
               'common_uv_image_fit_valid_fraction', 'common_texture_valid_fraction')
    def fmt(value):
        return 'n/a' if value is None else f'{value:.5g}' if isinstance(value, float) else str(value)
    table = '<table><thead><tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in columns) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>' + ''.join('<td>'+html.escape(fmt(row.get(k)))+'</td>' for k in columns) + '</tr>'
    table += '</tbody></table>'
    gallery = ''.join(f'<p><a href="{html.escape(str(Path(p).relative_to(output)))}">{html.escape(str(Path(p).relative_to(output)))}</a></p>' for p in plots)
    (output / 'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Texture flow diagnostic</title>'
        '<style>body{font:14px sans-serif;color:#111;background:#fff;margin:24px}table{border-collapse:collapse}'
        'td,th{border:1px solid #aaa;padding:7px}th{background:#e4eaf0}tr:nth-child(even){background:#f3f5f7}</style>'
        '<h1>Fixed-legacy-chart texture correspondence investigation</h1>'
        '<p>Geometry-only relative correction, not learned flow. Shared supported observations and common-valid subsets. '
        'Smaller variation alone is not a quality guarantee: check sampling density, coverage, depth, and reprojection too. '
        'Only two parent scenes; correlations are exploratory. Gray plot samples have no valid local measurement; '
        'blank image areas are unobserved, not zero error.</p>' + table + '<h2>Sparse field plots</h2>'+gallery+'</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument('--modes', nargs='+', choices=['regular', 'visuv'], default=['regular', 'visuv'])
    parser.add_argument('--prepared-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    parser.add_argument('--output', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_flow_v1'))
    parser.add_argument('--data-root', type=Path, default=REPO / 'data/datasets/waymo')
    parser.add_argument('--radius-pixels', type=float, default=80.)
    parser.add_argument('--neighbors', type=int, default=16)
    parser.add_argument('--min-support', type=int, default=6)
    parser.add_argument('--depth-absolute', type=float, default=1.)
    parser.add_argument('--depth-relative', type=float, default=.10)
    parser.add_argument('--plot-pairs', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (not np.isfinite([args.radius_pixels, args.depth_absolute, args.depth_relative]).all()
            or not 0 < args.radius_pixels <= 300 or not 4 <= args.min_support <= args.neighbors <= 64
            or args.depth_absolute < 0 or not 0 <= args.depth_relative <= 1 or not 0 <= args.plot_pairs <= 10):
        parser.error('Invalid neighborhood settings; use radius (0,300], 4<=support<=neighbors<=64, nonnegative depth gates, plot-pairs 0..10')
    if len(set(args.sequences)) != len(args.sequences) or len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate sequences or modes')
    if args.plot_pairs and importlib.util.find_spec('matplotlib') is None:
        parser.error('Matplotlib is required for plots; use the nag1 environment or --plot-pairs 0')
    args.output = args.output.resolve()
    if args.output.exists() and not args.dry_run:
        raise FileExistsError(f'{args.output} already exists; choose a fresh OUTPUT_ROOT/--output (no results overwritten)')
    config = NeighborhoodConfig(args.radius_pixels, args.neighbors, args.min_support,
                                args.depth_absolute, args.depth_relative)
    runs = select_runs(args)
    metadata = {}
    for label in args.sequences:
        meta = check_prepared(args.prepared_root / label / 'prepared.json', REPO)
        selected = [r for r in runs if r['sequence'] == label]
        pairs, _, _, _ = load_pairs(meta, selected, label)
        metadata[label] = meta
        print(f'[verified] {label}: {len(pairs["points"])} supported pairs, '
              f'{len(selected)} saved geometry/layout candidates', flush=True)
    if args.dry_run:
        print(f'Preflight passed: {len(runs)} comparisons; no files written. Output: {args.output}')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rows, region_rows, all_plots = [], [], []
    manifest = dict(schema=1, settings=asdict(config), sequences=args.sequences, modes=args.modes,
                    runs=[{k: v for k, v in r.items() if k not in ('cfg', 'camera')} for r in runs],
                    observations=metadata, code={str(p.relative_to(REPO)): sha(p) for p in
                        [Path(__file__), REPO / 'nag/texture_flow_diagnostics.py',
                         REPO / 'nag/texture_sampling_diagnostics.py', REPO / 'nag/scripts/ego_repeatability.py',
                         REPO / 'diagnose_ego_texture_flow.sh']},
                    versions={**{name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'PyYAML')},
                              'python': sys.version}, complete=False,
                    policy='Fixed actual legacy mapping and saved effective geometry; initial calibrated cameras. '
                    'No fitting, learned-flow/checkpoint queries, or image-error inputs. '
                    'Target-depth-supported, background-mask-selected LiDAR pairs, not guaranteed semantically static. '
                    'Equal source-frame, then target-pair, then occupied image-cell weights. Percentiles are unweighted. '
                    'Local affine fits use within-pair image neighbors gated by depth in both cameras; 3 IRLS solves. '
                    'Periodic longitude unwrapped locally; seam rates also reported. UV differences are relative '
                    'correction demands, not actual learned flow or a canonical reference-frame flow. '
                    'Visible-UV remaps initial texture lookup only, not global flow-network inputs. '
                    'Actual saved texture-budget units W,H; no synthetic crops or upright rotations. '
                    'Common masks are intersected across candidates within each mode. Invalid/unsupported values '
                    'are excluded and counted, never zero-filled. Correlations are exploratory, no significance claims.')
    write_json(args.output / 'manifest.json', manifest)
    for label in args.sequences:
        meta = metadata[label]
        selected = [r for r in runs if r['sequence'] == label]
        pairs, pixels, zi, zj = load_pairs(meta, selected, label)
        weights = pair_weights(pairs['source'], pairs['target'], pixels, meta['size'])
        idx, dist = neighborhoods(pixels, pairs['source'], pairs['target'], zi, zj, config)
        for mode in args.modes:
            candidates = []
            for r in [r for r in selected if r['mode'] == mode]:
                fields = correspondence_fields(pairs, r['geometry'], r['mapping'])
                errors = evaluate_pairs(pairs, r['geometry'], r['mapping'], meta['size'])
                sampling = sampling_metrics(pixels, pairs['source'], pairs['poses'], pairs['intrinsics'], r['geometry'], r['mapping'])
                candidates.append(dict(run=r, fields=fields, errors=errors, sampling=sampling))
            common_pair = np.logical_and.reduce([e['errors']['valid'] for e in candidates])
            common_texture = common_pair & np.logical_and.reduce([e['fields']['tile_covered'] for e in candidates])
            sphere_sampling = next(e['sampling']['texels_per_pixel_min'] for e in candidates
                                   if e['run']['family'] == 'sphere')
            for entry in candidates:
                fields, r = entry['fields'], entry['run']
                uv_fit = local_field(pixels, fields['uv_source'], fields['delta_uv'], idx, dist, common_pair, config, 1.)
                tex_fit = local_field(pixels, fields['texture_source'], fields['delta_texels'], idx, dist,
                                      common_texture, config, fields['texture_period'])
                entry['values'] = metric_arrays(fields, entry['errors'], entry['sampling'], uv_fit, tex_fit)
                if mode == 'regular':
                    small = entry['sampling']['texels_per_pixel_min']
                    good = np.isfinite(small) & np.isfinite(sphere_sampling) & (sphere_sampling > 0)
                    ratio = np.full(len(pixels), np.nan)
                    ratio[good] = small[good]/sphere_sampling[good]
                    entry['values']['sampling_vs_sphere_min_ratio'] = ratio
                    entry['values']['sampling_below_half_sphere'] = np.where(good, (ratio < .5).astype(float), np.nan)
                print(f'[measured] {label} {r["id"]}: '
                      f'{uv_fit["image_fit_valid"].sum()} local UV fits / {len(pixels)} pairs', flush=True)
            masks = common_masks([entry['values'] for entry in candidates])
            for entry in candidates:
                r, values = entry['run'], entry['values']
                row = {k: r[k] for k in ('sequence', 'family', 'mode', 'path', 'quality_n')}
                row.update(r['metrics'])
                row.update({m+'_sd': r['quality_sd'][m] for m in QUALITY})
                layout = r['mapping']['visible_uv']
                h, w = layout['resolution'] if layout['enabled'] else r['mapping']['resolution_hw']
                row.update(texture_height=h, texture_width=w, texture_texel_count=h*w,
                           uv_rectangle_fraction=layout['u_span']*layout['v_span'] if layout['enabled'] else 1.)
                g = r['geometry']
                camera_norm = np.linalg.norm(((pairs['poses'][:, :3, 3]-g['center']) @ np.asarray(g['rotation'])) / g['radii'], axis=1)
                row.update(max_camera_normalized_radius=float(camera_norm.max()),
                           conservative_camera_clearance_m=float(np.min(g['radii'])*(1-camera_norm.max())))
                row.update(measurements(values, masks, weights))
                rows.append(row)
                out = args.output / label / r['id']
                out.mkdir(parents=True)
                write_json(out / 'summary.json', row)
                np.savez_compressed(out / 'samples.npz', pixels=pixels, source=pairs['source'],
                    target=pairs['target'], baseline=pairs['baseline'], heading=pairs['heading'], weights=weights,
                    source_depth=zi, target_depth=zj, uv_source=entry['fields']['uv_source'],
                    uv_target=entry['fields']['uv_target'], delta_uv=entry['fields']['delta_uv'],
                    delta_texels=entry['fields']['delta_texels'], **values,
                    **{'common_'+k: v for k, v in masks.items()})
                for group, bin_name, take in group_masks(pairs, pixels, meta['size']):
                    if take.any():
                        region_rows.append(dict(sequence=label, family=r['family'], mode=mode,
                            group=group, bin=bin_name, **measurements(values, masks, weights, take)))
            all_plots.extend(make_plots(args.output / label, pairs, pixels, candidates, meta['size'],
                                       plot_choices(pairs, args.plot_pairs)))
        # Preserve completed work if a later interval fails; completion flag stays false.
        comparison_outputs(args.output, rows, region_rows, all_plots, copy.deepcopy(manifest))
    manifest['complete'] = True
    comparison_outputs(args.output, rows, region_rows, all_plots, manifest)
    print(f'Completed: {args.output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
