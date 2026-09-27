"""Offline, matched-ray sampling comparison of the completed pole repeats."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'nag'))
from ego_repeatability import completed, sha
from diagnostics.parallax_io import SEQUENCES, diagnostic, write_csv, write_json
from diagnostics.parallax_core import inside, intrinsics, intersect, surface_coordinates, surface_uv
from diagnostics.parallax_observations import check_prepared
from diagnostics.parallax_evaluate import batched_project
from texture_sampling_diagnostics import balanced_weights, sampling_metrics, statistics, texture_coordinates
from nag.model.ellipsoid_texture import world_y_texture_rotation


def saved(path):
    cfg = yaml.safe_load((path / 'training_config.yaml').read_text())['NAGConfig']
    d = diagnostic(path / 'initial/background_diagnostics/ellipsoid_geometry.txt')
    geometry = dict(center=d['center'], radii=d['effective_radii'], rotation=d['rotation_columns'])
    t = d['texture_orientation']
    mapping = dict(mode=t['mode'], rotation=t.get('normalized_local_to_texture_columns', np.eye(3).tolist()),
                   u_offset=t.get('u_offset', 0),
                   y_flip=cfg.get('background_plane_kwargs', {}).get('sphere_uv_y_flip', False),
                   resolution_hw=d['color_mask_resolution'], visible_uv=d['visible_uv_texture'])
    return cfg, geometry, mapping


def floor_summary(sigma, reference, weights, valid):
    valid = valid & np.isfinite(sigma) & np.isfinite(reference) & (sigma > 0) & (reference > 0)
    result = {'floor_valid_fraction': float(np.average(valid, weights=weights))}
    if not valid.any():
        return {**result, 'floor_below_half_fraction': None, 'floor_huber_loss': None}
    ratio = sigma[valid] / reference[valid]
    residual = np.maximum(0, np.log(.5 / ratio))
    huber = np.where(residual <= .2, .5 * residual**2, .2 * (residual - .1))
    result.update(floor_below_half_fraction=float(np.average(ratio < .5, weights=weights[valid])),
                  floor_huber_loss=float(np.average(huber, weights=weights[valid])))
    result.update({'floor_ratio_' + k: v for k, v in statistics(ratio, weights[valid]).items()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, default=REPO / 'runs')
    p.add_argument('--prepared-root', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    p.add_argument('--reference-manifest', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_v1/manifest.json'))
    p.add_argument('--output', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/poles_sampling_v1'))
    p.add_argument('--sequences', nargs='+', choices=list(SEQUENCES), default=list(SEQUENCES))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.output.exists() and not args.dry_run:
        raise FileExistsError(f'{args.output} already exists; choose a fresh --output to preserve results')
    historical = json.loads(args.reference_manifest.read_text())['inventory']['runs']
    groups, sources = {}, []
    for n in range(1, 5):
        root = args.runs_root / f'2026-09-25_ego_poles_repeatability_part{n}of4'
        manifest = root / 'manifest.json'
        sources.append(manifest)
        for job in json.loads(manifest.read_text())['jobs']:
            if job['label'] not in args.sequences:
                continue
            good = [(d, completed(d, job['frame_count'])) for d in (root / job['id']).glob('attempt_*')]
            good = [(d, v) for d, v in good if v is not None]
            if len(good) != 1:
                raise ValueError(f'Expected exactly one completed attempt: {job["id"]}')
            run, quality = good[0]
            cfg, geometry, mapping = saved(run)
            if cfg['seed'] != 42 or mapping['mode'] != job['orientation']:
                raise ValueError(f'Seed/orientation mismatch: {run}')
            if mapping['visible_uv']['enabled'] != (job['mode'] == 'visuv'):
                raise ValueError(f'Visible-UV mismatch: {run}')
            sources.extend([run / 'training_config.yaml', run / 'initial/background_diagnostics/ellipsoid_geometry.txt'])
            groups.setdefault((job['label'], job['mode'], job['orientation']), []).append(
                dict(path=str(run), cfg=cfg, geometry=geometry, mapping=mapping, quality=quality))
    metadata, references = {}, {}
    for label in args.sequences:
        pointer = args.prepared_root / label / 'prepared.json'
        meta = check_prepared(pointer, REPO)
        _, start, stop = SEQUENCES[label]
        if meta['scope'] != 'full_interval' or meta['frames'] != list(range(start, stop)):
            raise ValueError(f'Wrong observation interval: {label}')
        metadata[label] = meta
        sources.append(pointer)
        candidates = [r for r in historical if r['sequence'] == label and r['family'] == 'sphere'
                      and r['mode'] == 'regular' and r['cohort'] == 'benchmark']
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous/missing regular sphere reference: {label}')
        references[label] = candidates[0]
        reference = candidates[0]
        sphere_path = Path(reference['geometry']['provenance'])
        sphere_saved = diagnostic(sphere_path)
        if (reference['size'] != meta['size'] or reference['frames'] != meta['frames']
            or not np.array_equal(sphere_saved['center'], reference['geometry']['center'])
            or not np.array_equal([sphere_saved['radius']]*3, reference['geometry']['radii'])):
            raise ValueError(f'Stale/mismatched sphere reference: {label}')
        sources.append(sphere_path)
        geometry = None
        for mode in ('regular', 'visuv'):
            for orientation in ('legacy', 'world_y'):
                runs = groups[label, mode, orientation]
                if len(runs) != 3:
                    raise ValueError(f'Expected three repeats: {label} {mode} {orientation}')
                for r in runs:
                    if r['mapping'] != runs[0]['mapping']:
                        raise ValueError(f'Repeats have different mappings: {label}')
                    if geometry is not None and r['geometry'] != geometry:
                        raise ValueError(f'Physical geometry differs: {label}')
                    geometry = r['geometry']
                    if mode == 'regular' and r['mapping']['resolution_hw'] != reference['mapping']['resolution_hw']:
                        raise ValueError(f'Canonical sphere/ellipsoid texture budgets differ: {label}')
        print(f'[verified] {label}: 12 trainings, four deterministic mappings', flush=True)
    if args.dry_run:
        print(f'Preflight passed; no files written. Output: {args.output}')
        return
    rows, regions = [], []
    for label in args.sequences:
        meta = metadata[label]
        with np.load(Path(meta['directory']) / 'pairs.npz', allow_pickle=False) as f:
            points, ids, poses, k = (f[key] for key in ('source_points', 'source_ids', 'poses', 'intrinsics'))
        pixels, depth = batched_project(points, ids, poses, k)
        keep = inside(pixels, depth, meta['size'])
        pixels, ids = pixels[keep], ids[keep]
        if not len(ids):
            raise ValueError(f'No valid source observations: {label}')
        weights = balanced_weights(ids, pixels, meta['size'])
        measured = {}
        for mode in ('regular', 'visuv'):
            for orientation in ('legacy', 'world_y'):
                runs = groups[label, mode, orientation]
                run = runs[0]
                camera = run['cfg']['pinhole_camera_config']
                _, start, stop = SEQUENCES[label]
                if (run['cfg'].get('normalize_camera') or run['cfg'].get('position_spline_approximation')
                    or not np.allclose(poses, np.asarray(camera['position'])[start:stop], atol=1e-6, rtol=0)
                    or not np.allclose(k, intrinsics(camera, meta['size']), atol=1e-6, rtol=0)):
                    raise ValueError(f'Observation/run calibration mismatch: {label}')
                g, m = run['geometry'], run['mapping']
                sm = sampling_metrics(pixels, ids, poses, k, g, m)
                measured[mode, orientation] = sm
                row = dict(sequence=label, mode=mode, orientation=orientation, sample_count=len(ids))
                for name in ('MPSNR', 'SSIM', 'LPIPS'):
                    row[name + '_mean'] = float(np.mean([r['quality'][name] for r in runs]))
                    row[name + '_sd'] = float(np.std([r['quality'][name] for r in runs], ddof=1))
                for name, values in sm.items():
                    if name != 'valid':
                        row.update({name + '_' + key: value for key, value in statistics(values, weights).items()})
                local = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
                direction = np.einsum('nij,nj->ni', poses[ids, :3, :3], local)
                hits, _, valid = intersect(poses[ids, :3, 3], direction, g)
                uv = surface_uv(surface_coordinates(hits, g), m)
                row['intersection_valid_fraction'] = float(np.average(valid, weights=weights))
                row['near_seam_5deg_fraction'] = float(np.average(np.minimum(uv[valid, 0], 1-uv[valid, 0]) < 5/360, weights=weights[valid]))
                row['near_pole_10deg_fraction'] = float(np.average((uv[valid, 1] < 10/180) | (uv[valid, 1] > 170/180), weights=weights[valid]))
                _, covered = texture_coordinates(pixels, ids, poses, k, g, m)
                row['tile_covered_fraction'] = float(np.average(covered, weights=weights))
                layout = m['visible_uv']
                h, w = layout['resolution'] if layout['enabled'] else m['resolution_hw']
                row.update(texture_height=h, texture_width=w, texture_texel_count=h*w,
                           uv_rectangle_fraction=layout['u_span']*layout['v_span'] if layout['enabled'] else 1.)
                arrays = dict(pixels=pixels, source_ids=ids, weights=weights, uv=uv, **sm)
                if mode == 'regular':
                    sphere = references[label]['geometry']
                    ref_mapping = copy.deepcopy(m)
                    ref_mapping['rotation'] = (np.eye(3).tolist() if orientation == 'legacy' else
                        world_y_texture_rotation(torch.tensor(sphere['rotation'], dtype=torch.float64),
                                                 torch.tensor(sphere['radii'], dtype=torch.float64)).tolist())
                    ref = sampling_metrics(pixels, ids, poses, k, sphere, ref_mapping)
                    row.update(floor_summary(sm['texels_per_pixel_min'], ref['texels_per_pixel_min'], weights, sm['valid'] & ref['valid']))
                    row.update({'sphere_reference_min_' + key: value for key, value in statistics(ref['texels_per_pixel_min'], weights).items()})
                    arrays['sphere_reference_min'] = ref['texels_per_pixel_min']
                out = args.output / label / f'{mode}_{orientation}'
                out.mkdir(parents=True)
                np.savez_compressed(out / 'samples.npz', **arrays)
                cells = np.clip((pixels / np.asarray(meta['size']) * [12, 8]).astype(int), [0, 0], [11, 7])
                for grouping, bins in [('frame', ids), ('image_cell', cells[:, 1]*12+cells[:, 0])]:
                    for b in np.unique(bins):
                        take = bins == b
                        regions.append(dict(sequence=label, mode=mode, orientation=orientation, group=grouping,
                                            bin=int(b), **statistics(sm['texels_per_pixel_min'][take], weights[take])))
                rows.append(row)
                print(f'[measured] {label} {mode} {orientation}: {len(ids)} shared rays', flush=True)
        for mode in ('regular', 'visuv'):
            common = measured[mode, 'legacy']['valid'] & measured[mode, 'world_y']['valid']
            for row in [r for r in rows if r['sequence'] == label and r['mode'] == mode]:
                row['common_valid_fraction'] = float(np.average(common, weights=weights))
                for name, values in measured[mode, row['orientation']].items():
                    if name != 'valid':
                        row['common_' + name + '_mean'] = float(np.average(values[common], weights=weights[common])) if common.any() else None
    deltas = []
    for label in args.sequences:
        for mode in ('regular', 'visuv'):
            a, b = [next(r for r in rows if (r['sequence'], r['mode'], r['orientation']) == (label, mode, o)) for o in ('legacy', 'world_y')]
            deltas.append(dict(sequence=label, mode=mode, **{key: b[key]-a[key] for key in a
                if isinstance(a[key], (int, float)) and isinstance(b.get(key), (int, float))}))
    write_csv(args.output / 'metrics.csv', rows)
    write_csv(args.output / 'upright_minus_legacy.csv', deltas)
    write_csv(args.output / 'sampling_by_region.csv', regions)
    write_json(args.output / 'manifest.json', dict(observations=metadata, references=references,
        sources={str(f): sha(f) for f in sources + [args.reference_manifest, Path(__file__), REPO / 'nag/texture_sampling_diagnostics.py']},
        policy='All cached in-image background-mask-selected source LiDAR pixels, shared across mappings; not guaranteed static. '
        'Initial cameras, no learned flow. Equal occupied-cell/frame weights. Percentiles are unweighted. '
        'Floor measured only for regular mapping against a same-resolution sphere with matching pole policy; '
        'Visible-UV reports actual saved allocation, not a synthetic sphere crop. Common-valid statistics prevent coverage selection bias. '
        'Geometry and mapping verified identical across repeats; quality means and sample SD use all three repeats. '
        'Near-pole/seam rates are descriptive; floor violations do not prove training degradation.'))
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
