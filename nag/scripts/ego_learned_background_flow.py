#!/usr/bin/env python3
"""Read-only, parity-gated inspection of learned background texture flow."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_core import regions
from diagnostics.parallax_io import SEQUENCES, load_mask, mask_paths, write_csv, write_json
from learned_flow_diagnostics import association, flow_fields, parity, statistics

DEFAULT_SEQUENCES = ['W191_0_40', 'W254_0_37', 'W254_139_195', 'W254_86_137']
DEFAULT_FAMILIES = ['sphere', 'reprojection_v1', 'legacy_floor']


def file_record(path, hashed=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if hashed:
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def check_records(records):
    for expected in records:
        actual = file_record(expected['path'], 'sha256' in expected)
        if actual != expected:
            raise ValueError(f'Input changed since inventory/prepare: {expected["path"]}')


def check_code(manifest):
    for path, digest in manifest['code'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Diagnostic/model code changed during the batch: {path}')


def completed_result(folder, job):
    required = ('result.json', 'verification.json', 'fields.npz', 'metrics.csv', 'by_frame.csv')
    if not all((folder/name).is_file() and (folder/name).stat().st_size > 0 for name in required):
        return False
    result = json.loads((folder/'result.json').read_text())
    return (result.get('complete') and result.get('parity_passed') and result.get('job') == job['id']
            and result.get('frames') == [job['frames'][i] for i in job['indices']]
            and json.loads((folder/'verification.json').read_text()).get('passed'))


def image(path, size=None):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if size and im.size != tuple(size):
            im = im.resize(tuple(size), Image.Resampling.BILINEAR)
        return np.asarray(im)


def saved_layers(run, frames):
    complete = run / 'final/complete'
    backgrounds = [p for p in (run / 'final').iterdir() if p.is_dir() and 'Background Plane' in p.name]
    if len(backgrounds) != 1:
        raise ValueError(f'Expected one saved background layer: {run}')
    output = []
    for local, frame in enumerate(frames):
        composites = [p for p in complete.glob('*_t.png') if int(p.name.split('_')[0]) == local]
        layers = [p for p in backgrounds[0].glob('*_t_*.png')
                  if int(p.name.split('_')[0]) == local and int(p.stem.split('_')[-1]) == frame]
        if len(composites) != 1 or len(layers) != 1:
            raise ValueError(f'Missing/ambiguous saved images: {run}, local frame {local}')
        output.append(dict(composite=str(composites[0]), background=str(layers[0])))
    return output


def jobs_from_manifest(args):
    manifest = json.loads(args.inventory.read_text())
    if not manifest.get('complete'):
        raise ValueError('The previous texture-flow diagnostic must have completed')
    jobs = []
    for label in args.sequences:
        segment, first, stop = SEQUENCES[label]
        for mode in args.modes:
            for family in args.families:
                matches = [r for r in manifest['runs'] if r['sequence'] == label
                           and r['family'] == family and r['mode'] == mode]
                if len(matches) != 1:
                    raise ValueError(f'Missing/ambiguous inventory: {label} {family} {mode}')
                record = matches[0]
                # Old manifest mixes stat provenance and repeat SHA dictionaries.
                for source in record['sources']:
                    if 'path' in source:
                        check_records([source])
                    else:
                        for path, digest in source.items():
                            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                                raise ValueError(f'Inventory input changed: {path}')
                paths = record['quality_paths'] if family == 'legacy_floor' else [record['path']]
                repeats = args.floor_repeats if family == 'legacy_floor' else [1]
                for repeat in repeats:
                    if repeat > len(paths):
                        raise ValueError(f'Missing repeat {repeat}: {label} {mode}')
                    run = Path(paths[repeat-1]).resolve()
                    checkpoints = list((run / 'checkpoints').glob('final-*.ckpt'))
                    if len(checkpoints) != 1 or checkpoints[0].name != 'final-80.ckpt':
                        raise ValueError(f'Expected exactly final-80.ckpt (do not substitute another checkpoint): {run}')
                    frames = list(range(first, stop))
                    if args.max_frames:
                        indices = np.unique(np.linspace(0, len(frames)-1, min(args.max_frames, len(frames))).round().astype(int))
                    else:
                        indices = np.arange(len(frames))
                    layers = saved_layers(run, frames)
                    with Image.open(layers[0]['composite']) as im:
                        size = im.size
                    data = REPO / 'data/datasets/waymo' / f'segment-{segment}'
                    sources = [file_record(run / 'training_config.yaml', True), file_record(checkpoints[0])]
                    # All saved frames matter for the comparison, even with a
                    # limited flow grid; verify stat provenance on resume.
                    for local in indices:
                        sources.extend(file_record(p) for p in layers[local].values())
                        sources.append(file_record(data / 'images_undistorted' / f'{frames[local]:03d}.png'))
                        sources.extend(file_record(p) for p in mask_paths(data, frames[local]))
                    metrics = {}
                    for metric in ('MPSNR', 'SSIM', 'LPIPS'):
                        path = run / 'tracker' / f'train_epoch_Mean{metric}.csv'
                        with path.open() as f:
                            last = list(csv.DictReader(f, delimiter=';'))[-1]
                        if float(last['step']) != 80 or not np.isfinite(float(last['value'])):
                            raise ValueError(f'Expected final epoch-80 metric: {path}')
                        metrics[metric] = float(last['value'])
                        sources.append(file_record(path, True))
                    job_id = f'{label}_{family}_{mode}_repeat{repeat:02d}'
                    jobs.append(dict(id=job_id, sequence=label, family=family, mode=mode, repeat=repeat,
                        run=str(run), checkpoint=str(checkpoints[0]), size=list(size), frames=frames,
                        indices=indices.tolist(), layers=layers, data=str(data), metrics=metrics,
                        geometry=record['geometry'], mapping=record['mapping'], inputs=sources))
    return jobs[:args.limit_jobs] if args.limit_jobs else jobs


def render_verification(model, config, times, job, folder, settings):
    import torch
    take = job['indices']
    indices = np.unique(np.asarray(take)[np.linspace(0, len(take)-1, min(settings['verify_frames'], len(take))).round().astype(int)])
    rows = []
    for index in indices:
        # Same quantization as NAGCallback: clamp then truncate, not round.
        t = times[index:index+1]
        for layer, objects in [('composite', None), ('background', [model.get_background_plane()])]:
            output = model.generate_outputs(config, t=t, resolution=job['size'], objects=objects,
                                            progress_bar=False, device=model.device)
            a = (output[0, :3]*255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            saved = image(job['layers'][index][layer])
            measurements = parity(a, saved, settings['parity_mean_bytes'], settings['parity_p99_bytes'])
            rows.append(dict(local_frame=int(index), frame=job['frames'][index], layer=layer, **measurements))
            path = folder / 'verification'
            path.mkdir(exist_ok=True)
            # A side-by-side reference, re-render and exaggerated absolute diff.
            difference = np.clip(np.abs(a.astype(float)-saved)*16, 0, 255).astype(np.uint8)
            Image.fromarray(np.concatenate([saved, a, difference], axis=1)).save(path / f'{index:03d}_{layer}.png')
            print(f'  parity {index:03d} {layer}: mean={measurements["mean_bytes"]:.4g}, '
                  f'p99={measurements["p99_bytes"]:.4g}, passed={measurements["passed"]}', flush=True)
    passed = all(r['passed'] for r in rows)
    write_json(folder / 'verification.json', dict(passed=passed, thresholds=settings, rows=rows))
    write_csv(folder / 'verification.csv', rows)
    if not passed:
        raise RuntimeError('Saved-image reproduction failed; flow analysis deliberately not performed. See verification.json.')


def checkpoint_model(job, folder):
    import torch
    from nag.run.nag_runner import NAGRunner
    from tools.serialization.json_convertible import JsonConvertible

    class DiagnosticRunner(NAGRunner):
        def create_trainer(self, **kwargs):
            return None  # No training, callbacks, checkpoint writes or run loggers.

        def create_world(self, *args, **kwargs):
            if not kwargs.get('proxy_init'):
                raise RuntimeError('This diagnostic loader must only construct a checkpoint proxy world')
            return super().create_world(*args, **kwargs)

        def _compute_visible_plane_center_sphere_radius(self, **kwargs):
            # NAGRunner.load uses empty placeholder masks. The original radius
            # heuristic cannot run on those masks, and must not refit geometry.
            # This seeds ONLY the proxy constructor. load_from_checkpoint then
            # replaces the registered sphere_radius buffer with its exact saved
            # value; the render-parity gate verifies the resulting model.
            return torch.as_tensor(job['geometry']['radii'][0], dtype=self.config.dtype)

        _compute_visible_plane_corner_sphere_radius = _compute_visible_plane_center_sphere_radius

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the trained tiny-cuda-nn networks; activate the nag1 environment.')
    config = JsonConvertible.load_from_file(Path(job['run']) / 'training_config.yaml')
    # Proxy-world construction may write initialization diagnostics. Redirect
    # those to our NEW folder, never to the source run.
    config.output_path = folder / 'runtime'
    config.reevaluate_output_path = False
    config.checkpoint_path = None
    config.checkpoint_tracker_path = None
    config.prepare(create_output_path=False)
    if (Path(config.images_path).resolve() != (Path(job['data'])/'images_undistorted').resolve()
            or Path(config.masks_path).resolve() != (Path(job['data'])/'masks').resolve()):
        raise ValueError('Saved training image/mask paths differ from the diagnostic comparison data')
    config.output_path.mkdir(parents=True, exist_ok=True)
    runner = DiagnosticRunner(config)
    runner.load(job['checkpoint'], callbacks=[], logger=False)
    model = runner.model.to('cuda').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    times = runner.dataset.frame_timestamps.to(device=model.device, dtype=model.camera._translation.dtype)
    if len(times) != len(job['frames']):
        raise ValueError('Checkpoint/dataset interval differs from selected saved images')
    return runner, config, model, times


def query_flow(model, times, index, coordinates, factor):
    """Capture the background modality's native flow before pixel conversion.

    This wraps the *actual* modality call rather than reimplementing the learned
    camera, surface intersection, network or flow-reference-time convention.
    The instance wrapper is removed immediately after the read-only call.
    """
    import torch
    from nag.model.timed_discrete_scene_node_3d import global_to_local
    bg = model.get_background_plane()
    t, _ = model._get_t_with_flow_reference_times(times[index:index+1])
    selected = int(torch.nonzero(t == times[index], as_tuple=False).item())
    count = len(model.objects)-int(model.config.has_background_plane)-int(model.config.has_camera_aberration_plane)
    enabled = torch.zeros(count, dtype=torch.bool, device=model.device)
    uv = torch.as_tensor(coordinates.reshape(-1, 2), dtype=times.dtype, device=model.device)
    captures = []
    original = bg.forward_modality

    def capture(points, *args, **kwargs):
        rgb, alpha, native = original(points, *args, **kwargs)
        before = bg._global_points_to_sphere_uv(points, kwargs['global_position'])
        after = bg._apply_uv_bounds(before-.5+native)+.5
        moved = bg.get_flowed_global_points(points, native, kwargs['global_position'])
        positions, _, _, camera_idx = model._assemble_object_rigid_data(t, enabled, True)
        local = global_to_local(positions[camera_idx], moved, v_include_time=True)[..., :3]
        hit = kwargs['is_inside']
        projected = torch.isfinite(local).all(-1) & (local[..., 2] > 0)
        texture_size = np.asarray(bg.initial_rgb.shape[-2:][::-1], dtype=float)-int(bg.align_corners)
        if bool(bg.visible_uv_texture_enabled):
            covered, tile_u, _ = bg._visible_uv_texture_coordinates(after-.5)
            before_covered, before_u, _ = bg._visible_uv_texture_coordinates(before-.5)
            covered &= before_covered
            # Reject the finite-difference stencil across the tile's discontinuous
            # longitude branch, even when both ends happen to be in a full tile.
            bu = torch.remainder(before[..., 0]-bg.visible_uv_texture_bounds[0], 1.)
            au = torch.remainder(after[..., 0]-bg.visible_uv_texture_bounds[0], 1.)
            covered &= torch.abs(au-bu) <= .5
            bounds = bg.visible_uv_texture_bounds.cpu().numpy()
            scale = texture_size / bounds[[1, 3]]
            branch = torch.stack([before_u[:, selected], tile_u[:, selected]], -1).cpu().numpy()
        else:
            covered = torch.ones_like(hit)
            scale = texture_size
            branch = np.zeros((len(uv), 2))
        captures.append(dict(before=before[:, selected].cpu().numpy(), after=after[:, selected].cpu().numpy(),
                             native=native[:, selected].cpu().numpy(), hit=hit[:, selected].cpu().numpy(),
                             projected=projected[:, selected].cpu().numpy(), tile=covered[:, selected].cpu().numpy(),
                             scale=scale, branch=branch))
        return rgb, alpha, native

    # Keep the original class method untouched for other objects/processes.
    bg.forward_modality = capture
    try:
        _, _, pixels = model.object_modality_forward(uv, t, enabled, enabled_background=True,
                                                     color=False, alpha=False, flow=True)
    finally:
        del bg.forward_modality
    if len(captures) != 1 or pixels.shape[0] != 1:
        raise ValueError('Expected exactly one curved background modality query')
    capture = captures[0]
    result = {k: capture[k].reshape(5, -1, 2) for k in ('before', 'after', 'native')}
    result.update({k: capture[k].reshape(5, -1) for k in ('hit', 'projected', 'tile')})
    branch = capture['branch'].reshape(5, -1, 2)
    if bool(bg.visible_uv_texture_enabled):
        continuous = (np.ptp(branch, axis=0) < .5).all(-1)
        result['tile'] &= continuous[None]
    result['pixel'] = pixels[0, :, selected].cpu().numpy().reshape(5, -1, 2)*factor
    result['texel_scale'] = capture['scale']
    return result


def table_rows(job, arrays, selections, frames):
    rows = []
    for region, selection in selections.items():
        for name, values in arrays.items():
            stats = statistics(values, selection, frames)
            # Within-frame associations avoid conflating elapsed training-time
            # deformation with unrelated frame-to-frame reconstruction changes.
            correlations = [association(values[i], arrays['rgb_mae'][i], selection[i]) for i in range(len(frames))]
            valid = [x for x in correlations if x is not None]
            rows.append(dict(job=job['id'], sequence=job['sequence'], family=job['family'], mode=job['mode'],
                repeat=job['repeat'], region=region, measure=name, **stats,
                mean_within_frame_spearman_with_rgb_mae=float(np.mean(valid)) if valid else None,
                correlation_frames=len(valid), **job['metrics']))
    return rows


def plot_fields(folder, job, fields, masks, frames, x, y):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    names = ['rgb_mae', 'native_uv_magnitude', 'image_pixel_magnitude', 'native_uv_gradient',
             'image_pixel_gradient', 'transport_deviation']
    for index in np.unique(np.linspace(0, len(frames)-1, min(3, len(frames))).round().astype(int)):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, name in zip(axes.flat, names):
            values = fields[name][index].reshape(len(y), len(x)).copy()
            values[~masks['background'][index].reshape(values.shape)] = np.nan
            cap = float(np.quantile(fields[name][np.isfinite(fields[name])], .95)) if np.isfinite(fields[name]).any() else 1.
            cap = max(cap, 1e-10)
            cmap = plt.get_cmap('magma').copy()
            cmap.set_bad('#bdbdbd')
            im = axis.imshow(values, vmin=0, vmax=cap, cmap=cmap, extent=[x[0], x[-1], y[-1], y[0]], interpolation='nearest')
            figure.colorbar(im, ax=axis, shrink=.75)
            axis.set_title(name+' (background only)')
        figure.suptitle(f'{job["id"]} — frame {frames[index]}\n'
                       'Color scales are per run: use comparisons CSV, not cross-panel color, to compare geometries.')
        figure.savefig(folder / f'frame_{frames[index]:03d}_fields.png', dpi=120)
        plt.close(figure)


def worker(manifest_path, job_id):
    import torch
    manifest = json.loads(manifest_path.read_text())
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    settings = manifest['settings']
    folder = manifest_path.parent / job_id
    folder.mkdir(exist_ok=True)
    write_json(folder / 'result.json', dict(complete=False, parity_passed=False, job=job_id))
    check_code(manifest)
    check_records(job['inputs'])
    with torch.no_grad():
        runner, config, model, times = checkpoint_model(job, folder)
        render_verification(model, config, times, job, folder, settings)
        width, height = job['size']
        stride, step = settings['stride'], settings['difference_step']
        x, y = np.arange(stride//2, width-stride//2, stride), np.arange(stride//2, height-stride//2, stride)
        grid = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
        offsets = np.asarray([[0, 0], [step, 0], [-step, 0], [0, step], [0, -step]])
        if (grid[None]+offsets[:, None] < 0).any() or (grid[None]+offsets[:, None] >= [width, height]).any():
            raise ValueError('Finite-difference step exceeds grid border')
        # Mirror RegularUVGridSampler's linspace(0, camera_size, output_size),
        # which differs from ordinary integer-pixel camera coordinates.
        camera_size = model.camera._image_resolution.flip(-1).cpu().numpy()
        to_camera = camera_size / (np.asarray(job['size'])-1)
        coordinates = (grid[None]+offsets[:, None])*to_camera
        arrays, flags, selections, native, pixel, uvs, frames = {}, {}, {}, [], [], [], []
        frame_rows = []
        for local in job['indices']:
            frame = job['frames'][local]
            query = query_flow(model, times, local, coordinates, 1/to_camera)
            values, valid = flow_fields(**{k: query[k] for k in ('before', 'after', 'native', 'pixel', 'hit', 'projected', 'tile', 'texel_scale')}, step=step)
            foreground = load_mask(Path(job['data']), frame, (width, height))
            full_regions = regions(foreground, settings['boundary_width'])
            selected = {name: mask[y[:, None], x[None]].ravel() for name, mask in full_regions.items()}
            selected['all'] = np.ones(len(grid), dtype=bool)
            target = image(Path(job['data']) / 'images_undistorted' / f'{frame:03d}.png', (width, height))
            saved = image(job['layers'][local]['composite'])
            bg_saved = image(job['layers'][local]['background'])
            mae = np.abs(saved.astype(float)-target).mean(-1)/255
            mse = ((saved.astype(float)-target)/255)**2
            bg_mae = np.abs(bg_saved.astype(float)-target).mean(-1)/255
            values.update(rgb_mae=mae[y[:, None], x[None]].ravel(),
                          rgb_mse=mse.mean(-1)[y[:, None], x[None]].ravel(),
                          background_rgb_mae=bg_mae[y[:, None], x[None]].ravel())
            for name, value in values.items():
                arrays.setdefault(name, []).append(value)
            for name, value in valid.items():
                flags.setdefault(name, []).append(value)
            for name, mask in selected.items():
                selections.setdefault(name, []).append(mask)
                stats = {f'{k}_{s}': v for k in ('rgb_mae', 'native_uv_magnitude', 'native_uv_gradient', 'image_pixel_gradient')
                         for s, v in statistics(values[k][None], mask[None], [frame]).items()}
                frame_rows.append(dict(frame=frame, local_frame=local, region=name, **stats))
            native.append(query['native'][0]); pixel.append(query['pixel'][0]); uvs.append(query['before'][0])
            frames.append(frame)
            print(f'  frame {frame}: {valid["valid"].sum()}/{len(grid)} valid background rays', flush=True)
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        selections = {k: np.asarray(v) for k, v in selections.items()}
        rows = table_rows(job, arrays, selections, frames)
        write_csv(folder / 'metrics.csv', rows)
        write_csv(folder / 'by_frame.csv', frame_rows)
        np.savez_compressed(folder / 'fields.npz', frames=frames, pixels=grid, native_uv=native,
                            image_flow_pixels=pixel, uv=uvs, **arrays,
                            **{'mask_'+k: v for k, v in selections.items()},
                            **{'flag_'+k: np.asarray(v) for k, v in flags.items()})
        plot_fields(folder, job, arrays, selections, frames, x, y)
        check_records(job['inputs'])
        check_code(manifest)
        write_json(folder / 'result.json', dict(complete=True, parity_passed=True, job=job['id'],
            frames=frames, grid_points=len(grid), sin_epoch=float(model.sin_epoch.item()),
            flow_weight=float(model.get_background_plane().flow_weight.item()), metrics=job['metrics'],
            environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(model.device))))


def plot_comparison(output, candidate, reference, ca, ra):
    """Common scales and signed differences, distinct from per-run figures."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder = output / 'comparison_images' / f'{candidate["id"]}__vs__{reference["id"]}'
    folder.mkdir(parents=True, exist_ok=True)
    shape = (len(np.unique(ca['pixels'][:, 1])), len(np.unique(ca['pixels'][:, 0])))
    frames = ca['frames']
    for index in np.unique(np.linspace(0, len(frames)-1, min(3, len(frames))).round().astype(int)):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for row, name in enumerate(('rgb_mae', 'image_pixel_gradient')):
            a, b = ca[name], ra[name]
            common = ca['mask_background'] & np.isfinite(a) & np.isfinite(b)
            if common.any():
                cap = max(float(np.quantile(np.concatenate([a[common], b[common]]), .95)), 1e-8)
                delta_cap = max(float(np.quantile(np.abs(a[common]-b[common]), .95)), 1e-8)
            else:
                cap = delta_cap = 1.
            for column, (values, title) in enumerate(((b, 'Reference'), (a, 'Candidate'), (a-b, 'Candidate minus reference'))):
                data = values[index].copy()
                data[~common[index]] = np.nan
                signed = column == 2
                cmap = plt.get_cmap('coolwarm' if signed else 'magma').copy()
                cmap.set_bad('#bdbdbd')
                im = axes[row, column].imshow(data.reshape(shape), cmap=cmap,
                    vmin=-delta_cap if signed else 0, vmax=delta_cap if signed else cap,
                    extent=[0, candidate['size'][0], candidate['size'][1], 0], interpolation='nearest')
                axes[row, column].set_title(title+'\n'+name)
                figure.colorbar(im, ax=axes[row, column], shrink=.75)
        figure.suptitle(f'{candidate["id"]}\nvs {reference["id"]} — frame {frames[index]}\n'
                       'Blue = candidate has a smaller value. Smaller flow gradient is NOT necessarily better quality. '
                       'White = zero difference; grey = excluded.')
        figure.savefig(folder / f'frame_{frames[index]:03d}.png', dpi=120)
        plt.close(figure)


def comparison_outputs(output, jobs):
    rows, comparisons = [], []
    finished = []
    for job in jobs:
        folder = output / job['id']
        if not completed_result(folder, job):
            continue
        with (folder / 'metrics.csv').open() as f:
            rows.extend(csv.DictReader(f))
        finished.append(job)
    for candidate in finished:
        if candidate['family'] == 'sphere':
            continue
        references = [j for j in finished if j['sequence'] == candidate['sequence'] and j['mode'] == candidate['mode']
                      and j['family'] in ('sphere', 'reprojection_v1') and j['family'] != candidate['family']]
        with np.load(output / candidate['id'] / 'fields.npz', allow_pickle=False) as ca:
            for reference in references:
                with np.load(output / reference['id'] / 'fields.npz', allow_pickle=False) as ra:
                    if not np.array_equal(ca['pixels'], ra['pixels']) or not np.array_equal(ca['frames'], ra['frames']):
                        raise ValueError('Cannot compare different frame/grid supports')
                    for region in ('background', 'boundary', 'foreground', 'all'):
                        if not np.array_equal(ca['mask_'+region], ra['mask_'+region]):
                            raise ValueError('Comparison masks differ')
                        support = ca['mask_'+region]
                        for name in ('rgb_mae', 'background_rgb_mae', 'native_uv_magnitude', 'native_uv_gradient',
                                     'native_uv_curvature', 'image_pixel_magnitude', 'image_pixel_gradient',
                                     'initial_texture_texel_magnitude', 'initial_texture_texel_gradient', 'transport_deviation'):
                            a, b = ca[name], ra[name]
                            common = support & np.isfinite(a) & np.isfinite(b)
                            diff = a-b
                            stats = statistics(diff, common, ca['frames'])
                            associations = [association(diff[i], ca['rgb_mae'][i]-ra['rgb_mae'][i], common[i]) for i in range(len(ca['frames']))]
                            valid = [v for v in associations if v is not None]
                            comparisons.append(dict(sequence=candidate['sequence'], mode=candidate['mode'],
                                candidate=candidate['id'], reference=reference['id'], region=region, measure=name,
                                **stats, sign='candidate minus reference',
                                mean_within_frame_spearman_with_error_difference=float(np.mean(valid)) if valid else None,
                                correlation_frames=len(valid),
                                delta_MPSNR=candidate['metrics']['MPSNR']-reference['metrics']['MPSNR'],
                                delta_SSIM=candidate['metrics']['SSIM']-reference['metrics']['SSIM'],
                                delta_LPIPS=candidate['metrics']['LPIPS']-reference['metrics']['LPIPS']))
                    plot_comparison(output, candidate, reference, ca, ra)
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'comparisons.csv', comparisons)
    links = ''.join(f'<li>{html.escape(j["id"])}: <a href="{j["id"]}/verification.csv">render verification</a>, '
                    f'<a href="{j["id"]}/metrics.csv">flow measurements</a>, '
                    f'<a href="{j["id"]}/by_frame.csv">per-frame results</a>'
                    + ''.join(f' <a href="{j["id"]}/{p.name}">frame figure</a>' for p in sorted((output/j['id']).glob('*_fields.png')))
                    + '</li>' for j in finished)
    paired_links = ''.join(f'<li>{html.escape(p.parent.name)}: <a href="{p.relative_to(output).as_posix()}">{p.name}</a></li>'
                          for p in sorted((output/'comparison_images').glob('*/*.png')))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Learned background flow</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;color:#151515;background:white}a{color:#004aad}</style>'
        '<h1>Learned background flow</h1><p>Only parity-passing final checkpoints are included. '
        'Measurements are actual learned backward texture lookup offsets, not optical flow or geometric correspondence requirements. '
        'Lower flow magnitude is not inherently better. Associations with error are descriptive, not causal.</p>'
        '<p><a href="comparisons.csv">Common-support comparisons</a> · <a href="metrics.csv">All measurements</a> · '
        '<a href="manifest.json">Inventory, checkpoints and settings</a></p><ul>'+links+'</ul>'
        '<h2>Matched spatial comparisons</h2><p>Shared color scales; blue differences mean candidate minus reference '
        'is negative. This favors the candidate for RGB error, but is not a quality verdict for flow gradients. '
        'There is no dead-zone threshold: white is zero; grey is excluded.</p><ul>'+paired_links+'</ul>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_flow_v1/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('/mnt/d/CGProjectData/waymo/diagnostics/learned_background_flow_v1'))
    parser.add_argument('--sequences', nargs='+', choices=SEQUENCES, default=DEFAULT_SEQUENCES)
    parser.add_argument('--families', nargs='+', choices=['sphere', 'reprojection_v1', 'legacy_floor', 'reprojection_new_radius_scale_v1', 'ar3', 'raw_v4', 'moderate_y'], default=DEFAULT_FAMILIES)
    parser.add_argument('--modes', nargs='+', choices=['regular', 'visuv'], default=['regular', 'visuv'])
    parser.add_argument('--floor-repeats', nargs='+', type=int, choices=[1, 2, 3], default=[1])
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--difference-step', type=float, default=1.)
    parser.add_argument('--boundary-width', type=int, default=3)
    parser.add_argument('--verify-frames', type=int, default=3)
    parser.add_argument('--parity-mean-bytes', type=float, default=.5)
    parser.add_argument('--parity-p99-bytes', type=float, default=2.)
    parser.add_argument('--max-frames', type=int, help='Optional evenly spaced subset for a smoke test; default is every frame')
    parser.add_argument('--limit-jobs', type=int, help='Optional smoke-test job limit')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('MANIFEST', 'JOB'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.worker:
        path, job = args.worker
        try:
            worker(Path(path), job)
        except Exception:
            folder = Path(path).parent / job
            folder.mkdir(exist_ok=True)
            write_json(folder / 'failure.json', dict(job=job, error=traceback.format_exc()))
            raise
        return
    if (args.stride < 4 or args.verify_frames < 1 or args.difference_step <= 0 or args.boundary_width < 1
            or args.parity_mean_bytes < 0 or args.parity_p99_bytes < 0
            or (args.max_frames is not None and args.max_frames < 1)
            or (args.limit_jobs is not None and args.limit_jobs < 1)):
        parser.error('Invalid nonpositive setting, or stride smaller than four pixels')
    jobs = jobs_from_manifest(args)
    settings = {k: getattr(args, k) for k in ('stride', 'difference_step', 'boundary_width', 'verify_frames',
                'parity_mean_bytes', 'parity_p99_bytes', 'max_frames')}
    settings['floor_repeat_policy'] = 'Explicit repeat numbers; no best-quality checkpoint selection or repeat-mean substitution'
    # Cache compatibility is local to this tool: do not touch parallax caches.
    code_paths = [Path(__file__), REPO/'nag/learned_flow_diagnostics.py', REPO/'nag/run/nag_runner.py',
                  REPO/'nag/config/nag_config.py', REPO/'nag/dataset/nag_dataset.py',
                  REPO/'nag/diagnostics/parallax_io.py', REPO/'nag/diagnostics/parallax_core.py']
    code = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            code_paths + sorted((REPO/'nag/model').glob('*.py'))}
    manifest = dict(schema=1, inventory=file_record(args.inventory, True), settings=settings, jobs=jobs, code=code,
                    policy='Read-only final checkpoints; parity before flow; fixed image grid; background mask interior is not a semantic/static label',
                    complete=False)
    print(f'{len(jobs)} checkpoint diagnostics (NO training):', flush=True)
    for job in jobs:
        print(f'  {job["id"]}: {len(job["indices"])} frames; {job["checkpoint"]}', flush=True)
    print(f'Output: {args.output}', flush=True)
    if args.dry_run:
        return
    args.output = args.output.resolve()
    if args.output.is_relative_to(REPO/'runs') or any(args.output.is_relative_to(Path(j['run'])) for j in jobs):
        raise ValueError('Output must not be inside an existing training run')
    manifest_path = args.output / 'manifest.json'
    if args.output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError('Output already exists; use --resume for identical inputs/settings or choose a new --output')
        old = json.loads(manifest_path.read_text())
        for key in ('schema', 'inventory', 'settings', 'jobs', 'code', 'policy'):
            if old.get(key) != manifest[key]:
                raise ValueError(f'Resume refused: {key} changed. Choose a new output folder.')
    else:
        args.output.mkdir(parents=True)
    write_json(manifest_path, manifest)
    failures = []
    for i, job in enumerate(jobs):
        folder = args.output / job['id']
        check_code(manifest)
        check_records(job['inputs'])
        if completed_result(folder, job):
            print(f'[{i+1}/{len(jobs)}] resume skip {job["id"]}', flush=True)
            continue
        folder.mkdir(exist_ok=True)
        print(f'[{i+1}/{len(jobs)}] {job["id"]} — log: {folder/"console.log"}', flush=True)
        with (folder/'console.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), '--worker', str(manifest_path), job['id']],
                                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(job['id'])
            print(f'  FAILED: {job["id"]}. See console.log / failure.json.', flush=True)
        else:
            print('  render parity passed; learned-flow measurements saved.', flush=True)
    comparison_outputs(args.output, jobs)
    manifest.update(complete=not failures, failures=failures)
    write_json(manifest_path, manifest)
    print(f'Finished: {len(jobs)-len(failures)}/{len(jobs)} passed. Open {args.output/"index.html"}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
