#!/usr/bin/env python3
"""Parity-gated dense learned background flow from one trusted final checkpoint.

Inference only. Historical runs are read-only; all outputs use a new autowork
directory. Restart with --resume to continue completed frame extractions.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'nag'), str(REPO/'nag/scripts')]
from diagnostics.parallax_core import regions
from diagnostics.parallax_io import diagnostic, load_mask, mask_paths, write_csv, write_json
from ego_learned_background_flow import (checkpoint_model, check_records, file_record,
    image, render_verification, saved_layers)
from learned_flow_diagnostics import association


def resolved(value):
    while isinstance(value, dict):
        value = value['value']
    return (REPO/str(value)).resolve()


def build_job(run):
    run = Path(run).resolve()
    config = run/'training_config.yaml'
    c = yaml.safe_load(config.read_text())['NAGConfig']
    if not (run/'exit_000.txt').read_text().startswith('Exit code: 0.'):
        raise ValueError('Source training did not exit successfully')
    sl = c['frame_indices_filter']
    frames = list(range(sl['start'], sl['stop'], sl.get('step') or 1))
    data = resolved(c['data_path'])
    layers = saved_layers(run, frames)
    with Image.open(layers[0]['composite']) as im:
        size = list(im.size)
    geo = run/'initial/background_diagnostics/ellipsoid_geometry.txt'
    if not geo.exists():
        geo = run/'initial/background_diagnostics/sphere_geometry.txt'
    g = diagnostic(geo)
    geometry = dict(center=g['center'], radii=g.get('effective_radii', [g.get('radius')]*3),
                    rotation=g.get('rotation_columns', np.eye(3).tolist()))
    checkpoints = list((run/'checkpoints').glob('final-*.ckpt'))
    if len(checkpoints) != 1 or checkpoints[0].name != 'final-80.ckpt':
        raise ValueError('Exactly one final-80.ckpt is required')
    sources = [file_record(config, True), file_record(geo, True), file_record(checkpoints[0], True)]
    metrics = {}
    for m in ('MPSNR', 'SSIM', 'LPIPS'):
        p = run/'tracker'/f'train_epoch_Mean{m}.csv'
        with p.open() as f:
            row = list(csv.DictReader(f, delimiter=';'))[-1]
        if float(row['step']) != 80 or not np.isfinite(float(row['value'])):
            raise ValueError('Source metrics are not finite final epoch-80 scalars')
        metrics[m] = float(row['value']); sources.append(file_record(p, True))
    for frame, layer in zip(frames, layers):
        sources.extend(file_record(p) for p in layer.values())
        sources.append(file_record(data/'images_undistorted'/f'{frame:03d}.png'))
        sources.extend(file_record(p) for p in mask_paths(data, frame))
    return dict(id=run.parent.name, run=str(run), checkpoint=str(checkpoints[0]),
                frames=frames, indices=list(range(len(frames))), size=size,
                data=str(data), layers=layers, geometry=geometry, metrics=metrics, inputs=sources)


def magnitude_mean(magnitudes, valid):
    """Mean of magnitudes on valid support, never norm of mean vectors."""
    keep = np.asarray(valid) & np.isfinite(magnitudes)
    count = keep.sum(axis=0)
    total = np.where(keep, magnitudes, 0).sum(axis=0, dtype=np.float64)
    return np.divide(total, count, out=np.full(count.shape, np.nan), where=count > 0), count


def dense_fields(pixel, native, effective, valid):
    """Image-grid magnitudes and central differences, units at saved resolution."""
    valid = valid & np.isfinite(pixel).all(-1) & np.isfinite(native).all(-1)
    gradient = np.full(valid.shape, np.nan, dtype=np.float32)
    stencil = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    stencil &= valid[2:, 1:-1] & valid[:-2, 1:-1]
    dx = (pixel[1:-1, 2:]-pixel[1:-1, :-2])/2
    dy = (pixel[2:, 1:-1]-pixel[:-2, 1:-1])/2
    gradient[1:-1, 1:-1] = np.where(stencil, np.sqrt((dx*dx+dy*dy).sum(-1)), np.nan)
    result = dict(image_pixel_magnitude=np.linalg.norm(pixel, axis=-1),
                  native_uv_magnitude=np.linalg.norm(native, axis=-1),
                  effective_uv_magnitude=np.linalg.norm(effective, axis=-1),
                  image_pixel_gradient=gradient)
    for a in result.values():
        a[~valid] = np.nan
    return result, valid


def query_dense(model, times, local, coordinates, factor):
    """Read the actual runtime modality call, including flow reference times."""
    import torch
    from nag.model.timed_discrete_scene_node_3d import global_to_local
    bg = model.get_background_plane()
    if not hasattr(bg, '_global_points_to_sphere_uv'):
        raise ValueError('This initial dense exporter supports sphere/ellipsoid checkpoints only')
    t, _ = model._get_t_with_flow_reference_times(times[local:local+1])
    selected = int(torch.nonzero(t == times[local], as_tuple=False).item())
    enabled = torch.zeros(len(model.objects)-int(model.config.has_background_plane)
                          -int(model.config.has_camera_aberration_plane), dtype=torch.bool, device=model.device)
    uv = torch.as_tensor(coordinates, dtype=times.dtype, device=model.device)
    original = bg.forward_modality
    captures = []

    def capture(points, *args, **kwargs):
        rgb, alpha, native = original(points, *args, **kwargs)
        before = bg._global_points_to_sphere_uv(points, kwargs['global_position'])
        after = bg._apply_uv_bounds(before-.5+native)+.5
        effective = after-before
        effective[..., 0] = torch.remainder(effective[..., 0]+.5, 1.)-.5
        moved = bg.get_flowed_global_points(points, native, kwargs['global_position'])
        positions, _, _, idx = model._assemble_object_rigid_data(t, enabled, True)
        camera = global_to_local(positions[idx], moved, v_include_time=True)[..., :3]
        valid = kwargs['is_inside'] & torch.isfinite(camera).all(-1) & (camera[..., 2] > 0)
        captures.append((native[:, selected].cpu().numpy(), effective[:, selected].cpu().numpy(),
                         valid[:, selected].cpu().numpy()))
        return rgb, alpha, native

    bg.forward_modality = capture
    try:
        _, _, flow = model.object_modality_forward(uv, t, enabled, enabled_background=True,
                                                   color=False, alpha=False, flow=True)
    finally:
        del bg.forward_modality
    if len(captures) != 1 or flow.shape[0] != 1:
        raise ValueError('Expected exactly one background flow query')
    native, effective, valid = captures[0]
    return flow[0, :, selected].cpu().numpy()*factor, native, effective, valid


def stats(values, mask):
    a = values[mask & np.isfinite(values)]
    return dict(n=int(a.size), mean=float(a.mean()) if a.size else None,
                median=float(np.median(a)) if a.size else None,
                p90=float(np.quantile(a, .9)) if a.size else None)


def extract(args, job, root, started):
    import torch
    frames_root = root/'frames'; frames_root.mkdir(exist_ok=True)
    pending = [i for i, f in enumerate(job['frames']) if not (frames_root/f'frame_{f:03d}.npz').is_file()]
    if not pending:
        return
    runner, config, model, times = checkpoint_model(job, root)
    render_verification(model, config, times, job, root,
                        dict(verify_frames=3, parity_mean_bytes=.5, parity_p99_bytes=2.))
    w, h = job['size']
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 2)
    to_camera = model.camera._image_resolution.flip(-1).cpu().numpy()/(np.array([w,h])-1)
    durations = []
    with torch.no_grad():
        for local in pending:
            if time.perf_counter()-started > args.max_seconds:
                raise TimeoutError('Time budget reached between frames; resume with the identical command and --resume')
            t0 = time.perf_counter()
            chunks = [query_dense(model, times, local, grid[start:start+args.chunk_pixels]*to_camera, 1/to_camera)
                      for start in range(0, len(grid), args.chunk_pixels)]
            pixel, native, effective, valid = [np.concatenate([c[k] for c in chunks]).reshape(
                (h,w,2) if k < 3 else (h,w)) for k in range(4)]
            fields, valid = dense_fields(pixel, native, effective, valid)
            frame = job['frames'][local]
            selections = regions(load_mask(Path(job['data']), frame, (w,h)), width=3)
            selections['all'] = np.ones((h,w), dtype=bool)
            target = image(Path(job['data'])/'images_undistorted'/f'{frame:03d}.png', (w,h))
            saved = image(job['layers'][local]['composite'])
            bg_saved = image(job['layers'][local]['background'])
            fields['rgb_mae'] = np.abs(saved.astype(np.float32)-target).mean(-1)/255
            fields['background_rgb_mae'] = np.abs(bg_saved.astype(np.float32)-target).mean(-1)/255
            # Atomic frame publication: a partial write is never reusable.
            destination = frames_root/f'frame_{frame:03d}.npz'
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, image_flow_pixels=pixel.astype(np.float32),
                    native_uv=native.astype(np.float32), valid=valid, **fields,
                    **{'mask_'+k:v for k,v in selections.items()})
            temporary.replace(destination)
            durations.append(time.perf_counter()-t0)
            write_json(root/'progress.json', dict(last_frame=frame, frame_seconds=durations[-1],
                elapsed_seconds=time.perf_counter()-started, pending_frames=len(pending)-len(durations),
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            print(f'frame {frame}: {durations[-1]:.1f}s; elapsed {(time.perf_counter()-started)/60:.1f}min; '
                  f'ETA {np.mean(durations)*(len(pending)-len(durations))/60:.1f}min', flush=True)
    del runner, model
    torch.cuda.empty_cache()


def heatmap(values, mask, cap, title):
    from matplotlib import colormaps
    good = mask & np.isfinite(values)
    rgb = (colormaps['magma'](np.clip(np.nan_to_num(values)/cap, 0, 1))[..., :3]*255).astype(np.uint8)
    rgb[~good] = [175,175,175]
    h,w = mask.shape
    out = Image.new('RGB', (w,h+48), 'white'); out.paste(Image.fromarray(rgb), (0,48))
    draw = ImageDraw.Draw(out)
    draw.text((8,4), title, fill='black')
    draw.text((8,20), f'Black=0; bright={cap:.5g} or more; grey=excluded. Display clipping only.', fill='black')
    bar = (colormaps['magma'](np.linspace(0,1,160))[None,:,:3]*255).astype(np.uint8)
    out.paste(Image.fromarray(np.repeat(bar, 8, axis=0)), (w-168,4))
    return out


def summarize(root, job):
    """CPU-only second pass with one common display scale for the whole run."""
    import cv2
    rows, samples = [], {}
    names = ('image_pixel_magnitude','native_uv_magnitude','effective_uv_magnitude',
             'image_pixel_gradient','rgb_mae','background_rgb_mae')
    h,w = job['size'][1], job['size'][0]
    totals = {k:np.zeros((h,w), dtype=np.float64) for k in names}
    counts = {k:np.zeros((h,w), dtype=np.uint16) for k in names}
    for frame in job['frames']:
        with np.load(root/'frames'/f'frame_{frame:03d}.npz', allow_pickle=False) as a:
            for name in names:
                keep = a['mask_background'] & a['valid'] & np.isfinite(a[name])
                totals[name] += np.where(keep, a[name], 0)
                counts[name] += keep
                samples.setdefault(name, []).append(a[name][keep][::97])
                for region in ('background','foreground','boundary','all'):
                    mask = a['mask_'+region] & a['valid']
                    rows.append(dict(frame=frame, region=region, measure=name, **stats(a[name],mask),
                        spearman_with_rgb_mae=association(a[name][::4,::4],a['rgb_mae'][::4,::4],mask[::4,::4])))
    caps = {k:max(float(np.quantile(np.concatenate(v), .95)),1e-8) for k,v in samples.items()}
    mean_fields = {k:np.divide(totals[k], counts[k], out=np.full((h,w),np.nan),where=counts[k]>0) for k in names}
    np.savez_compressed(root/'temporal_average.npz', **mean_fields, **{k+'_count':v for k,v in counts.items()})
    write_csv(root/'by_frame.csv', rows)
    (root/'temporal_average').mkdir(exist_ok=True)
    for name in names:
        heatmap(mean_fields[name], counts[name]>0, caps[name], 'Temporal mean: '+name).save(root/'temporal_average'/f'{name}.png')
    # Count is an image-coordinate support map, not a correspondence track.
    heatmap(counts['image_pixel_magnitude'].astype(float), np.ones((h,w),bool), len(job['frames']),
            'Valid background frame count').save(root/'temporal_average/valid_frame_count.png')
    for folder in ('magnitude','gradient','panels'):
        (root/folder).mkdir(exist_ok=True)
    video_path = root/'background_flow.mp4'
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10., (w*2,(h+48)*2))
    if not video.isOpened():
        raise RuntimeError('MP4 encoder unavailable; PNG/NPZ outputs remain available')
    try:
        for local, frame in enumerate(job['frames']):
            with np.load(root/'frames'/f'frame_{frame:03d}.npz',allow_pickle=False) as a:
                mask = a['mask_background'] & a['valid']
                mag = heatmap(a['image_pixel_magnitude'], mask, caps['image_pixel_magnitude'], f'Frame {frame}: flow magnitude, pixels')
                grad = heatmap(a['image_pixel_gradient'], mask, caps['image_pixel_gradient'], f'Frame {frame}: flow spatial gradient, pixels/pixel')
                err = heatmap(a['rgb_mae'], mask, caps['rgb_mae'], f'Frame {frame}: composite RGB MAE [0,1]')
                rgb = Image.new('RGB',(w,h+48),'white');rgb.paste(Image.fromarray(image(job['layers'][local]['composite'])),(0,48))
                ImageDraw.Draw(rgb).text((8,8),f'Saved reconstruction, frame {frame}',fill='black')
                panel=Image.new('RGB',(w*2,(h+48)*2),'white')
                for im,xy in ((rgb,(0,0)),(mag,(w,0)),(err,(0,h+48)),(grad,(w,h+48))):panel.paste(im,xy)
                mag.save(root/'magnitude'/f'frame_{frame:03d}.png');grad.save(root/'gradient'/f'frame_{frame:03d}.png')
                panel.save(root/'panels'/f'frame_{frame:03d}.png')
                video.write(cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
    finally:
        video.release()
    if video_path.stat().st_size < 1000:
        raise RuntimeError('Video encoder produced an empty file')
    summary=[]
    for name in names:
        selected=[r for r in rows if r['region']=='background' and r['measure']==name and r['n']]
        summary.append(dict(measure=name,frames=len(selected),
            mean_of_frame_means=float(np.mean([r['mean'] for r in selected])),
            mean_of_frame_medians=float(np.mean([r['median'] for r in selected])),
            mean_of_frame_p90=float(np.mean([r['p90'] for r in selected])),
            mean_within_frame_spearman=float(np.mean([r['spearman_with_rgb_mae'] for r in selected
                if r['spearman_with_rgb_mae'] is not None])), display_cap=caps[name]))
    write_csv(root/'summary.csv',summary)
    write_json(root/'summary.json',dict(metrics=summary,source_metrics=job['metrics'],
        interpretation='Actual learned backward texture-lookup corrections, not inter-frame optical flow. '
            'Temporal means average magnitudes in image coordinates on valid background support. '
            'Magnitude is not a causal measure of reconstruction difficulty; colour clipping is display-only. '
            'Correlations use every fourth row/column and are descriptive, not significance tests.'))
    links=''.join(f'<li><a href="temporal_average/{name}.png">{name}</a></li>' for name in names)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense background flow</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;color:#111;background:white}img,video{max-width:100%}</style>'
        '<h1>Dense learned background flow</h1><p>'+html.escape(job['run'])+'</p>'
        '<p>Final-checkpoint inference; no training. Backward texture corrections, not optical flow. '
        'Shared scales within this run; grey is excluded, not zero. Temporal averages use image locations, not tracked scene points.</p>'
        '<video controls src="background_flow.mp4"></video><ul>'+links+
        '<li><a href="temporal_average/valid_frame_count.png">Valid frame counts</a></li>'
        '<li><a href="summary.csv">Summary</a></li><li><a href="by_frame.csv">All frame/region statistics</a></li>'
        '<li><a href="verification.csv">Checkpoint render parity</a></li></ul>'
        '<h2>Temporal average magnitude</h2><img src="temporal_average/image_pixel_magnitude.png">'
        '<h2>First-frame panel</h2><img src="panels/frame_'+f'{job["frames"][0]:03d}'+'.png">',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-pixels',type=int,default=32768)
    p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); started=time.perf_counter()
    if args.chunk_pixels<1 or args.max_seconds<=0:p.error('Positive chunk size and budget required')
    root=args.output.resolve();run=args.run.resolve()
    if not root.name.startswith('autowork_') or run==root or run in root.parents:
        p.error('Choose a separate autowork_ output root, never a source-run subfolder')
    job=build_job(run)
    code_paths=[Path(__file__),REPO/'nag/learned_flow_diagnostics.py',REPO/'nag/scripts/ego_learned_background_flow.py',
                REPO/'nag/model/nag_functional_model.py',REPO/'nag/run/nag_runner.py']
    manifest=dict(schema=1,job=job,chunk_pixels=args.chunk_pixels,
                  code={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in code_paths})
    if root.exists():
        if not args.resume or not (root/'manifest.json').is_file():p.error('Existing output requires matching --resume')
        if json.loads((root/'manifest.json').read_text())!=manifest:p.error('Inputs/code differ from existing export')
    else:
        root.mkdir(parents=True);write_json(root/'manifest.json',manifest)
    check_records(job['inputs'])
    try:
        extract(args,job,root,started)
    except TimeoutError as exc:
        print(str(exc),flush=True);raise SystemExit(2)
    summarize(root,job)
    check_records(job['inputs'])
    for path,digest in manifest['code'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Code changed during export')
    write_json(root/'result.json',dict(complete=True,frames=len(job['frames']),size=job['size'],
        elapsed_seconds=time.perf_counter()-started,source_run=job['run'],no_training=True))
    print('COMPLETE',root,'elapsed seconds',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
