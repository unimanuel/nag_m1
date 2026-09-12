"""Verified experiment inventory and provenance for offline parallax diagnostics."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .parallax_core import intrinsics

SEQUENCES = {
    'W191_0_40': ('1918764220984209654', 0, 41),
    'W191_40_128': ('1918764220984209654', 40, 129),
    'W191_128_198': ('1918764220984209654', 128, 199),
    'W254_0_37': ('2547899409721197155', 0, 38),
    'W254_37_81': ('2547899409721197155', 37, 82),
    'W254_86_137': ('2547899409721197155', 86, 138),
    'W254_139_195': ('2547899409721197155', 139, 196),
}
ROOTS = ['2026-08-19_ellipsoid_non_center', '2026-08-21_missing_ego',
         '2026-08-20_ellipsoid_center_clipped', '2026-09-02_depth_proxy_v4',
         '2026-09-06_seams_compare_test', '2026-09-12_regular_sphere']


def jsonable(x):
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()


def snapshot(paths):
    return [{'path': str(p.resolve()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(set(map(Path, paths)))]


def code_signature(repo):
    paths = list((repo/'nag/diagnostics').glob('*.py')) + [
        repo/'libs/nag-waymo-processing/nag_waymo_processing/ellipsoid.py',
        repo/'libs/nag-waymo-processing/nag_waymo_processing/point_cloud.py']
    return digest({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic(path):
    fields = {}
    for line in path.read_text().splitlines():
        if ': ' not in line:
            continue
        key, value = line.strip().split(': ', 1)
        try:
            fields[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            fields[key] = value
    return fields


def classify_run(path):
    name = path.name.replace('\r', '')
    if '_texture_' in name and name.endswith('_regular'):
        mapping = name.split('_texture_')[1].removesuffix('_regular')
        if mapping not in ('legacy', 'world_y', 'world_y_auto_seam'):
            return None
        return 'moderate_y', 'regular', 'seam_'+mapping
    uv = name.endswith('_visuv')
    base = name[:-6] if uv else name
    if base.endswith('_ellipsoid_reprojection_depth_new_radius_scale_v1'):
        return 'reprojection_new_radius_scale_v1', 'visuv' if uv else 'regular', 'benchmark'
    if base.endswith('_ellipsoid_reprojection_depth_v1'):
        return 'reprojection_v1', 'visuv' if uv else 'regular', 'benchmark'
    if base.endswith('_ellipsoid_ray_depth_proxy_yup_v4_moderate_y'):
        return 'moderate_y', 'visuv' if uv else 'regular', 'benchmark'
    if base.endswith('_ellipsoid_ray_depth_proxy_yup_v4'):
        return 'raw_v4', 'visuv' if uv else 'regular', 'benchmark'
    if base.endswith(('_ellipsoid_centered_ar3_sphere_clipped', '_ellipsoid_sphere_clipped_current')):
        return 'ar3', 'visuv' if uv else 'regular', 'benchmark'
    if name.endswith('_sph_visuv_radius_visible_plane_centers'):
        return 'sphere', 'visuv', 'benchmark'
    if name.endswith('_radius_visible_plane_centers'):
        return 'sphere', 'regular', 'benchmark'
    return None


def inventory(repo, labels, data_root, extra_roots=()):
    runs, missing = [], []
    roots = list(ROOTS)
    for root in extra_roots:
        root = str(root)
        if root not in roots:
            roots.append(root)
    for folder in roots:
        root = repo/'runs'/folder
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            match = re.search(r'W(?:191|254)_\d+_\d+(?=_)', path.name)
            kind = classify_run(path)
            if not match or match[0] not in labels or kind is None:
                continue
            label = match[0]
            cfgpath = path/'training_config.yaml'
            cfg = yaml.safe_load(cfgpath.read_text())['NAGConfig']
            camera = cfg['pinhole_camera_config']
            if cfg.get('normalize_camera') or cfg.get('position_spline_approximation'):
                raise ValueError(f'Unsupported transformed camera configuration: {path}')
            if np.any(camera.get('lens_distortion', [0]*5)):
                raise ValueError(f'Expected undistorted calibration: {path}')
            segment, start, stop = SEQUENCES[label]
            sl = cfg['frame_indices_filter']
            if (sl['start'], sl['stop'], sl.get('step') or 1) != (start, stop, 1):
                raise ValueError(f'Unexpected frame slice: {path}')
            renders = sorted((path/'final/complete').glob('*_t.png'), key=lambda p:int(p.name.split('_')[0]))
            if [int(p.name.split('_')[0]) for p in renders] != list(range(stop-start)):
                raise ValueError(f'Missing/extra final frames: {path}')
            with Image.open(renders[0]) as im:
                size = list(im.size)
            gp = path/'initial/background_diagnostics'/('sphere_geometry.txt' if kind[0]=='sphere' else 'ellipsoid_geometry.txt')
            geom = diagnostic(gp)
            radii = [geom['radius']]*3 if kind[0]=='sphere' else geom['effective_radii']
            geometry = {'center': geom['center'], 'radii': radii,
                        'rotation': np.eye(3).tolist() if kind[0]=='sphere' else geom['rotation_columns'],
                        'provenance': str(gp),
                        'precision': 'sphere diagnostic rounded to 0.0001 m' if kind[0]=='sphere' else 'saved effective float32 geometry'}
            rot = np.asarray(geometry['rotation'])
            if (not np.isfinite(radii).all() or np.min(radii)<=0 or
                    not np.isfinite(geometry['center']).all() or
                    not np.allclose(rot.T@rot,np.eye(3),atol=1e-5) or np.linalg.det(rot)<0):
                raise ValueError(f'Invalid saved geometry: {gp}')
            tex = geom.get('texture_orientation', {'mode':'legacy'})
            mapping = {'mode': tex['mode'], 'rotation': tex.get('normalized_local_to_texture_columns', np.eye(3).tolist()),
                       'u_offset': tex.get('u_offset',0), 'y_flip': cfg.get('background_plane_kwargs',{}).get('sphere_uv_y_flip',False),
                       'visible_uv': geom['visible_uv_texture'], 'resolution_hw': geom['color_mask_resolution']}
            if mapping['visible_uv']['enabled'] != (kind[1]=='visuv'):
                raise ValueError(f'Run name and saved Visible-UV layout disagree: {path}')
            poses = np.asarray(camera['position'],dtype=float)[start:stop]
            pose_doc = data_root/f'segment-{segment}'/'waymo_pose_config_no_distortion.json'
            original = json.loads(pose_doc.read_text())
            if not np.allclose(poses,np.asarray(original['position'])[start:stop],atol=1e-6,rtol=0):
                raise ValueError(f'Run initial camera poses differ from dataset: {path}')
            if not np.allclose(intrinsics(camera,size),intrinsics(original,size),atol=1e-6):
                raise ValueError(f'Run intrinsics differ from dataset: {path}')
            metrics, sources = {}, [cfgpath,gp]
            for m in ('MPSNR','SSIM','LPIPS'):
                file=path/'tracker'/f'train_epoch_Mean{m}.csv'
                values=list(csv.DictReader(file.open(), delimiter=';'))
                metrics[m] = float(values[-1]['value'])
                sources.append(file)
            run_id = f'{kind[0]}_{kind[1]}_{kind[2]}'
            if any(r['sequence']==label and r['id']==run_id for r in runs):
                raise ValueError(f'Ambiguous run selection for {label} {run_id}')
            runs.append({'sequence':label, 'id':run_id, 'path':str(path.resolve()), 'family':kind[0], 'mode':kind[1],
                         'cohort':kind[2], 'geometry':geometry, 'mapping':mapping, 'metrics':metrics,
                         'geometry_id':digest({k:geometry[k] for k in ('center','radii','rotation')}),
                         'size':size, 'frames':list(range(start,stop)), 'initial_camera_learnable':
                         bool(cfg.get('is_camera_rotation_learnable') or cfg.get('is_camera_translation_learnable')),
                         'sources':snapshot(sources)})
    for label in labels:
        selected=[r for r in runs if r['sequence']==label]
        if not selected:
            raise ValueError(f'No runs for {label}')
        if len({tuple(r['size']) for r in selected})!=1:
            raise ValueError(f'Mixed render sizes in {label}; choose a common calibration explicitly')
        for family in ('sphere','ar3','raw_v4','moderate_y'):
            for mode in ('regular','visuv'):
                if not any(r['family']==family and r['mode']==mode and r['cohort']=='benchmark' for r in selected):
                    missing.append({'sequence':label,'family':family,'mode':mode})
    return {'runs':runs, 'missing':missing, 'camera_policy':'shared original calibrated poses; learned camera changes are not applied'}


@lru_cache(maxsize=8)
def _mask_index(data_dir):
    index={}
    for path in (data_dir/'masks').iterdir():
        match=re.fullmatch(r'img_(\d+)_ov_\d+\.png',path.name)
        if match:
            index.setdefault(int(match[1]),[]).append(path)
    return index


def mask_paths(data_dir, frame):
    paths=sorted(_mask_index(data_dir).get(int(frame),[]))
    if not paths:
        raise FileNotFoundError(f'No mask layers for frame {frame}: {data_dir}')
    return paths


def load_mask(data_dir, frame, size):
    union=np.zeros((size[1],size[0]), dtype=bool)
    for p in mask_paths(data_dir,frame):
        with Image.open(p) as image:
            image=image.resize(size,Image.Resampling.NEAREST)
            a=np.asarray(image)
            if a.ndim==3:
                a=a[...,0]
            union |= a!=0
    return union


def input_files(data_dir, cloud_dir, frames):
    files=[data_dir/'waymo_pose_config_no_distortion.json',data_dir/'waymo_pose_config.json',
           data_dir/'masks',data_dir/'images_undistorted',cloud_dir]
    for frame in frames:
        files += [cloud_dir/f'{frame:03d}.npz', data_dir/'images_undistorted'/f'{frame:03d}.png']
        files += mask_paths(data_dir,frame)
    return files
