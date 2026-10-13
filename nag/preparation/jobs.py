"""Dataset descriptors shared by preparation, inspection and training launchers."""
from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from dotenv import dotenv_values
import numpy as np
from PIL import Image
import yaml

REPO = Path(__file__).resolve().parents[2]
SPHERE = 'nag.model.view_dependent_background_image_sphere_scene_node_3d.ViewDependentBackgroundImageSphereSceneNode3D'
PLANE = 'nag.model.view_dependent_background_image_plane_scene_node_3d.ViewDependentBackgroundImagePlaneSceneNode3D'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(path):
    return yaml.safe_load(Path(path).read_text())['NAGConfig']


def path_value(value, values, environment, base):
    text = str(value)
    for _ in range(8):
        new = re.sub(r'\{env:([^}]+)\}', lambda m: environment[m[1]], text)
        new = re.sub(r'\{([^}:]+)\}', lambda m: str(values[m[1]]), new)
        if new == text:
            return str((base / new).resolve())
        text = new
    raise ValueError(f'Unresolved dataset path: {value}')


@dataclass(frozen=True)
class PreparationJob:
    segment: str
    first: int
    stop: int
    step: int
    data_dir: str
    pose_path: str
    raw_pose_path: str
    image_size: tuple
    size: tuple
    regular_config: str
    visible_config: str
    source_configs: tuple

    @property
    def sequence(self):
        return 'segment-' + self.segment

    @property
    def frames(self):
        return tuple(range(self.first, self.stop, self.step))

    @property
    def uid(self):
        return f'{self.sequence}_frames_{self.first:03d}_{self.frames[-1]:03d}_step{self.step}'

    @property
    def label(self):
        return f'W{self.segment[:3]}_{self.first}_{self.frames[-1]}'

    def document(self):
        return {**asdict(self), 'uid': self.uid, 'frames': list(self.frames), 'label': self.label}


def normalized_training(c):
    c = copy.deepcopy(c)
    for key in ['name', 'background_plane_type', 'output_path', 'experiment_datetime',
                'background_ellipsoid_path', 'background_ellipsoid_scale_margin',
                'background_ellipsoid_texture_orientation', 'background_sphere_radius_from_visible_plane_centers']:
        c.pop(key, None)
    c.get('background_plane_kwargs', {}).pop('sphere_visible_uv_texture', None)
    c['save_intermediate_checkpoints'] = c.get('save_intermediate_checkpoints', False)
    return c


def discover(config_dir=None, config_paths=(), repo=REPO, data_root=None):
    paths = set(Path(p).resolve() for p in config_paths)
    if config_dir:
        paths.update(Path(config_dir).resolve().rglob('*.yaml'))
        paths.update(Path(config_dir).resolve().rglob('*.yml'))
    if not paths:
        raise ValueError('No YAML configurations selected')
    groups = {}
    for p in sorted(paths):
        c = settings(p)
        overrides=['background_sphere_radius_override','background_ellipsoid_path',
                   'background_ellipsoid_radii_override','background_ellipsoid_max_radius']
        if any(c.get(k) is not None for k in overrides) or c.get('background_sphere_radius_from_visible_plane_corners'):
            raise ValueError(f'Use a baseline YAML without geometry overrides: {p}')
        if c.get('normalize_camera') or c.get('position_spline_approximation') or c.get('bundle_path'):
            raise ValueError(f'Unsupported transformed/bundle camera configuration: {p}')
        env = {**dotenv_values(repo / (c.get('env_file') or '.env')), **os.environ}
        data = path_value(c['data_path'], c, env, repo)
        segment = re.fullmatch(r'segment-(\d+)', Path(data).name)
        if not segment:
            raise ValueError(f'Expected a complete Waymo segment ID: {data}')
        if data_root:
            data = str((Path(data_root) / segment[0]).resolve())
        values = {**c, 'data_path': data}
        for field, suffix in [('images_path','images_undistorted'),('masks_path','masks'),('depths_path','depth'),
                              ('boxes_path','boxes'),('boxes_object_id_mask_mapping_path','box_id_mapping.json'),
                              ('mask_ids_filter_path','mask_indices_filter.json')]:
            if c.get(field) and path_value(c[field],values,env,repo) != str((Path(data)/suffix).resolve()):
                raise ValueError(f'Unsupported nonstandard Waymo {field}: {p}')
        pose = path_value(c['pinhole_camera_config'], values, env, repo)
        sl = c['frame_indices_filter']
        first, stop, step = sl.get('start') or 0, sl.get('stop'), sl.get('step') or 1
        if not isinstance(stop, int) or first < 0 or step < 1 or len(range(first, stop, step)) < 2:
            raise ValueError(f'Requires an explicit interval with at least two frames: {p}')
        key = segment[1], first, stop, step
        row = groups.setdefault(key, [])
        row.append((p, c, data, pose))
    jobs = []
    for (segment, first, stop, step), rows in sorted(groups.items()):
        canonical = normalized_training(rows[0][1])
        if any(normalized_training(c) != canonical or (data, pose) != rows[0][2:]
               for _, c, data, pose in rows):
            raise ValueError(f'Conflicting calibration/preprocessing/training settings for segment-{segment} {first}:{stop}:{step}')
        regular = [r for r in rows if r[1]['background_plane_type'] == SPHERE and
                   not r[1].get('background_plane_kwargs', {}).get('sphere_visible_uv_texture', {}).get('enabled')]
        visible = [r for r in rows if r[1]['background_plane_type'] == SPHERE and
                   r[1].get('background_plane_kwargs', {}).get('sphere_visible_uv_texture', {}).get('enabled')]
        if len(regular) != 1 or len(visible) > 1:
            raise ValueError('Supply one canonical regular sphere YAML and at most one Visible-UV variant per interval')
        p, c, data, pose = regular[0]
        root = Path(data)
        images = {int(m[1]): f for f in (root / 'images_undistorted').iterdir()
                  if (m := re.fullmatch(r'(\d+)\.(png|jpg|jpeg)', f.name))}
        if first not in images:
            raise FileNotFoundError(f'Missing first image for {p}')
        with Image.open(images[first]) as im:
            image_size = im.size
        # This is the final renderer's convention: native camera resolution
        # times final_plot_resolution_factor, truncated to integer pixels.
        size = tuple((np.asarray(image_size) * c['final_plot_resolution_factor']).astype(int).tolist())
        if min(size) < 1:
            raise ValueError('Invalid final image dimensions')
        camera = json.loads(Path(pose).read_text())
        poses = np.asarray(camera['position'])
        if len(poses) < stop or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all() or np.any(camera.get('lens_distortion', [0]*5)):
            raise ValueError(f'Invalid/incomplete undistorted calibration: {pose}')
        jobs.append(PreparationJob(segment, first, stop, step, data, pose,
            str(root / 'waymo_pose_config.json'), tuple(image_size), size, str(p),
            str(visible[0][0]) if visible else '', tuple(str(r[0]) for r in rows)))
    return jobs


def select(jobs, intervals=()):
    if not intervals:
        return jobs
    selected = []
    for name in intervals:
        matches = [j for j in jobs if name == j.uid or name == j.label]
        if len(matches) != 1:
            raise ValueError(f'Unknown/ambiguous interval {name}; use full identity')
        selected.append(matches[0])
    if len({j.uid for j in selected}) != len(selected):
        raise ValueError('Duplicate interval selection')
    return selected


def audit(job, cloud_root):
    root = Path(job.data_dir)
    patterns = {'images_undistorted': r'(\d+)\.(png|jpg|jpeg)', 'depth': r'(\d+)\.tiff',
                'masks': r'img_(\d+)_ov_\d+\.png'}
    inputs, missing = [], {}
    frames = set(job.frames)
    for folder, pattern in patterns.items():
        available = (root / folder).iterdir() if (root / folder).is_dir() else []
        paths = [p for p in available if (m := re.fullmatch(pattern, p.name)) and int(m[1]) in frames]
        found = {int(re.fullmatch(pattern, p.name)[1]) for p in paths}
        if frames - found:
            missing[folder] = sorted(frames-found)
        inputs.extend(paths)
    for n in job.frames:
        p = Path(cloud_root) / job.sequence / f'{n:03d}.npz'
        if p.is_file():
            inputs.append(p)
        else:
            missing.setdefault('pointclouds', []).append(n)
    for name in [job.pose_path, job.raw_pose_path, root / 'box_id_mapping.json']:
        p = Path(name)
        if p.is_file():
            inputs.append(p)
        else:
            missing.setdefault('metadata', []).append(str(p))
    boxes = sorted((root / 'boxes').glob('*.json'))
    if not boxes:
        missing['boxes'] = ['No object-level box files']
    inputs.extend(boxes)
    optional = root / 'mask_indices_filter.json'
    if optional.exists():
        inputs.append(optional)
    snapshot = []
    for p in sorted(set(inputs)):
        st = p.stat()
        snapshot.append({'path': str(p.absolute()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
    centers = np.asarray(json.loads(Path(job.pose_path).read_text())['position'])[list(job.frames), :3, 3]
    maximum_baseline = float(np.linalg.norm(centers[:, None] - centers[None], axis=-1).max())
    return {'ready': not missing, 'missing': missing, 'inputs': snapshot,
            'maximum_camera_baseline_m': maximum_baseline, 'weak_parallax': maximum_baseline < .1,
            'weak_parallax_policy': 'diagnostic only; unchanged fitting weights and safety constraints'}
