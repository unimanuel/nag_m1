"""Content-validated stages with immutable attempts and explicit recipe signatures."""
import hashlib
import json
from pathlib import Path

from .jobs import REPO, sha


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def code_signature(*, validation_compatible=True):
    paths = list((REPO / 'nag/preparation').glob('*.py'))
    paths += list((REPO / 'nag/diagnostics').glob('*.py'))
    paths += [REPO / p for p in ['nag/reprojection_ellipsoid.py', 'nag/sampling_floor_ellipsoid.py',
        'nag/texture_sampling_diagnostics.py', 'nag/scripts/generalized_raw_v4.py',
        'nag/scripts/export_preparation_reference.py',
        'nag/scripts/generalized_ellipsoid_preparation.py',
        'libs/nag-waymo-processing/nag_waymo_processing/ellipsoid.py',
        'libs/nag-waymo-processing/nag_waymo_processing/point_cloud.py']]
    for folder in ['nag','tools/tools']:
        paths += list((REPO/folder).rglob('*.py'))
    snapshot = {str(p.relative_to(REPO)): sha(p) for p in sorted(set(paths)) if p.exists()}
    if validation_compatible:
        from .compatibility import reviewed_validation_snapshot
        return reviewed_validation_snapshot(snapshot)
    return snapshot


class Stage:
    def __init__(self, root, name, signature):
        self.root = Path(root) / name
        self.signature = json.loads(json.dumps(signature, sort_keys=True))
        self.manifest = self.root / 'stage.json'
        self.owner = self.root / 'owned_signature.json'
        self.result = None
        if self.manifest.exists():
            doc = json.loads(self.manifest.read_text())
            if doc['signature'] != self.signature:
                raise ValueError(f'Conflicting stage inputs: {self.root}; use a new autowork output root')
            for p, h in doc['files'].items():
                if sha(p) != h:
                    raise ValueError(f'Completed stage artifact changed: {p}')
            self.result = Path(doc['result'])
        elif self.root.exists() and any(self.root.iterdir()):
            if not self.owner.exists() or json.loads(self.owner.read_text()) != self.signature:
                raise ValueError(f'Unowned or conflicting partial stage: {self.root}')

    def attempt(self):
        self.root.mkdir(parents=True,exist_ok=True)
        if not self.owner.exists():write_json(self.owner,self.signature)
        attempts = list(self.root.glob('attempt_*'))
        n = max([int(p.name.split('_')[-1]) for p in attempts] or [0]) + 1
        path = self.root / f'attempt_{n:03d}'
        path.mkdir(parents=True)
        return path

    def complete(self, result, attempt):
        files = {str(p.resolve()): sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()}
        write_json(self.manifest, {'schema': 1, 'signature': self.signature, 'result': str(Path(result).resolve()), 'files': files})
        self.result = Path(result)
        return self.result


def recipe():
    return {'name': 'generalized_legacy_sampling_floor_v2', 'observations': {'rays_per_frame': 256,
        'targets_per_frame': 8, 'frame_limit': 0, 'visibility_radius': 2., 'depth_absolute': .5,
        'depth_relative': .03, 'alignment_p90_limit': 15., 'seed': 20260912},
        'raw_v4': {'max_samples': 24000, 'max_camera_normalized_radius': .8, 'minimum_camera_surface_distance': 2.},
        'reprojection': {'max_pairs': 24000, 'max_depth_rays': 14000, 'seed': 20260912,
            'reprojection_weight': 1., 'depth_weight': .15, 'reprojection_huber_px': 8., 'depth_huber_log': .2,
            'radius_excess_weight': 0., 'radius_excess_huber_log': .2, 'max_camera_normalized_radius': .65,
            'minimum_camera_surface_distance': 6., 'minimum_axis_radius': 8., 'max_axis_ratio': 10., 'max_iterations': 200},
        'floor_kernel_commit': '07cbd74',
        'floor': {'max_sampling': 4096, 'floor_threshold': .5, 'preserve_zero_floor': True,
                  'compactness_weight': 0., 'axis_multistart': False},
        'mapping': 'legacy regular', 'runtime_margin': 0., 'sphere_clipping': False}
