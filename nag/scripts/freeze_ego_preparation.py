"""Freeze immutable historical ego inputs before generalizing preparation."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_io import SEQUENCES, inventory, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    output = Path(output)
    if output.exists():
        doc = json.loads(output.read_text())
        for path, value in doc['files'].items():
            if sha(path) != value:
                raise ValueError(f'Frozen historical input changed: {path}')
        return doc
    inv = inventory(REPO, list(SEQUENCES), REPO / 'data/datasets/waymo', ['2026-09-12_reprojection_depth_v1'])
    rows, files = [], {}
    for label, (segment, first, stop) in SEQUENCES.items():
        root = Path('/mnt/d/CGProjectData/waymo/ellipsoids') / f'segment-{segment}'
        prefix = f'frames_{first:03d}_{stop-1:03d}'
        version = 2 if label in {'W191_40_128', 'W191_128_198', 'W254_86_137'} else 1
        pointer = Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1') / label / 'prepared.json'
        prepared = json.loads(pointer.read_text())
        source = root / (prefix + '_ray_depth_proxy_yup_v4') / 'ellipsoid.json'
        reprojection = root / (prefix + '_reprojection_depth_v1') / 'ellipsoid.json'
        floor = root / (prefix + f'_reprojection_depth_sampling_floor_v{version}') / 'ellipsoid.json'
        sphere = next(r for r in inv['runs'] if r['sequence'] == label and r['family'] == 'sphere'
                      and r['mode'] == 'regular' and r['cohort'] == 'benchmark')
        config = REPO / 'config/waymo/large-ego' / f'segment-{segment}_{first}_{stop-1}_large-undistorted.yaml'
        row = {'label': label, 'segment': segment, 'first': first, 'stop': stop, 'config': str(config),
               'prepared': str(pointer), 'pairs': str(Path(prepared['directory']) / 'pairs.npz'),
               'raw_v4': str(source), 'reprojection': str(reprojection), 'floor': str(floor),
               'floor_version': version, 'sphere_run': sphere['path'], 'sphere': sphere['geometry'],
               'mapping': sphere['mapping'], 'size': sphere['size'], 'input_snapshot': prepared['inputs']}
        for p in [config, pointer, Path(row['pairs']), source, reprojection, floor,
                  floor.parent / 'sampling_floor_diagnostics.json',
                  Path(sphere['path']) / 'training_config.yaml', Path(sphere['geometry']['provenance'])]:
            files[str(p)] = sha(p)
        # Content hashes for the decoded LiDAR frames; other original inputs
        # retain their recorded size/mtime snapshot, which is checked at replay.
        for item in prepared['inputs']:
            p = Path(item['path'])
            if p.suffix == '.npz':
                files[str(p)] = sha(p)
        rows.append(row)
        print(f'[freeze] {label}', flush=True)
    code = {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / 'nag/diagnostics').glob('*.py'))}
    for relative in ['nag/reprojection_ellipsoid.py', 'nag/sampling_floor_ellipsoid.py',
                     'libs/nag-waymo-processing/nag_waymo_processing/ellipsoid.py',
                     'libs/nag-waymo-processing/nag_waymo_processing/point_cloud.py']:
        code[relative] = sha(REPO / relative)
    doc = {'schema': 1, 'rows': rows, 'files': files, 'historical_code': code,
           'baseline_commit': '293b8f9', 'policy': 'read-only fixtures; never rewrite historical caches or fits'}
    write_json(output, doc)
    return doc


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        '/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_v1/ego_fixtures.json')
    freeze(target)
    print(f'Frozen fixture manifest: {target}')
