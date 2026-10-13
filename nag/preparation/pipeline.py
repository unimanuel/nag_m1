"""Unchanged raw-v4 -> reprojection -> generic v2-floor geometry stages."""
import importlib.metadata
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .jobs import REPO, audit, sha
from .storage import Stage, code_signature, recipe, write_json
from diagnostics.parallax_io import jsonable
from diagnostics.parallax_observations import prepare_sequence
from reprojection_ellipsoid import ReprojectionFitConfig, fit_reprojection_depth_ellipsoid
from .legacy_floor import fit_sampling_floor
from fit_reprojection_ellipsoid import (_source_geometry, _copy_inspection_assets, _document,
                                      _write_diagnostics, _write_mesh)


RAW_PYTHON = REPO / 'libs/nag-waymo-processing/.venv/bin/python'


@lru_cache(maxsize=1)
def raw_environment():
    if not RAW_PYTHON.is_file():raise FileNotFoundError(f'Missing original raw-v4 interpreter: {RAW_PYTHON}')
    script="import sys,json,importlib.metadata as m; print(json.dumps({'executable':sys.executable,'python':sys.version,'versions':{k:m.version(k) for k in ['numpy','scipy']}}))"
    return json.loads(subprocess.check_output([str(RAW_PYTHON),'-c',script],text=True))


def signature(job, cloud_root):
    state = audit(job, cloud_root)
    if not state['ready']:
        raise ValueError(f'{job.uid}: missing inputs {state["missing"]}')
    return {'job': job.document(), 'inputs': state['inputs'], 'recipe': recipe(),
            'code': code_signature(), 'configs': {p: sha(p) for p in job.source_configs},
            'native_executable': sys.executable,
            'native_versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'torch', 'lightning', 'opencv-python']},
            'raw_executable': str(RAW_PYTHON), 'raw_environment':raw_environment(),
            'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES']},
            'motion': {k:state[k] for k in ['maximum_camera_baseline_m','weak_parallax']}}


def reference_stage(job, root, sig, sphere_run=None, frozen=None):
    source = {'completed_run': str(sphere_run)} if sphere_run else {'bootstrap': True}
    if sphere_run:
        source.update(config=sha(Path(sphere_run)/'training_config.yaml'),
                      diagnostic=sha(Path(sphere_run)/'initial/background_diagnostics/sphere_geometry.txt'))
    if frozen:
        source = {'frozen_fixture': frozen}
    stage = Stage(root, 'reference', {**sig, 'reference_source': source})
    if stage.result:
        return stage.result
    out = stage.attempt()
    target = out / 'sphere_reference.json'
    if frozen:
        document = {'schema': 1, 'job': job.document(), 'geometry': frozen['sphere'],
                    'mapping': frozen['mapping'], 'source': 'immutable_ego_verification_fixture',
                    'precision': 'historical diagnostic 0.0001 m'}
        write_json(target, document)
    elif sphere_run:
        from .reference import completed_reference
        write_json(target, completed_reference(job, sphere_run, out))
    else:
        job_path = out / 'job.json'
        write_json(job_path, job.document())
        subprocess.run([sys.executable, str(REPO/'nag/scripts/export_preparation_reference.py'),
                        '--job-json', str(job_path), '--output', str(out)], cwd=REPO, check=True)
    return stage.complete(target, out)


def observations_stage(job, root, sig, cloud_root):
    stage = Stage(root, 'observations', sig)
    if stage.result:
        return stage.result
    out = stage.attempt()
    args = SimpleNamespace(repo=REPO, data_root=Path(job.data_dir).parent, cloud_root=Path(cloud_root),
                           output=out, force=False, **recipe()['observations'])
    inv = {'runs': [{'sequence': job.uid, 'size': job.size}],
           'camera_policy': 'initial undistorted NAG dataset camera poses; no learned offsets'}
    prepare_sequence(args, inv, job.uid, job=job)
    return stage.complete(out/job.uid/'prepared.json', out)


def raw_stage(job, root, sig, cloud_root, frozen=None):
    stage = Stage(root, 'raw_v4', {**sig, 'frozen_initializer': str(frozen) if frozen else None,
                                 'frozen_sha256': sha(frozen) if frozen else None})
    if stage.result:
        return stage.result
    if frozen:
        # Verification-only input shortcut; never expose it in static prepare.
        return Path(frozen)
    if not RAW_PYTHON.is_file():
        raise FileNotFoundError(f'Missing original raw-v4 interpreter: {RAW_PYTHON}')
    out = stage.attempt()
    job_path = out/'job.json'
    write_json(job_path, job.document())
    subprocess.run([str(RAW_PYTHON), str(REPO/'nag/scripts/generalized_raw_v4.py'), '--job-json', str(job_path),
                    '--cloud-root', str(cloud_root), '--output', str(out)], cwd=REPO, check=True)
    meta = json.loads((out/'raw_environment.json').read_text())
    return stage.complete(meta['result'], out)


def fit_stage(job, root, sig, raw_path, prepared_path, reference_path, floor=False):
    name = 'sampling_floor_v2' if floor else 'reprojection_v1'
    stage = Stage(root, name, {**sig, 'source': sha(raw_path), 'prepared': sha(prepared_path),
                              'pairs': sha(Path(json.loads(Path(prepared_path).read_text())['directory'])/'pairs.npz'),
                              'reference': sha(reference_path)})
    if stage.result:
        return stage.result
    out = stage.attempt()
    source = json.loads(Path(raw_path).read_text())
    prepared = json.loads(Path(prepared_path).read_text())
    if prepared['scope'] != 'full_interval' or prepared['frames'] != list(job.frames) or prepared['size'] != list(job.size):
        raise ValueError('Prepared observations differ from requested job')
    reference = json.loads(Path(reference_path).read_text())
    with np.load(Path(prepared['directory'])/'pairs.npz', allow_pickle=False) as archive:
        pairs = dict(archive)
    cfg = ReprojectionFitConfig(**recipe()['reprojection'])
    if floor:
        fit = fit_sampling_floor(pairs, _source_geometry(source), reference['geometry'],
                                 reference['mapping'], job.size, cfg, **recipe()['floor'])
    else:
        fit = fit_reprojection_depth_ellipsoid(pairs, _source_geometry(source), cfg)
    _copy_inspection_assets(Path(raw_path), out)
    _write_mesh(out/'ellipsoid.ply', fit['geometry'])
    _write_diagnostics(out, fit, job.uid, Path(raw_path), Path(prepared_path))
    doc = _document(source, fit, Path(raw_path), Path(prepared_path), job.uid)
    if floor:
        doc['fit']['method'] = 'Generic legacy regular-mapping v2 sampling floor with exact-zero preservation'
        doc['fit']['sampling_floor'] = fit['sampling_floor']
        doc['files']['sampling_floor_diagnostics_json'] = 'sampling_floor_diagnostics.json'
        write_json(out/'sampling_floor_diagnostics.json', jsonable(fit['sampling_floor']))
    doc['generalized_preparation'] = {'job': job.document(), 'recipe': recipe(), 'reference': str(reference_path),
                                     'motion': sig['motion'], 'code': sig['code']}
    write_json(out/'ellipsoid.json', jsonable(doc))
    write_json(out/'sphere_reference.json', reference)
    # Export precisely the geometry used at runtime (zero margin, no cap).
    from export_background_geometry_meshes import mesh_surface, write_mesh_ply, SPHERE_RED, ELLIPSOID_BLUE
    for filename, g, color in [('sphere_red.ply', reference['geometry'], SPHERE_RED),
                                ('ellipsoid.ply', fit['geometry'], ELLIPSOID_BLUE)]:
        vertices, faces = mesh_surface(np.asarray(g['center']), np.asarray(g['radii']), np.asarray(g['rotation']), 48, 96)
        write_mesh_ply(out/filename, vertices, faces, color)
    return stage.complete(out/'ellipsoid.json', out)


def prepare(job, output_root, cloud_root, sphere_run=None, fixture=None, end_to_end=False):
    root = Path(output_root)/job.uid
    sig = signature(job, cloud_root)
    print(f'[prepare] {job.label}; weak parallax={sig["motion"]["weak_parallax"]}', flush=True)
    reference = reference_stage(job, root, sig, sphere_run, fixture if not end_to_end else None)
    prepared = observations_stage(job, root, sig, cloud_root)
    raw = raw_stage(job, root, sig, cloud_root, fixture['raw_v4'] if fixture and not end_to_end else None)
    reprojection = fit_stage(job, root, sig, raw, prepared, reference)
    final = fit_stage(job, root, sig, reprojection, prepared, reference, floor=True)
    result = {'job': job.document(), 'prepared': str(prepared), 'raw_v4': str(raw),
              'sphere_reference': str(reference), 'reprojection': str(reprojection), 'floor': str(final),
              'signature': sig}
    write_json(root/'result.json', result)
    print(f'[inspect] {final.parent}/ellipsoid.ply', flush=True)
    return result
