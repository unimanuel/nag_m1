"""YAML-driven Waymo geometry preparation, audits and isolated training commands."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'nag'))
sys.path.insert(1, str(REPO/'nag/scripts'))
from preparation.jobs import audit, discover, select, sha, settings
from preparation.storage import code_signature, recipe, write_json, Stage
from preparation.pipeline import prepare, reference_stage, signature, RAW_PYTHON
from preparation.verification import verify, release_gate

DEFAULT_VERIFICATION = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_release_v4')
DEFAULT_FIXTURES = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_v1/ego_fixtures.json')
DEFAULT_OUTPUT = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_generalized_legacy_v1')
CLOUD_ROOT = Path('/mnt/d/CGProjectData/waymo/pointclouds')
ELLIPSOID_TYPE = 'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D'

DATA_FIELDS = {'images_path':'images_undistorted','masks_path':'masks','depths_path':'depth',
               'boxes_path':'boxes','boxes_object_id_mask_mapping_path':'box_id_mapping.json',
               'mask_ids_filter_path':'mask_indices_filter.json'}


def reference_availability(job, output_root, sphere_runs):
    """Read-only inventory, not a replacement for full reuse validation."""
    if job.uid in sphere_runs:
        run=Path(sphere_runs[job.uid])
        present=all((run/p).is_file()for p in ['training_config.yaml','exit_000.txt',
                    'initial/background_diagnostics/sphere_geometry.txt'])
        return {'status':'supplied_candidate' if present else 'supplied_candidate_missing_files',
                'path':str(run),'reuse_validation':'required before reuse'}
    manifest=Path(output_root)/job.uid/'reference/stage.json'
    if manifest.exists():
        recorded=json.loads(manifest.read_text())
        return {'status':'prepared_candidate' if Path(recorded['result']).is_file() else 'prepared_candidate_missing_file',
                'path':recorded['result'],'reuse_validation':'input/recipe/artifact hashes checked before reuse'}
    return {'status':'native_initialization_required','reuse_validation':'no supplied or prepared reference'}


def cloud_commands(jobs, raw_root, cloud_root):
    lines = (REPO/'libs/nag-waymo-processing/data/waymo_train_list.txt').read_text().splitlines()
    cli = REPO/'libs/nag-waymo-processing/.venv/bin/nagwp'
    commands = []
    for segment in sorted({j.segment for j in jobs}):
        matches = [(n, line) for n,line in enumerate(lines) if line.split('_')[0] == 'segment-'+segment]
        if len(matches) != 1:
            raise ValueError(f'No unique exact training-list match for segment-{segment}')
        scene_id, name = matches[0]
        commands.append([str(cli), 'download', '--scene-ids', str(scene_id), '--target-dir', str(raw_root)])
        commands.append([str(cli), 'pointcloud', '--scene-ids', str(scene_id), '--source-dir', str(raw_root),
                         '--target-dir', str(cloud_root)])
    return commands


def training_jobs(jobs, preparation_root, cloud_root=CLOUD_ROOT):
    result = []
    for repeat in range(1,4):
        for job in jobs:
            root = Path(preparation_root)/job.uid
            record = json.loads((root/'result.json').read_text())
            current = signature(job, cloud_root)
            # Training does not execute the fitter. Retain its recorded thread/
            # device environment, while checking code, inputs and library versions.
            current['environment']=record['signature']['environment']
            if json.loads(json.dumps(current)) != record['signature']:
                raise ValueError('Prepared geometry inputs/recipe changed')
            for name in ['reference','observations','raw_v4','reprojection_v1','sampling_floor_v2']:
                manifest = json.loads((root/name/'stage.json').read_text())
                for key, value in json.loads(json.dumps(current)).items():
                    if manifest['signature'].get(key) != value:
                        raise ValueError('Preparation stage provenance mismatch')
                Stage(root, name, manifest['signature'])
            for mode in (['regular','visuv'] if repeat%2 else ['visuv','regular']):
                cfg = job.regular_config if mode=='regular' else job.visible_config
                if not cfg:
                    raise ValueError('Matched Visible-UV YAML is missing')
                c = settings(cfg)
                if (c.get('seed')!=42 or c.get('max_epochs')!=80 or c.get('deterministic',False)
                        or c.get('checkpoint_path') or c.get('save_intermediate_checkpoints',False)):
                    raise ValueError('Requires matched fresh epoch-80 seed-42 final-only training')
                result.append({'id': f'autowork_{job.uid}_ellipsoid_{mode}_seed42_repeat{repeat:02d}',
                    'label':job.label, 'uid':job.uid, 'mode':mode, 'repeat':repeat, 'seed':42,
                    'frame_count':len(job.frames), 'config':cfg, 'config_sha256':sha(cfg),
                    'data_dir':job.data_dir,'pose_path':job.pose_path,'size':list(job.size),
                    'geometry':record['floor'], 'geometry_sha256':sha(record['floor'])})
    return result


def training_command(job, out):
    command=[sys.executable,str(REPO/'nag/scripts/run_nag.py'),'--config-path',job['config'],
        '--background-plane-type',ELLIPSOID_TYPE,'--background-ellipsoid-path',job['geometry'],
        '--background-ellipsoid-scale-margin','0','--background-ellipsoid-texture-orientation','legacy',
        '--seed','42','--name',job['id'],'--output-path',str(out),
        '--data-path',job['data_dir'],'--pinhole-camera-config',job['pose_path']]
    for field,suffix in DATA_FIELDS.items():
        command.extend(['--'+field.replace('_','-'),str(Path(job['data_dir'])/suffix)])
    return command


def validate_completed_input_paths(c, job):
    """Match NAG's effective paths, including its absent optional filter."""
    def resolved(v):
        while isinstance(v,dict):v=v['value']
        return None if v is None else (REPO/v).resolve()
    if resolved(c.get('data_path'))!=Path(job['data_dir']).resolve():
        raise ValueError('Completed training used a different prepared dataset root')
    for field,suffix in DATA_FIELDS.items():
        expected=(Path(job['data_dir'])/suffix).resolve()
        if field=='mask_ids_filter_path' and not expected.exists():
            expected=None  # NAGConfig clears this optional path when absent.
        if resolved(c.get(field))!=expected:
            raise ValueError(f'Completed training input path mismatch: {field}')


def training_completed(path, job):
    from paper_plane_baselines import basic_completed
    result=basic_completed(path,job)
    if result is None:return None
    c=settings(path/'training_config.yaml')
    validate_completed_input_paths(c,job)
    import numpy as np
    from diagnostics.parallax_io import intrinsics
    calibration=json.loads(Path(job['pose_path']).read_text())
    if (not np.array_equal(c['pinhole_camera_config']['position'],calibration['position']) or
            not np.array_equal(intrinsics(c['pinhole_camera_config'],job['size']),intrinsics(calibration,job['size']))):
        raise ValueError('Completed training calibration mismatch')
    ref=copy.deepcopy(settings(job['config']))
    ref.update(name=job['id'],background_plane_type=ELLIPSOID_TYPE,
               background_ellipsoid_scale_margin=0.,background_ellipsoid_texture_orientation='legacy',
               object_rigid_control_points=job['frame_count'],used_runner_type='nag.run.nag_runner.NAGRunner')
    ref['plane_init_strategy_kwargs']['position_spline_fitting']=ref['object_position_spline_approximation']
    start=0
    for phase in ref['training_phases']:
        phase['start_at']=start;start+=phase['length']
    resolved={'data_path','images_path','masks_path','depths_path','boxes_path',
              'boxes_object_id_mask_mapping_path','mask_ids_filter_path','pinhole_camera_config',
              'output_path','experiment_datetime','num_objects','background_ellipsoid_path'}
    def normalize(v):
        if isinstance(v,dict):
            if v.get('__class__')=='type':return v['value']
            return {k:normalize(x)for k,x in v.items()}
        if isinstance(v,list):return [normalize(x)for x in v]
        return v
    for k,v in ref.items():
        if k not in resolved and normalize(c.get(k))!=v:raise ValueError(f'Unmatched completed training setting: {k}')
    geometry=c.get('background_ellipsoid_path')
    while isinstance(geometry,dict):geometry=geometry['value']
    if geometry is None or (REPO/geometry).resolve()!=Path(job['geometry']).resolve():
        raise ValueError('Completed run used another ellipsoid')
    if c.get('checkpoint_path') or c.get('save_intermediate_checkpoints',False):
        raise ValueError('Completed run is not fresh/final-only')
    if c.get('background_ellipsoid_max_radius') is not None or c.get('background_ellipsoid_radii_override') is not None:
        raise ValueError('Unexpected ellipsoid size override or sphere clipping')
    from diagnostics.parallax_io import diagnostic
    from fit_reprojection_ellipsoid import _source_geometry
    diag=diagnostic(path/'initial/background_diagnostics/ellipsoid_geometry.txt')
    wanted=_source_geometry(json.loads(Path(job['geometry']).read_text()))
    for key,field in [('center','center'),('radii','effective_radii'),('rotation','rotation_columns')]:
        if not np.allclose(wanted[key],diag[field],rtol=1e-6,atol=1e-4 if key!='rotation' else 1e-6):
            raise ValueError('Runtime ellipsoid differs from prepared geometry')
    return result


def train(jobs, args):
    from paper_sphere_baselines import duration_seconds, code_snapshot
    root = args.training_root
    planned = training_jobs(jobs,args.output_root,args.cloud_root)
    context = {'code':code_snapshot(), 'preparation_code':code_signature(), 'gate':sha(args.verification_root/'verification.json'),
               'versions': {p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','pytorch-lightning']},
               'executable':sys.executable, 'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    manifest = {'schema':1,'jobs':planned,'context':context}
    if root.exists() and any(root.iterdir()) and (not (root/'manifest.json').exists() or json.loads((root/'manifest.json').read_text())!=manifest):
        raise ValueError('Unowned/conflicting training output; use another autowork root')
    for job in planned:
        print(shlex.join(training_command(job,root/job['id']/'attempt_001')),flush=True)
    if args.dry_run:
        print(f'{len(planned)} planned trainings; no files or training created.')
        return
    root.mkdir(parents=True,exist_ok=True)
    lock_root=REPO/'runs/autowork_generalized_background_training_lock'
    lock_root.mkdir(parents=True,exist_ok=True)
    with (lock_root/'gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another training invocation is active') from None
        if not (root/'manifest.json').exists():write_json(root/'manifest.json',manifest)
        start = time.monotonic()
        for job in planned:
            attempts = sorted((root/job['id']).glob('attempt_*'))
            good = [p for p in attempts if training_completed(p,job) is not None]
            if len(good)>1:raise ValueError('Multiple completed attempts')
            if good:
                print('SKIP completed '+job['id'],flush=True);continue
            if time.monotonic()-start>=args.max_hours*3600:
                print('Invocation budget reached; rerun the same selection to continue.');break
            if context['code']!=code_snapshot() or context['preparation_code']!=code_signature():
                raise ValueError('Code changed during training batch')
            if sha(job['config'])!=job['config_sha256'] or sha(job['geometry'])!=job['geometry_sha256']:
                raise ValueError('Frozen training inputs changed')
            n=max([int(p.name.split('_')[-1]) for p in attempts]or[0])+1
            out=root/job['id']/f'attempt_{n:03d}'
            subprocess.run(training_command(job,out),cwd=REPO,check=True)
            if training_completed(out,job) is None:raise RuntimeError('Incomplete final outputs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','clouds','reference','prepare','verify','train'])
    group=p.add_mutually_exclusive_group()
    group.add_argument('--config-dir',type=Path)
    group.add_argument('--config-paths',nargs='+',type=Path)
    p.add_argument('--intervals',nargs='+')
    p.add_argument('--cloud-root',type=Path,default=CLOUD_ROOT)
    p.add_argument('--raw-root',type=Path,default=Path('/mnt/d/CGProjectData/waymo/raw'))
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--verification-root',type=Path,default=DEFAULT_VERIFICATION)
    p.add_argument('--fixtures',type=Path,default=DEFAULT_FIXTURES)
    p.add_argument('--verify-level',choices=['cached','end_to_end','all'],default='all')
    p.add_argument('--sphere-run-map',type=Path,help='JSON full-job-identity to completed regular-sphere run path')
    p.add_argument('--training-root',type=Path,default=REPO/'runs/autowork_final_ellipsoid_benchmarks')
    p.add_argument('--max-hours',type=float,default=9.)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    from preparation.compatibility import reviewed_validation_snapshot
    raw_code=code_signature(validation_compatible=False)
    if reviewed_validation_snapshot(raw_code)!=raw_code:
        print('Compatibility: reviewed optional-mask validation fix; exact hash-backed '
              'aliases preserve existing preparation certificates and run manifests. '
              'No fitting or training settings are changed.',flush=True)
    for folder in [args.output_root,args.verification_root,args.training_root]:
        if not folder.name.startswith('autowork_'):p.error('New output namespaces must begin with autowork_')
    if args.max_hours<=0:p.error('max-hours must be positive')
    if args.stage=='verify':
        if args.dry_run:
            from preparation.verification import verify_fixtures
            verify_fixtures(args.fixtures)
            print('Seven immutable ego fixtures validated; no files or fitting performed.')
        else:
            report=verify(args.fixtures,args.verification_root,args.cloud_root,args.verify_level,args.intervals or ())
            if any(not r.get('passed')for m in ['cached','end_to_end']for r in report[m].values()):
                raise ValueError('Compatibility comparisons failed; release stays blocked. See verification.json.')
        return
    jobs=select(discover(args.config_dir or (None if args.config_paths else REPO/'config/waymo/autowork_paper-final'),
                         args.config_paths or (),data_root=args.data_root),args.intervals or ())
    sphere_runs=json.loads(args.sphere_run_map.read_text()) if args.sphere_run_map else {}
    audits={j.uid:audit(j,args.cloud_root) for j in jobs}
    for j in jobs:
        a=audits[j.uid];print(f'{j.label} [{j.uid}]: ready={a["ready"]}; baseline={a["maximum_camera_baseline_m"]:.6f} m; '
                             f'weak_parallax={a["weak_parallax"]}; missing={a["missing"]}; '
                             f'reference={reference_availability(j,args.output_root,sphere_runs)}',flush=True)
    if args.stage=='audit':return
    if args.stage=='clouds':
        for command in cloud_commands(jobs,args.raw_root,args.cloud_root):
            print(shlex.join(command),flush=True)
            if not args.dry_run:subprocess.run(command,cwd=REPO,check=True)
        return
    if not all(a['ready'] for a in audits.values()):
        raise ValueError('Selected intervals have missing inputs; repair or explicitly select a ready subset. No frames skipped.')
    if args.stage in ['prepare','train']:
        release_gate(args.verification_root/'verification.json')
    if args.stage=='train':
        train(jobs,args);return
    if args.dry_run:
        print(f'{len(jobs)} geometry jobs; no files, reference initialization or fitting performed.');return
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'preparation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another preparation invocation is active') from None
        for j in jobs:
            if args.stage=='reference':reference_stage(j,args.output_root/j.uid,signature(j,args.cloud_root),sphere_runs.get(j.uid))
            else:prepare(j,args.output_root,args.cloud_root,sphere_runs.get(j.uid))


if __name__=='__main__':main()
