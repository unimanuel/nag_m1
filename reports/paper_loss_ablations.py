#!/usr/bin/env python3
"""Audit, prepare and serially train the five-interval matched loss ablation.

No production code changes. Audit/dry-run/summary never create files or train.
The main benchmark is reused by identity, never by selecting quality metrics.
"""
import argparse
import codecs
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import time

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'reports'),str(REPO/'nag'),str(REPO/'nag/scripts'),str(REPO)]
import generate_preliminary_background_statistics as stats
from generate_main_background_statistics import ABLATION_INTERVALS, safe_yaml
from paper_ablation_kernel import SOURCE, SOURCE_SHA256, fit_depth_only
from diagnostics.parallax_io import diagnostic, snapshot, write_json, jsonable
from preparation.storage import Stage, recipe
from preparation.verification import release_gate
from reprojection_ellipsoid import ReprojectionFitConfig, evaluate_geometry
from fit_reprojection_ellipsoid import _source_geometry, _document, _write_diagnostics, _copy_inspection_assets
from generalized_ellipsoid_preparation import training_command, validate_completed_input_paths
from paper_sphere_baselines import code_snapshot, duration_seconds
from paper_final_baselines import training_context

DEFAULT_CATALOG=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_main_background_statistics_2026-11-01/summary.json')
DEFAULT_PREP=Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_paper_loss_ablations_v1')
DEFAULT_TRAIN=REPO/'runs/autowork_paper_loss_ablations_v1'
FIXTURES=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_v1/ego_fixtures.json')
GATE=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_release_v4/verification.json')
CONSOLE_LEDGER=REPO/'reports/paper_loss_ablation_console_compatibility.json'
ORIGINAL_LAUNCHER_SHA='00eb49a36852378216773d070a9cc42540e8e29dc70c3c485b594b06e7b3f050'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def code_identity(path):
    """Alias only the exact reviewed console-only launcher revision.

    Historical manifests stay unchanged. Any further launcher edit disables
    the alias until separately reviewed; no fitting/model hash is waived.
    """
    actual=sha(path)
    if Path(path).resolve()!=Path(__file__).resolve() or not CONSOLE_LEDGER.is_file():return actual
    ledger=load(CONSOLE_LEDGER)
    if (ledger.get('scope')=='console_streaming_only_v1' and ledger.get('before')==ORIGINAL_LAUNCHER_SHA
            and actual==ledger.get('after')):
        return ORIGINAL_LAUNCHER_SHA
    return actual


def assert_pins(pins):
    for path,digest in pins.items():
        if code_identity(path)!=digest:raise ValueError('Frozen input changed: '+path)


def native_context():
    # Matches the approved benchmark interpreter, versions, environment and
    # training code. Never canonicalize or waive another historical revision.
    return training_context()


def same_runtime(a,b):
    return all(np.array_equal(np.asarray(a[k],dtype=np.float32),np.asarray(b[k],dtype=np.float32))
               for k in ('center','radii','rotation_columns'))


def run_config(path):return safe_yaml((Path(path)/'training_config.yaml').read_text())['NAGConfig']


def validate_run(path,item,geometry):
    """Completion plus matched settings, calibration and exact float32 geometry."""
    path=Path(path)
    if stats.complete_means(path,item['frame_count']) is None:return False
    c=run_config(path)
    if stats.normalized_config(c,'ellipsoid_regular',REPO)!=item['normalized_training']:
        raise ValueError('Unmatched ablation training settings: '+str(path))
    validate_completed_input_paths(c,item)
    calibration=load(item['pose_path'])
    from diagnostics.parallax_io import intrinsics
    if (not np.array_equal(c['pinhole_camera_config']['position'],calibration['position']) or
            not np.array_equal(intrinsics(c['pinhole_camera_config'],item['size']),intrinsics(calibration,item['size']))):
        raise ValueError('Unmatched calibration: '+str(path))
    d=diagnostic(path/'initial/background_diagnostics/ellipsoid_geometry.txt')
    actual=dict(center=d['center'],radii=d['effective_radii'],rotation_columns=d['rotation_columns'])
    if not same_runtime(actual,load(geometry)['fit']):raise ValueError('Runtime geometry mismatch: '+str(path))
    return True


def _pin_geometry_source(path,pins):
    p=Path(path);pins[str(p)]=sha(p)
    return load(p)


def original_config(config):
    # The immutable September-original documents predate these two optional
    # zero-regularizer fields. Do not default any missing core fitting setting.
    defaults=asdict(ReprojectionFitConfig())
    optional={k:defaults[k] for k in ('radius_excess_weight','radius_excess_huber_log')}
    result={**optional,**config}
    if result!=defaults:raise ValueError('D1 is not the frozen original objective')
    return result


def inventory(catalog):
    gate=release_gate(GATE)
    doc=load(catalog)
    if doc.get('status')!='MAIN_BENCHMARK_COMPLETE_ABLATIONS_PLANNED' or len(doc['runs'])!=465:
        raise ValueError('Use the audited complete main-benchmark catalog')
    fixtures=load(FIXTURES)
    fixture_rows={r['label']:r for r in fixtures['rows']}
    global_pins={str(catalog):sha(catalog),str(FIXTURES):sha(FIXTURES),str(GATE):sha(GATE),str(SOURCE):SOURCE_SHA256}
    assert_pins(global_pins)
    items=[]
    for label in ABLATION_INTERVALS:
        rows=sorted([r for r in doc['runs'] if r['label']==label and r['treatment']=='ellipsoid_regular'],key=lambda r:r['repeat'])
        if [r['repeat'] for r in rows]!=[1,2,3]:raise ValueError('Missing main D2 repeats: '+label)
        manifest=load(rows[0]['manifest'])
        sources=[j for j in manifest['jobs'] if j.get('label')==label and j.get('mode')=='regular' and j.get('repeat')==1
                 and j.get('orientation','legacy')=='legacy']
        if len(sources)!=1:raise ValueError('Ambiguous main geometry source: '+label)
        j=sources[0];pins={str(Path(rows[0]['manifest'])):sha(rows[0]['manifest'])}
        for field in ('config','geometry'):
            if sha(j[field])!=j[field+'_sha256']:raise ValueError('Main source changed: '+j[field])
            pins[j[field]]=j[field+'_sha256']
        floor=load(j['geometry']);d1_path=Path(floor['reprojection_fit']['source_geometry'])
        d1=_pin_geometry_source(d1_path,pins)
        original_config(d1['reprojection_fit']['config'])
        if d1['fit'].get('regularizer_version','original')!='original':raise ValueError('Wrong D1 regularizer')
        raw_path=Path(d1['reprojection_fit']['source_geometry']);raw=_pin_geometry_source(raw_path,pins)
        prepared=Path(d1['reprojection_fit']['prepared_observations']);meta=_pin_geometry_source(prepared,pins)
        pairs=Path(meta['directory'])/'pairs.npz';pins[str(pairs)]=sha(pairs)
        first,last=map(int,label.split('_')[1:]);frames=list(range(first,last+1))
        c=run_config(rows[0]['path']);data=stats.resolved_value(c['data_path'],REPO)
        pose=Path(data)/'waymo_pose_config_no_distortion.json';pins[str(pose)]=sha(pose)
        if (meta['frames']!=frames or meta['scope']!='full_interval' or meta['size']!=[960,640]
                or floor['sequence']!=Path(data).name):raise ValueError('Interval/cache identity mismatch')
        for g in (raw,d1,floor):
            if g['sequence']!=floor['sequence'] or (g['frame_slice']['start'],g['frame_slice']['stop'],g['frame_slice']['step'])!=(first,last+1,1):
                raise ValueError('Geometry frame identity mismatch')
        if meta['inputs']!=snapshot([r['path'] for r in meta['inputs']]):
            raise ValueError('Cached observation inputs changed; do not silently regenerate: '+label)
        # Explicit immutable fixture or released-stage evidence, not a broad
        # bypass of historical cache code signatures.
        if label in fixture_rows:
            for p in (prepared,pairs,raw_path,d1_path,Path(j['geometry'])):
                expected=fixtures['files'].get(str(p))
                if expected is None or sha(p)!=expected:raise ValueError('Ego fixture pin mismatch: '+str(p))
        else:
            root=Path(j['geometry']).parents[2]
            for name in ('observations','raw_v4','reprojection_v1','sampling_floor_v2','reference'):
                stage_path=root/name/'stage.json';stage=load(stage_path)
                if stage['signature']['code']!=gate['code'] or stage['signature']['recipe']!=recipe():
                    raise ValueError('Static preparation code/recipe differs from release gate')
                for p,h in stage['files'].items():
                    if sha(p)!=h:raise ValueError('Released artifact changed: '+p)
                pins[str(stage_path)]=sha(stage_path)
        item=dict(label=label,segment=floor['sequence'].removeprefix('segment-'),frame_count=len(frames),first=first,stop=last+1,
            config=j['config'],config_sha256=j['config_sha256'],data_dir=data,pose_path=str(pose),size=meta['size'],
            D1_geometry=str(d1_path),D2_geometry=j['geometry'],raw=str(raw_path),prepared=str(prepared),pairs=str(pairs),
            normalized_training=stats.normalized_config(c,'ellipsoid_regular',REPO),pins=pins,D2_runs=[r['path'] for r in rows],
            D1_same_as_D2=same_runtime(d1['fit'],floor['fit']),
            old_single_runs='Not reused: no frozen training-code/environment manifest or explicit repeat identity')
        times=[]
        for r in rows:
            if not validate_run(r['path'],item,j['geometry']):raise ValueError('Incomplete main D2 run')
            for name in ('training_config.yaml','exit_000.txt','initial/background_diagnostics/ellipsoid_geometry.txt'):
                p=Path(r['path'])/name;pins[str(p)]=sha(p)
            for m in stats.METRICS:
                for prefix in ('','Mean'):
                    p=Path(r['path'])/'tracker'/f'train_epoch_{prefix}{m}.csv';pins[str(p)]=sha(p)
            times.append(duration_seconds(Path(r['path'])/'exit_000.txt'))
        item['estimated_training_seconds']=max(times)*1.35
        items.append(item)
    return dict(schema=1,intervals=items,pins=global_pins,context=native_context(),
        diagnostic_code={str(p):code_identity(p) for p in (Path(__file__),REPO/'reports/paper_ablation_kernel.py',
            REPO/'reports/generate_preliminary_background_statistics.py',REPO/'reports/generate_main_background_statistics.py')},
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy')},executable=sys.executable,
        recipes={'D0':{**asdict(ReprojectionFitConfig()),'reprojection_weight':0.},'D1':asdict(ReprojectionFitConfig()),
                 'D2':'existing main paper geometry; no refit'},
        cache_policy='Exact ego fixture pins or released static stage pins plus unchanged observation input snapshots')


def matrix(plan,prep_root,require_prepared=False):
    cells=[]
    for item in plan['intervals']:
        for recipe_id in ('D0','D1','D2'):
            if recipe_id=='D0':
                pointer=Path(prep_root)/item['label']/'D0/stage.json'
                if pointer.exists():
                    stage=Stage(Path(prep_root)/item['label'],'D0',d0_signature(plan,item,prep_root))
                    geometry=str(stage.result)
                elif require_prepared:raise ValueError('Prepare D0 first: '+item['label'])
                else:geometry=str(Path(prep_root)/item['label']/'D0/attempt_001/ellipsoid.json')
            else:geometry=item[recipe_id+'_geometry']
            for repeat in (1,2,3):
                reused=item['D2_runs'][repeat-1] if recipe_id=='D2' or (recipe_id=='D1' and item['D1_same_as_D2']) else None
                cells.append(dict(id=f'autowork_{item["label"]}_{recipe_id}_regular_seed42_repeat{repeat:02d}',
                    label=item['label'],recipe=recipe_id,repeat=repeat,seed=42,mode='regular',geometry=geometry,
                    geometry_sha256=sha(geometry) if Path(geometry).is_file() else None,
                    reused_run=reused,reuse_reason=('D2 main result' if recipe_id=='D2' else 'Exact float32 D1/D2 runtime geometry equivalence') if reused else None))
    return cells


def d0_signature(plan,item,root):
    return dict(plan=sha(Path(root)/'manifest.json'),source=item['pins'],recipe=plan['recipes']['D0'])


def show(plan,cells):
    for item in plan['intervals']:
        take=[c for c in cells if c['label']==item['label']]
        new=sum(c['reused_run'] is None for c in take)
        print(f'{item["label"]}: {new} new, {9-new} reused; buffered estimate {new*item["estimated_training_seconds"]/3600:.2f} h',flush=True)
    print(f'TOTAL {sum(c["reused_run"] is None for c in cells)} new / {len(cells)} cells; '
          'regular only; three seed-42 repeats; no training launched by audit.',flush=True)


def freeze(root,manifest,dry_run=False):
    root=Path(root);path=root/'manifest.json'
    if path.exists():
        if load(path)!=manifest:raise ValueError('Output manifest conflicts; preserve it and use another autowork root: '+str(root))
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Unowned nonempty output: '+str(root))
    elif not dry_run:
        root.mkdir(parents=True,exist_ok=True);write_json(path,manifest)


def check_plan(plan):
    assert_pins(plan['pins']);assert_pins(plan['diagnostic_code'])
    if native_context()!=plan['context']:raise ValueError('Training code/environment changed')
    for item in plan['intervals']:
        assert_pins(item['pins'])
        meta=load(item['prepared'])
        if meta['inputs']!=snapshot([r['path'] for r in meta['inputs']]):raise ValueError('Observation inputs changed')


def prepare(plan,root):
    root=Path(root)
    freeze(root,plan,dry_run=True)
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/(root.name+'_preparation.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(root,plan)
        for item in plan['intervals']:
            check_plan(plan)
            stage=Stage(root/item['label'],'D0',d0_signature(plan,item,root))
            if stage.result:
                print('SKIP prepared '+item['label'],flush=True);continue
            out=stage.attempt();started=time.monotonic()
            with np.load(item['pairs'],allow_pickle=False) as z:pairs=dict(z)
            if not np.array_equal(pairs['frames'],np.arange(item['first'],item['stop'])):raise ValueError('Wrong pair frames')
            from diagnostics.parallax_io import intrinsics
            c=run_config(item['D2_runs'][0]);poses=np.asarray(c['pinhole_camera_config']['position'])[item['first']:item['stop']]
            if not np.allclose(pairs['poses'],poses,rtol=0,atol=1e-6) or not np.array_equal(pairs['intrinsics'],intrinsics(c['pinhole_camera_config'],item['size'])):
                raise ValueError('Observation camera/calibration mismatch')
            print('FIT D0 '+item['label'],flush=True)
            fit=fit_depth_only(pairs,_source_geometry(load(item['raw'])))
            full=evaluate_geometry(pairs,fit['geometry'],np.flatnonzero(pairs['visibility']==1),np.arange(len(pairs['source_points'])),
                                   ReprojectionFitConfig(reprojection_weight=0.))
            if (min(full['valid_pair_fraction'],full['valid_depth_fraction'])!=1 or full['max_camera_normalized_radius']>.6500001 or
                full['conservative_camera_clearance_m']<5.9999999 or full['minimum_axis_radius_m']<7.9999999 or full['maximum_axis_ratio']>10.0000001):
                raise ValueError('D0 failed full-ray validity/safety checks; preserve attempt and inspect')
            _copy_inspection_assets(Path(item['raw']),out)
            _write_diagnostics(out,fit,item['label'],Path(item['raw']),Path(item['prepared']))
            doc=_document(load(item['raw']),fit,Path(item['raw']),Path(item['prepared']),item['label'])
            doc['fit']['method']='Matched depth-only ablation: original solver/starts/safety; reprojection weight zero; no sampling floor'
            doc['ablation']=dict(recipe='D0',full_diagnostics=full,kernel_source_sha256=SOURCE_SHA256,
                elapsed_seconds=time.monotonic()-started,training_launched=False)
            from export_background_geometry_meshes import mesh_surface,write_mesh_ply,ELLIPSOID_BLUE
            vertices,faces=mesh_surface(fit['geometry']['center'],fit['geometry']['radii'],fit['geometry']['rotation'],48,96)
            write_mesh_ply(out/'ellipsoid.ply',vertices,faces,ELLIPSOID_BLUE)
            write_json(out/'ellipsoid.json',jsonable(doc))
            stage.complete(out/'ellipsoid.json',out)
            print('INSPECT '+str(out/'ellipsoid.ply'),flush=True)
        cells=matrix(plan,root,True)
        write_json(root/'matrix.json',dict(cells=cells,plan_sha256=sha(root/'manifest.json')))


def completed(path,item,cell):
    if not validate_run(path,item,cell['geometry']):return False
    c=run_config(path)
    if c.get('save_intermediate_checkpoints',False):raise ValueError('New ablations must save final checkpoint only')
    if Path(stats.resolved_value(c['background_ellipsoid_path'],REPO))!=Path(cell['geometry']).resolve():
        raise ValueError('Completed ablation used another geometry source')
    return True


def next_attempt(folder):
    ids=[]
    if folder.exists():
        for p in folder.iterdir():
            match=re.fullmatch(r'attempt_(\d+)(?:\.launch\.json)?',p.name)
            if match:ids.append(int(match[1]))
    return max(ids or [0])+1


def command(item,cell,out):
    # Native command builder supplies every input path, including the optional
    # mask filter; never reproduce the earlier missing-filter launcher bug.
    job={**item,**cell,'num_workers':10}
    # The approved current NAGConfig default is final-only; its bool CLI flag
    # is store_true, so passing a string 'False' would be invalid/enabling.
    c=safe_yaml(Path(item['config']).read_text())['NAGConfig']
    if c.get('save_intermediate_checkpoints',False):raise ValueError('Source YAML enables intermediate checkpoints')
    return training_command(job,out)


def fits_budget(elapsed,estimated,max_hours):return elapsed+estimated<=max_hours*3600


def stream_training(cmd,log_path):
    """Live stdout/stderr tee, retaining terminal progress bars when possible."""
    terminal=bool(getattr(sys.stdout,'isatty',lambda:False)())
    master=slave=None
    if terminal:
        import pty
        import termios
        master,slave=pty.openpty()
        try:
            size=fcntl.ioctl(sys.stdout.fileno(),termios.TIOCGWINSZ,b'\0'*8)
            fcntl.ioctl(slave,termios.TIOCSWINSZ,size)
        except OSError:
            pass
    target=getattr(sys.stdout,'buffer',None)
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace') if target is None else None
    process=None
    try:
        with Path(log_path).open('xb') as log:
            process=subprocess.Popen(cmd,cwd=REPO,stdout=slave if terminal else subprocess.PIPE,
                stderr=subprocess.STDOUT,bufsize=0,env={**os.environ,'PYTHONUNBUFFERED':'1'})
            if slave is not None:os.close(slave);slave=None
            fd=master if terminal else process.stdout.fileno()
            while True:
                ready,_,_=select.select([fd],[],[],.2)
                if not ready:
                    if process.poll() is not None:break
                    continue
                try:
                    chunk=os.read(fd,8192)
                except OSError as exc:
                    # A closed PTY reports EIO instead of an empty read.
                    if terminal and exc.errno==5:break
                    raise
                if not chunk:break
                log.write(chunk);log.flush()
                if target is not None:target.write(chunk);target.flush()
                else:sys.stdout.write(decoder.decode(chunk));sys.stdout.flush()
            if decoder is not None:sys.stdout.write(decoder.decode(b'',final=True));sys.stdout.flush()
            return process.wait()
    finally:
        if slave is not None:os.close(slave)
        if master is not None:os.close(master)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            if process.stdout is not None:process.stdout.close()


def train_or_summary(args,plan,cells):
    items={x['label']:x for x in plan['intervals']}
    selected=[c for c in cells if not args.intervals or c['label'] in args.intervals]
    if args.intervals and set(args.intervals)-set(items):raise ValueError('Unknown ablation interval')
    manifest=dict(schema=1,plan=plan,cells=cells,preparation_manifest=sha(args.preparation_root/'manifest.json'))
    freeze(args.training_root,manifest,dry_run=True)
    reused=done=pending=0
    remaining=[]
    for cell in selected:
        item=items[cell['label']]
        if cell['reused_run']:
            if not validate_run(cell['reused_run'],item,cell['geometry']):raise ValueError('Reused run is incomplete')
            reused+=1;continue
        folder=args.training_root/cell['id']
        good=[p for p in sorted(folder.glob('attempt_*')) if p.is_dir() and re.fullmatch('attempt_\d+',p.name) and completed(p,item,cell)]
        if len(good)>1:raise ValueError('Multiple completed attempts')
        if good:done+=1
        else:
            pending+=1;remaining.append(cell)
            if args.dry_run:print(shlex.join(command(item,cell,folder/f'attempt_{next_attempt(folder):03d}')),flush=True)
    print(f'STATUS reused={reused}; new completed={done}; new pending={pending}',flush=True)
    if args.dry_run or args.stage=='summary':return
    lock_root=REPO/'runs/autowork_generalized_background_training_lock';lock_root.mkdir(exist_ok=True,parents=True)
    with (lock_root/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(args.training_root,manifest)
        started=time.monotonic()
        # Interleave D0/D1 within each repeat. Never choose order by quality.
        remaining.sort(key=lambda c:(ABLATION_INTERVALS.index(c['label']),c['repeat'],c['recipe']))
        for cell in remaining:
            item=items[cell['label']]
            if not fits_budget(time.monotonic()-started,item['estimated_training_seconds'],args.max_hours):
                print('Budget stop before next job; rerun the same interval script to continue.',flush=True);break
            check_plan(plan)
            if sha(cell['geometry'])!=cell['geometry_sha256']:raise ValueError('Frozen geometry changed')
            folder=args.training_root/cell['id'];folder.mkdir(exist_ok=True,parents=True)
            n=next_attempt(folder);out=folder/f'attempt_{n:03d}'
            cmd=command(item,cell,out)
            write_json(folder/f'attempt_{n:03d}.launch.json',dict(command=cmd,cell=cell))
            print('TRAIN '+cell['id'],flush=True)
            log_path=folder/f'attempt_{n:03d}.console.log'
            returncode=stream_training(cmd,log_path)
            if returncode!=0:raise RuntimeError(f'Training failed; preserved log {log_path}; retry with the same script')
            if not completed(out,item,cell):raise RuntimeError('Incomplete final outputs; preserve attempt and inspect')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','prepare','train','summary'])
    p.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    p.add_argument('--preparation-root',type=Path,default=DEFAULT_PREP)
    p.add_argument('--training-root',type=Path,default=DEFAULT_TRAIN)
    p.add_argument('--intervals',nargs='+',choices=ABLATION_INTERVALS)
    p.add_argument('--max-hours',type=float,default=8.)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summary',action='store_true')
    args=p.parse_args()
    if args.summary:args.stage='summary'
    if args.intervals and args.stage not in ('train','summary'):p.error('--intervals selects training/summary queues; preparation audits all five')
    if not np.isfinite(args.max_hours) or args.max_hours<=0:p.error('Positive finite hours required')
    for root in (args.preparation_root,args.training_root):
        if not root.name.startswith('autowork_') or root.resolve()==REPO:
            p.error('Use separate autowork_ namespaces')
    if args.preparation_root.resolve()==args.training_root.resolve():p.error('Preparation/training roots must be distinct')
    plan=inventory(args.catalog)
    frozen=args.preparation_root/'manifest.json'
    if frozen.exists() and load(frozen)!=plan:raise ValueError('Preparation inputs/code/environment changed; use another root')
    cells=matrix(plan,args.preparation_root,require_prepared=args.stage in ('train','summary'))
    show(plan,cells)
    if args.stage=='audit' or args.dry_run and args.stage=='prepare':return
    if args.stage=='prepare':prepare(plan,args.preparation_root)
    else:train_or_summary(args,plan,cells)


if __name__=='__main__':main()
