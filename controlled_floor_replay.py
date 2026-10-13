"""CPU-only historical-v1 / historical-v2 / generalized-v2 refinement replay.

This independent diagnostic neither changes nor certifies the preparation gate.
All arms use the same frozen inputs, shared numerical helpers and runtime.
"""
import argparse
import contextlib
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'nag'))
ARMS = ('historical_v1', 'historical_v2', 'generalized_v2')
COMMITS = {'historical_v1':'73e387497c011077f2ee4c74f62b691f56e363f9',
           'historical_v2':'07cbd743d0751a0c85167fbe0baff1c4112b6a13'}
THREAD_ENV = {'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
              'NUMEXPR_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1', 'OMP_DYNAMIC':'FALSE',
              'PYTHONHASHSEED':'0', 'PYTHONDONTWRITEBYTECODE':'1'}
HELPERS = ('nag/reprojection_ellipsoid.py', 'nag/texture_sampling_diagnostics.py',
           'nag/diagnostics/parallax_core.py', 'nag/diagnostics/parallax_evaluate.py',
           'nag/diagnostics/parallax_io.py')
DEFAULT_FIXTURE = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_generalized_preparation_v1/ego_fixtures.json')
DEFAULT_OUTPUT = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_controlled_floor_replay_v1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_sources():
    result = {}
    for arm, commit in COMMITS.items():
        source = subprocess.check_output(['git','show',f'{commit}:nag/sampling_floor_ellipsoid.py'],cwd=REPO)
        result[arm] = source
        for name in HELPERS:
            old = subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=REPO)
            if old != (REPO/name).read_bytes():
                raise ValueError(f'Shared numerical helper differs from {commit}: {name}; do not mix implementations')
    if result['historical_v2'] != (REPO/'nag/preparation/_legacy_floor_kernel.py').read_bytes():
        raise ValueError('Pinned generalized kernel differs from historical v2')
    return result


def engines():
    # Import/compile every arm in every worker, before loading observation arrays.
    result = {}
    for arm, source in historical_sources().items():
        module = types.ModuleType('autowork_replay_'+arm)
        exec(compile(source, f'git:{COMMITS[arm]}:sampling_floor_ellipsoid.py', 'exec'), module.__dict__)
        result[arm] = module.fit_sampling_floor
    from preparation.legacy_floor import fit_sampling_floor
    result['generalized_v2'] = fit_sampling_floor
    return result


def runtime(cpu):
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    with contextlib.redirect_stdout(io.StringIO()) as output:
        np.show_config()
    pools = sorted(threadpool_info(), key=lambda x:x['filepath'])
    if any(p['num_threads'] != 1 for p in pools):
        raise ValueError('Numerical thread pools must all have exactly one thread')
    if any(k in sys.modules for k in ('torch','tensorflow','lightning','pytorch_lightning')):
        raise ValueError('This diagnostic must not load a GPU/training framework')
    cpuinfo = Path('/proc/cpuinfo').read_text().split('\n\n')[0]
    return {'executable':sys.executable, 'python':sys.version, 'platform':platform.platform(),
            'numpy':np.__version__, 'scipy':scipy.__version__, 'numpy_build':output.getvalue(),
            'threadpools':pools, 'environment':{k:os.environ.get(k) for k in THREAD_ENV},
            'cpu':cpu, 'affinity':sorted(os.sched_getaffinity(0)),
            'cpu_features':[l for l in cpuinfo.splitlines() if l.startswith(('model name','flags'))],
            'aslr':Path('/proc/sys/kernel/randomize_va_space').read_text().strip()}


def geometry(document):
    import numpy as np
    f=document['fit']
    return {k:np.asarray(f[field],dtype=np.float64) for k,field in
            [('center','center'),('radii','radii'),('rotation','rotation_columns')]}


def input_fingerprint(pairs, original, row, config):
    arrays={k:{'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest(), 'dtype':v.dtype.str,
               'shape':list(v.shape), 'strides':list(v.strides)} for k,v in pairs.items()}
    return {'arrays':arrays, 'original':{k:v.tolist() for k,v in original.items()},
            'sphere':row['sphere'], 'mapping':row['mapping'], 'size':row['size'], 'config':config,
            'max_sampling':4096}


def fit_arm(functions, arm, pairs, original, row, cfg):
    kwargs={'max_sampling':4096}
    if arm!='historical_v1':kwargs['preserve_zero_floor']=True
    return functions[arm](pairs,original,row['sphere'],row['mapping'],tuple(row['size']),cfg,**kwargs)


def difference(a,b):
    import numpy as np
    errors={k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ('center','radii','rotation')}
    exact=all(np.array_equal(a[k],b[k]) for k in errors)
    passes=all(np.allclose(a[k],b[k],rtol=0 if k=='rotation' else 1e-6,
                         atol=1e-6 if k=='rotation' else 1e-4) for k in errors)
    return {'maximum_errors':errors,'exact':exact,'passes_existing_tolerance':bool(passes)}


def schedule(labels,repeats):
    # Rotate arm order to avoid always measuring one arm first/last.
    return [{'label':label,'arm':arm,'repeat':i+1,
             'id':f'autowork_{label}_{arm}_repeat{i+1:02d}'}
            for i in range(repeats) for label in labels for arm in ARMS[i%3:]+ARMS[:i%3]]


def load_rows(fixture_path, labels):
    fixture=json.loads(Path(fixture_path).read_text())
    rows=[]
    for label in labels:
        matches=[r for r in fixture['rows'] if r['label']==label]
        if len(matches)!=1 or matches[0]['floor_version']!=1:
            raise ValueError(f'{label}: choose an unambiguous historical v1-refined interval')
        row=matches[0]
        paths=[row['pairs'],row['reprojection'],row['floor'],row['sphere']['provenance'],
               str(Path(row['sphere_run'])/'training_config.yaml')]
        for p in paths:
            if p not in fixture['files'] or sha(p)!=fixture['files'][p]:
                raise ValueError(f'Frozen replay input changed or is not registered: {p}')
        row={**row,'replay_inputs':{p:fixture['files'][p] for p in paths}}
        rows.append(row)
    return rows


def worker(plan_path,job_id):
    plan=json.loads(Path(plan_path).read_text())
    os.sched_setaffinity(0,{plan['cpu']})
    import numpy as np
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    from reprojection_ellipsoid import ReprojectionFitConfig
    from diagnostics.parallax_io import jsonable
    from preparation.storage import recipe, write_json
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    row=next(r for r in plan['rows'] if r['label']==job['label'])
    functions=engines()
    for p,h in row['replay_inputs'].items():
        if sha(p)!=h:raise ValueError(f'Replay input changed: {p}')
    source=json.loads(Path(row['reprojection']).read_text())
    saved=json.loads(Path(row['floor']).read_text())
    cfg=ReprojectionFitConfig(**source['reprojection_fit']['config'])
    if asdict(cfg)!=recipe()['reprojection']:
        raise ValueError('Saved fitter parameters differ from the frozen publication recipe')
    if saved['fit']['sampling_floor']['provenance']['code_sha256'] != plan['historical_source_sha256']['historical_v1']:
        raise ValueError('Historical stored fit does not use the selected v1 source')
    with np.load(row['pairs'],allow_pickle=False) as archive:pairs=dict(archive)
    original=geometry(source)
    fingerprint=input_fingerprint(pairs,original,row,asdict(cfg))
    array_debug={k:{'alignment_mod64':v.ctypes.data%64,'c_contiguous':bool(v.flags.c_contiguous),
                    'f_contiguous':bool(v.flags.f_contiguous)} for k,v in pairs.items()}
    with threadpool_limits(limits=1):
        context=runtime(plan['cpu'])
        if context!=plan['runtime']:raise ValueError('Worker runtime differs from the controlled probe')
        start=time.monotonic()
        fit=fit_arm(functions,job['arm'],pairs,original,row,cfg)
        seconds=time.monotonic()-start
    report={'schema':1,'job':job,'runtime':context,'input_digest':digest(fingerprint),
            'input_fingerprint':fingerprint,'array_debug':array_debug,'elapsed_seconds':seconds,
            'geometry':jsonable(fit['geometry']),'fit':jsonable(fit),
            'versus_stored':difference(fit['geometry'],geometry(saved))}
    write_json(Path(plan_path).parent/'result.json',report)


def summarize(records,rows):
    def aggregate(pairs):
        d=[difference(a['geometry'],b['geometry']) for a,b in pairs]
        return {'pair_count':len(d),'exact_pair_count':sum(x['exact'] for x in d),
                'tolerance_pass_count':sum(x['passes_existing_tolerance'] for x in d),
                'maximum_errors':{k:max([x['maximum_errors'][k] for x in d],default=0.) for k in ('center','radii','rotation')},
                'maximum_combined_objective_difference':max([abs(a['fit']['sampling_floor']['final_combined_objective']-
                    b['fit']['sampling_floor']['final_combined_objective']) for a,b in pairs],default=0.)}
    result={}
    for row in rows:
        by_arm={arm:[r for r in records if r['job']['label']==row['label'] and r['job']['arm']==arm] for arm in ARMS}
        if len({r['input_digest'] for arm in ARMS for r in by_arm[arm]})!=1:
            raise ValueError('Worker observation arrays/layout or numerical inputs differ')
        result[row['label']]={'within_arm':{a:aggregate(list(itertools.combinations(by_arm[a],2))) for a in ARMS},
            'between_arms':{a+'_vs_'+b:aggregate(list(itertools.product(by_arm[a],by_arm[b]))) for a,b in itertools.combinations(ARMS,2)},
            'versus_stored':[{'arm':r['job']['arm'],'repeat':r['job']['repeat'],**r['versus_stored'],
                            'combined_objective_difference':r['fit']['sampling_floor']['final_combined_objective']-
                              json.loads(Path(row['floor']).read_text())['fit']['sampling_floor']['final_combined_objective']}
                             for arm in ARMS for r in by_arm[arm]]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,default=DEFAULT_FIXTURE)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--sequences',nargs='+',default=['W191_0_40'])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cpu',type=int)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--probe',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker-plan',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker-job',help=argparse.SUPPRESS)
    args=p.parse_args()
    os.environ.update(THREAD_ENV)
    sys.dont_write_bytecode=True
    if args.worker_plan:
        worker(args.worker_plan,args.worker_job);return
    cpu=args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    if cpu not in os.sched_getaffinity(0):p.error('Selected CPU is unavailable to this process')
    if args.probe:
        os.sched_setaffinity(0,{cpu});engines()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):print(json.dumps(runtime(cpu)))
        return
    if args.repeats<2 or len(set(args.sequences))!=len(args.sequences):p.error('Use at least two repeats and unique sequences')
    if not args.output_root.name.startswith('autowork_'):p.error('Output namespace must begin with autowork_')
    sources=historical_sources()
    rows=load_rows(args.fixtures,args.sequences)
    probe=subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--probe','--cpu',str(cpu)],cwd=REPO,text=True)
    from preparation.storage import Stage,write_json
    context={'schema':1,'fixture_sha256':sha(args.fixtures),'rows':rows,'cpu':cpu,'runtime':json.loads(probe),
        'jobs':schedule(args.sequences,args.repeats),'historical_commits':COMMITS,
        'historical_source_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
        'current_sources':{n:sha(REPO/n) for n in (*HELPERS,'nag/diagnostics/parallax_observations.py',
            'nag/preparation/_legacy_floor_kernel.py','nag/preparation/legacy_floor.py','nag/preparation/storage.py',
            'nag/preparation/jobs.py','controlled_floor_replay.py','run_controlled_ellipsoid_replay.sh')},
        'acceptance_policy':'diagnostic only; existing geometry tolerances unchanged; never releases training'}
    manifest=args.output_root/'manifest.json'
    if args.output_root.exists() and any(args.output_root.iterdir()) and (
            not manifest.exists() or json.loads(manifest.read_text())!=context):
        raise ValueError('Unowned/conflicting replay output; choose another autowork root')
    for job in context['jobs']:print(job['id'],flush=True)
    if args.dry_run:
        print('Read-only preflight passed; no fits or files created.');return
    args.output_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output_root/'replay.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This replay is already running') from None
        if not manifest.exists():write_json(manifest,context)
        records=[]
        for job in context['jobs']:
            if any(sha(REPO/n)!=h for n,h in context['current_sources'].items()):
                raise ValueError('Diagnostic or numerical code changed during replay')
            stage=Stage(args.output_root,job['id'],context)
            if stage.result:
                result=stage.result;print('[reuse] '+job['id'],flush=True)
            else:
                attempt=stage.attempt();plan=attempt/'plan.json';write_json(plan,context)
                print('[fit] '+job['id'],flush=True)
                with (attempt/'console.txt').open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-plan',str(plan),
                                    '--worker-job',job['id']],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=stage.complete(attempt/'result.json',attempt)
            record=json.loads(result.read_text());records.append(record)
            print(f'  {record["elapsed_seconds"]:.2f}s; stored comparison: {record["versus_stored"]}',flush=True)
        summary={'schema':1,'manifest_sha256':sha(manifest),'diagnostic_only':True,
                 'results':summarize(records,rows),'record_count':len(records)}
        write_json(args.output_root/'summary.json',summary)
        print('Summary: '+str(args.output_root/'summary.json'),flush=True)


if __name__=='__main__':main()
