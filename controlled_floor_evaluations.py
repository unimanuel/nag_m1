"""Evaluate frozen floor objectives/Jacobians without running an optimizer."""
import argparse
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import controlled_floor_replay as replay

DEFAULT_OUTPUT = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_floor_evaluations_v1')
OUTPUT_NAMES = ['score_weight5', 'score_weight10', 'score_weight20',
                'weighted_data_loss', 'floor_loss', 'reprojection_huber_loss', 'depth_huber_loss',
                'constraint_data_budget', 'constraint_camera_radius', 'constraint_clearance',
                'constraint_minimum_radius', 'constraint_axis_ratio', 'constraint_pair_validity',
                'constraint_depth_validity', 'constraint_sampling_validity']
PARAMETER_NAMES = ['center_local_x_m', 'center_local_y_m', 'center_local_z_m',
                   'log_radius_x', 'log_radius_y', 'log_radius_z']


class Intercepted(Exception):
    def __init__(self, capture):
        self.capture = capture


def capture_objective(functions, arm, pairs, original, row, cfg):
    """Let unmodified initialization execute, stop at first minimize call."""
    function = functions[arm]
    kernel = function.__globals__.get('kernel', function)
    namespace = kernel.__globals__
    actual_minimize = namespace['minimize']

    def intercept(fun, x0, **kwargs):
        if kwargs['method'] != 'SLSQP' or 'jac' in kwargs or 'eps' in kwargs['options']:
            raise ValueError('Unexpected optimizer/gradient options')
        score = inspect.getclosurevars(fun).nonlocals['score']
        measure = inspect.getclosurevars(score).nonlocals['measure']
        constraints = kwargs['constraints']['fun']
        local = inspect.getclosurevars(measure).nonlocals
        decode = local['decode']
        encode = inspect.getclosurevars(decode).nonlocals
        raise Intercepted({'score': score, 'measure': measure, 'constraints': constraints,
                           'cache': local['cache'], 'x0': x0.copy(), 'bounds': kwargs['bounds'],
                           'limit': inspect.getclosurevars(constraints).nonlocals['limit'],
                           'mean': encode['mean'], 'rotation': encode['rotation'],
                           'options': kwargs['options']})

    namespace['minimize'] = intercept
    try:
        replay.fit_arm(functions, arm, pairs, original, row, cfg)
    except Intercepted as caught:
        return caught.capture
    finally:
        namespace['minimize'] = actual_minimize
    raise ValueError('Fit skipped optimization: this diagnostic requires a nonzero-floor interval')


def aligned_copy(array, alignment=64):
    import numpy as np
    if not array.flags.c_contiguous:
        raise ValueError('Alignment intervention must not change array strides')
    buffer = np.empty(array.nbytes + alignment, dtype=np.uint8)
    offset = (-buffer.ctypes.data) % alignment
    result = buffer[offset:offset + array.nbytes].view(array.dtype).reshape(array.shape)
    result[...] = array
    if result.ctypes.data % alignment or result.strides != array.strides:
        raise ValueError('Could not preserve aligned array layout')
    return result


def evaluate_stencils(capture, parameter_vectors):
    import numpy as np
    from scipy.optimize._numdiff import approx_derivative
    from scipy.optimize._slsqp_py import _epsilon
    bounds = np.asarray(capture['bounds'], dtype=float).T
    methods = [('slsqp_forward', '2-point', float(_epsilon)),
               ('central_1e-6', '3-point', 1e-6), ('central_1e-4', '3-point', 1e-4)]
    records = {}
    for name, values in parameter_vectors.items():
        x = np.asarray(values, dtype=np.float64)
        if np.any(x < bounds[0]) or np.any(x > bounds[1]):
            raise ValueError('Probe vector lies outside historical bounds')
        capture['cache'].clear()
        samples = []

        def vector(parameters):
            data, floor = capture['measure'](parameters)
            outputs = np.r_[[capture['score'](parameters, w) for w in (5., 10., 20.)],
                            data['weighted_data_loss'], floor['loss'],
                            data['reprojection_huber_loss'], data['depth_huber_loss'],
                            capture['constraints'](parameters)]
            if len(outputs) != len(OUTPUT_NAMES) or not np.isfinite(outputs).all():
                raise ValueError('Unexpected/nonfinite objective vector')
            samples.append({'parameters': parameters.tolist(), 'values': outputs.tolist()})
            return outputs

        f0 = vector(x)
        repeated = []
        for _ in range(2):
            capture['cache'].clear()
            repeated.append(vector(x).tolist())
        derivatives = {}
        for label, method, step in methods:
            samples.clear()
            capture['cache'].clear()
            jac = approx_derivative(vector, x, method=method, abs_step=step, f0=f0, bounds=bounds)
            derivatives[label] = {'method': method, 'absolute_step': step,
                                  'jacobian': jac.tolist(), 'samples': list(samples)}
        records[name] = {'parameters': x.tolist(), 'values': f0.tolist(),
                         'uncached_repeats_exact': all(np.array_equal(f0, v) for v in repeated),
                         'derivatives': derivatives}
    return records


def frozen_context(replay_root, label):
    import numpy as np
    from preparation.storage import Stage
    from reprojection_ellipsoid import enforce_safety, ReprojectionFitConfig
    root = Path(replay_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, h in manifest['current_sources'].items():
        if replay.sha(replay.REPO / name) != h:
            raise ValueError('Previous replay code changed: ' + name)
    rows = replay.load_rows(replay.DEFAULT_FIXTURE, [label])
    row = rows[0]
    if row not in manifest['rows']:
        raise ValueError('Frozen row differs from previous replay')
    source = json.loads(Path(row['reprojection']).read_text())
    cfg = ReprojectionFitConfig(**source['reprojection_fit']['config'])
    with np.load(row['pairs'], allow_pickle=False) as archive:
        poses = archive['poses']
    original = replay.geometry(source)
    initial = enforce_safety(original, poses, cfg)
    mean, rotation = poses[:, :3, 3].mean(axis=0), initial['rotation']

    def encode(geometry):
        g = {k: np.asarray(v) for k, v in geometry.items()}
        if not np.array_equal(g['rotation'], rotation):
            raise ValueError('Probe geometries must have identical rotation')
        return np.r_[(g['center'] - mean) @ rotation, np.log(g['radii'])].tolist()

    vectors = {'initial': encode(initial), 'stored': encode(replay.geometry(json.loads(Path(row['floor']).read_text())))}
    prior_files = {str((root / 'manifest.json').resolve()): replay.sha(root / 'manifest.json')}
    for arm in replay.ARMS:
        records = []
        for job in manifest['jobs']:
            if job['label'] == label and job['arm'] == arm:
                stage = Stage(root, job['id'], manifest)
                if not stage.result:
                    raise ValueError('Previous replay is incomplete')
                records.append(json.loads(stage.result.read_text()))
                prior_files[str(stage.manifest.resolve())] = replay.sha(stage.manifest)
                prior_files[str(stage.result)] = replay.sha(stage.result)
        if len(records) < 2 or not all(replay.difference(records[0]['geometry'], r['geometry'])['exact'] for r in records):
            raise ValueError('Previous arm is absent or not exactly repeatable')
        vectors[arm + '_optimum'] = encode(records[0]['geometry'])
    return {'row': row, 'vectors': vectors, 'prior_files': prior_files,
            'previous_runtime': manifest['runtime']}


def worker(plan_path, job_id):
    plan = json.loads(Path(plan_path).read_text())
    os.sched_setaffinity(0, {plan['cpu']})
    import numpy as np
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    from preparation.storage import write_json, recipe
    from reprojection_ellipsoid import ReprojectionFitConfig
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    functions = replay.engines()
    row = plan['frozen']['row']
    for p, h in {**row['replay_inputs'], **plan['frozen']['prior_files']}.items():
        if replay.sha(p) != h:
            raise ValueError('Frozen input changed: ' + p)
    source = json.loads(Path(row['reprojection']).read_text())
    cfg = ReprojectionFitConfig(**source['reprojection_fit']['config'])
    if asdict(cfg) != recipe()['reprojection']:
        raise ValueError('Frozen fitting settings changed')
    with np.load(row['pairs'], allow_pickle=False) as archive:
        pairs = dict(archive)
    if job['layout'] == 'aligned64':
        pairs = {k: aligned_copy(v) for k, v in pairs.items()}
    original = replay.geometry(source)
    fingerprint = replay.input_fingerprint(pairs, original, row, asdict(cfg))
    with threadpool_limits(limits=1):
        runtime = replay.runtime(plan['cpu'])
        if runtime != plan['frozen']['previous_runtime']:
            raise ValueError('Runtime differs from previous controlled replay')
        start = time.monotonic()
        capture = capture_objective(functions, job['arm'], pairs, original, row, cfg)
        evaluations = evaluate_stencils(capture, plan['frozen']['vectors'])
    write_json(Path(plan_path).parent / 'result.json', {
        'schema': 1, 'job': job, 'runtime': runtime, 'elapsed_seconds': time.monotonic() - start,
        'input_digest': replay.digest(fingerprint), 'input_fingerprint': fingerprint,
        'alignment_mod64': {k: int(v.ctypes.data % 64) for k, v in pairs.items()},
        'native_x0': capture['x0'].tolist(), 'constraint_data_limit': capture['limit'],
        'optimizer_options': capture['options'], 'evaluations': evaluations})


def summarize(records):
    import numpy as np
    if len({r['input_digest'] for r in records}) != 1:
        raise ValueError('Input values, dtype or strides differ across evaluation jobs')

    def compare(a, b):
        results = {}
        for name in a['evaluations']:
            x, y = a['evaluations'][name], b['evaluations'][name]
            if x['parameters'] != y['parameters']:
                raise ValueError('Parameter probes differ')
            delta = np.asarray(x['values']) - y['values']
            derivatives = {}
            for key in x['derivatives']:
                sx, sy = x['derivatives'][key]['samples'], y['derivatives'][key]['samples']
                if [s['parameters'] for s in sx] != [s['parameters'] for s in sy]:
                    raise ValueError('Finite-difference stencils differ')
                jdelta = np.asarray(x['derivatives'][key]['jacobian']) - y['derivatives'][key]['jacobian']
                derivatives[key] = {'max_abs_by_output': np.max(np.abs(jdelta), axis=1).tolist(),
                                    'exact': bool(np.all(jdelta == 0))}
            results[name] = {'value_abs_differences': np.abs(delta).tolist(),
                             'values_exact': bool(np.all(delta == 0)), 'derivatives': derivatives}
        return results

    within, between, alignment = {}, {}, {}
    for layout in ('native', 'aligned64'):
        for arm in replay.ARMS:
            group = [r for r in records if r['job']['layout'] == layout and r['job']['arm'] == arm]
            within[layout + '_' + arm] = [compare(a, b) for a, b in itertools.combinations(group, 2)]
        for a, b in itertools.combinations(replay.ARMS, 2):
            pairs = [(x, y) for x in records for y in records if x['job']['layout'] == layout
                     and y['job']['layout'] == layout and x['job']['arm'] == a and y['job']['arm'] == b
                     and x['job']['repeat'] == y['job']['repeat']]
            between[layout + '_' + a + '_vs_' + b] = [compare(x, y) for x, y in pairs]
    for arm in replay.ARMS:
        pairs = [(x, y) for x in records for y in records if x['job']['layout'] == 'native'
                 and y['job']['layout'] == 'aligned64' and x['job']['arm'] == y['job']['arm'] == arm
                 and x['job']['repeat'] == y['job']['repeat']]
        alignment[arm] = [compare(x, y) for x, y in pairs]
    return {'input_digest': records[0]['input_digest'], 'within_arm': within,
            'between_arms': between, 'alignment_intervention': alignment,
            'all_uncached_repeats_exact': all(e['uncached_repeats_exact'] for r in records for e in r['evaluations'].values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-root', type=Path, default=replay.DEFAULT_OUTPUT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--sequence', default='W191_0_40')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-job', help=argparse.SUPPRESS)
    args = p.parse_args()
    os.environ.update(replay.THREAD_ENV)
    sys.dont_write_bytecode = True
    if args.worker_plan:
        worker(args.worker_plan, args.worker_job)
        return
    if args.repeats < 2 or not args.output_root.name.startswith('autowork_'):
        p.error('Use at least two repeats and an autowork output namespace')
    replay.historical_sources()
    frozen = frozen_context(args.replay_root, args.sequence)
    cpu = frozen['previous_runtime']['cpu']
    probe = subprocess.check_output([sys.executable, str(replay.REPO / 'controlled_floor_replay.py'),
                                    '--probe', '--cpu', str(cpu)], text=True)
    if json.loads(probe) != frozen['previous_runtime']:
        raise ValueError('Runtime differs from previous replay')
    jobs = []
    for i in range(args.repeats):
        for layout in ('native', 'aligned64'):
            for arm in replay.ARMS[i % 3:] + replay.ARMS[:i % 3]:
                jobs.append({'arm': arm, 'layout': layout, 'repeat': i + 1,
                             'id': f'autowork_{args.sequence}_{layout}_{arm}_repeat{i+1:02d}'})
    context = {'schema': 1, 'cpu': cpu, 'frozen': frozen, 'jobs': jobs,
               'sources': {n: replay.sha(replay.REPO / n) for n in
                           ['controlled_floor_evaluations.py', 'run_controlled_floor_evaluations.sh',
                            'controlled_floor_replay.py', *replay.HELPERS]},
               'output_names': OUTPUT_NAMES, 'parameter_names': PARAMETER_NAMES,
               'diagnostic_only': True, 'optimizer_runs': 0, 'tolerances_changed': False}
    manifest = args.output_root / 'manifest.json'
    if args.output_root.exists() and any(args.output_root.iterdir()) and (
            not manifest.exists() or json.loads(manifest.read_text()) != context):
        raise ValueError('Conflicting/unowned output: use a new autowork root')
    print(f'{len(jobs)} objective/gradient probes; no optimization or training', flush=True)
    if args.dry_run:
        print('Read-only preflight passed; no outputs written.'); return
    from preparation.storage import Stage, write_json
    import fcntl
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Evaluation already running') from None
        if not manifest.exists():
            write_json(manifest, context)
        records = []
        for job in jobs:
            if any(replay.sha(replay.REPO / n) != h for n, h in context['sources'].items()):
                raise ValueError('Diagnostic code changed during execution')
            stage = Stage(args.output_root, job['id'], context)
            if stage.result:
                result = stage.result
                print('[reuse] ' + job['id'], flush=True)
            else:
                attempt = stage.attempt()
                plan = attempt / 'plan.json'
                write_json(plan, context)
                print('[probe] ' + job['id'], flush=True)
                with (attempt / 'console.txt').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-plan', str(plan),
                                    '--worker-job', job['id']], stdout=log, stderr=subprocess.STDOUT, check=True)
                result = stage.complete(attempt / 'result.json', attempt)
            record = json.loads(result.read_text())
            records.append(record)
            print(f'  {record["elapsed_seconds"]:.2f}s', flush=True)
        write_json(args.output_root / 'summary.json', {'schema': 1, 'manifest_sha256': replay.sha(manifest),
                   'output_names': OUTPUT_NAMES, 'parameter_names': PARAMETER_NAMES, **summarize(records)})
        print('Summary: ' + str(args.output_root / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
