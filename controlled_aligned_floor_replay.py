"""Full frozen sampling-floor replay with aligned observation arrays only.

Diagnostic, not a production numerical change or a preparation release gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import controlled_floor_replay as replay
import controlled_floor_evaluations as evaluations

DEFAULT_OUTPUT = Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_aligned_floor_replay_v1')


def schedule(label, repeats):
    return [{**job, 'layout': 'aligned64',
             'id': f'autowork_{label}_aligned64_{job["arm"]}_repeat{job["repeat"]:02d}'}
            for job in replay.schedule([label], repeats)]


def validate_sources(context):
    for name, h in context['current_sources'].items():
        if replay.sha(replay.REPO / name) != h:
            raise ValueError('Replay/numerical source changed: ' + name)


def baseline_records(context):
    from preparation.storage import Stage
    root = Path(context['baseline_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    result = []
    for job in manifest['jobs']:
        if job['label'] == context['row']['label']:
            stage = Stage(root, job['id'], manifest)
            if not stage.result:
                raise ValueError('Unaligned baseline is incomplete')
            result.append(json.loads(stage.result.read_text()))
    if not result:
        raise ValueError('Baseline interval is missing')
    return result


def summarize(records, context):
    if len(records) != len(context['jobs']) or {r['job']['id'] for r in records} != {j['id'] for j in context['jobs']}:
        raise ValueError('Aligned replay is incomplete or contains conflicting jobs')
    if any(any(v != 0 for v in r['alignment_mod64'].values()) for r in records):
        raise ValueError('Observation alignment is not controlled')
    prior = baseline_records(context)
    if len({r['input_digest'] for r in [*records, *prior]}) != 1:
        raise ValueError('Aligned/unaligned numerical inputs or array layouts differ')
    summary = replay.summarize(records, [context['row']])[context['row']['label']]
    summary['versus_prior_unaligned'] = {}
    for arm in replay.ARMS:
        original = [r for r in prior if r['job']['arm'] == arm]
        if len(original) < 2 or not all(replay.difference(original[0]['geometry'], r['geometry'])['exact'] for r in original):
            raise ValueError('Unaligned baseline arm is missing or not exactly repeatable')
        summary['versus_prior_unaligned'][arm] = [
            {'repeat': r['job']['repeat'], **replay.difference(r['geometry'], original[0]['geometry']),
             'combined_objective_difference': r['fit']['sampling_floor']['final_combined_objective'] -
                                             original[0]['fit']['sampling_floor']['final_combined_objective']}
            for r in records if r['job']['arm'] == arm]
    summary['all_rotations_exact_to_stored'] = all(r['versus_stored']['maximum_errors']['rotation'] == 0 for r in records)
    summary['all_stored_geometry_comparisons_pass'] = all(r['versus_stored']['passes_existing_tolerance'] for r in records)
    summary['all_between_arm_geometry_comparisons_pass'] = all(
        g['tolerance_pass_count'] == g['pair_count'] for g in summary['between_arms'].values())
    summary['all_within_arm_repeats_exact'] = all(
        g['exact_pair_count'] == g['pair_count'] for g in summary['within_arm'].values())
    # Guard against silently interpreting a tiny objective delta as a release.
    summary['release_authorized'] = False
    return summary


def worker(plan_path, job_id):
    plan = json.loads(Path(plan_path).read_text())
    os.sched_setaffinity(0, {plan['cpu']})
    import numpy as np
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    from reprojection_ellipsoid import ReprojectionFitConfig
    from diagnostics.parallax_io import jsonable
    from preparation.storage import recipe, write_json
    validate_sources(plan)
    functions = replay.engines()
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    row = plan['row']
    for p, h in {**row['replay_inputs'], **plan['prior_files']}.items():
        if replay.sha(p) != h:
            raise ValueError('Frozen input/baseline changed: ' + p)
    source = json.loads(Path(row['reprojection']).read_text())
    stored = json.loads(Path(row['floor']).read_text())
    cfg = ReprojectionFitConfig(**source['reprojection_fit']['config'])
    if asdict(cfg) != recipe()['reprojection']:
        raise ValueError('Original optimizer configuration differs from frozen recipe')
    source_hash = hashlib.sha256(replay.historical_sources()['historical_v1']).hexdigest()
    if stored['fit']['sampling_floor']['provenance']['code_sha256'] != source_hash:
        raise ValueError('Stored baseline does not use historical v1')
    with np.load(row['pairs'], allow_pickle=False) as archive:
        pairs = dict(archive)
    original = replay.geometry(source)
    native = replay.input_fingerprint(pairs, original, row, asdict(cfg))
    pairs = {k: evaluations.aligned_copy(v) for k, v in pairs.items()}
    aligned = replay.input_fingerprint(pairs, original, row, asdict(cfg))
    if native != aligned:
        raise ValueError('Alignment changed observation values, dtype or strides')
    alignment = {k: int(v.ctypes.data % 64) for k, v in pairs.items()}
    if any(alignment.values()):
        raise ValueError('All observation arrays must be 64-byte-aligned')
    with threadpool_limits(limits=1):
        runtime = replay.runtime(plan['cpu'])
        if runtime != plan['runtime']:
            raise ValueError('Worker runtime differs from frozen baseline')
        start = time.monotonic()
        # Execute the unmodified continuation, candidate selection, safety
        # repair and full-observation checks. No minimize interception here.
        fit = replay.fit_arm(functions, job['arm'], pairs, original, row, cfg)
        seconds = time.monotonic() - start
    if replay.input_fingerprint(pairs, original, row, asdict(cfg)) != aligned:
        raise ValueError('Fitter mutated frozen inputs')
    validate_sources(plan)
    write_json(Path(plan_path).parent / 'result.json', {
        'schema': 1, 'job': job, 'runtime': runtime, 'input_digest': replay.digest(aligned),
        'input_fingerprint': aligned, 'alignment_mod64': alignment,
        'elapsed_seconds': seconds, 'geometry': jsonable(fit['geometry']), 'fit': jsonable(fit),
        'versus_stored': replay.difference(fit['geometry'], replay.geometry(stored)),
        'diagnostic_only': True, 'production_recipe_changed': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-root', type=Path, default=replay.DEFAULT_OUTPUT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--sequence', default='W191_0_40')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-new-fits', type=int, help='Limit fits in this invocation; later invocations reuse them')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-job', help=argparse.SUPPRESS)
    args = p.parse_args()
    os.environ.update(replay.THREAD_ENV)
    sys.dont_write_bytecode = True
    if args.worker_plan:
        worker(args.worker_plan, args.worker_job); return
    if args.repeats < 2 or not args.output_root.name.startswith('autowork_'):
        p.error('Use at least two repeats and an autowork output namespace')
    if args.max_new_fits is not None and args.max_new_fits < 1:
        p.error('--max-new-fits must be positive')
    replay.historical_sources()
    frozen = evaluations.frozen_context(args.replay_root, args.sequence)
    baseline = json.loads((args.replay_root / 'manifest.json').read_text())
    cpu = frozen['previous_runtime']['cpu']
    probe = subprocess.check_output([sys.executable, str(replay.REPO / 'controlled_floor_replay.py'),
                                    '--probe', '--cpu', str(cpu)], text=True, cwd=replay.REPO)
    if json.loads(probe) != frozen['previous_runtime']:
        raise ValueError('Current runtime differs from unaligned replay')
    extra_sources = ['controlled_aligned_floor_replay.py', 'run_controlled_aligned_floor_replay.sh',
                     'controlled_floor_evaluations.py']
    context = {'schema': 1, 'cpu': cpu, 'runtime': frozen['previous_runtime'], 'row': frozen['row'],
               'baseline_root': str(args.replay_root.resolve()), 'prior_files': frozen['prior_files'],
               'current_sources': {**baseline['current_sources'],
                                   **{n: replay.sha(replay.REPO / n) for n in extra_sources}},
               'jobs': schedule(args.sequence, args.repeats), 'observation_alignment_bytes': 64,
               'optimizer_options_changed': False, 'production_recipe_changed': False,
               'acceptance_policy': 'diagnostic only; existing tolerances unchanged; never releases training'}
    manifest = args.output_root / 'manifest.json'
    if args.output_root.exists() and any(args.output_root.iterdir()) and (
            not manifest.exists() or json.loads(manifest.read_text()) != context):
        raise ValueError('Unowned/conflicting output: use a new autowork root')
    print(f'{len(context["jobs"])} full aligned refinements; no training', flush=True)
    if args.dry_run:
        print('Read-only preflight passed; no fits or files created.'); return
    from preparation.storage import Stage, write_json
    import fcntl
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / 'replay.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Aligned replay already running') from None
        if not manifest.exists():
            write_json(manifest, context)
        records, new_fits = [], 0
        for job in context['jobs']:
            validate_sources(context)
            stage = Stage(args.output_root, job['id'], context)
            if stage.result:
                result = stage.result
                print('[reuse] ' + job['id'], flush=True)
            else:
                if args.max_new_fits is not None and new_fits >= args.max_new_fits:
                    print('Invocation fit limit reached; completed stages retained. '
                          'Run again without --max-new-fits to finish. No complete summary written.', flush=True)
                    return
                attempt = stage.attempt()
                plan = attempt / 'plan.json'
                write_json(plan, context)
                print('[fit] ' + job['id'], flush=True)
                with (attempt / 'console.txt').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-plan', str(plan),
                                    '--worker-job', job['id']], cwd=replay.REPO, stdout=log,
                                    stderr=subprocess.STDOUT, check=True)
                result = stage.complete(attempt / 'result.json', attempt)
                new_fits += 1
            record = json.loads(result.read_text())
            records.append(record)
            print(f'  {record["elapsed_seconds"]:.2f}s; stored: {record["versus_stored"]}', flush=True)
        write_json(args.output_root / 'summary.json', {'schema': 1, 'diagnostic_only': True,
                   'manifest_sha256': replay.sha(manifest), 'record_count': len(records),
                   'sequence': args.sequence, 'results': summarize(records, context)})
        print('Summary: ' + str(args.output_root / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
