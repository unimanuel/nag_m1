"""Audit the completed static ellipsoid cohort and export a reproducible report.

No training, rendering, checkpoint loading or source-result modification.
Generated CSV/JSON/Markdown outputs must use a new autowork_ namespace.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))
METRICS = ('MPSNR', 'SSIM', 'LPIPS')
ROOT_NAMES = (
    'autowork_final_ellipsoid_W125', 'autowork_final_ellipsoid_W141',
    'autowork_final_ellipsoid_W203_ready', 'autowork_final_ellipsoid_W324_10workers',
    'autowork_final_ellipsoid_W344_10workers', 'autowork_final_ellipsoid_W952_10workers',
    'autowork_final_ellipsoid_W975_ready_10workers',
)
OMITTED = {'W203_60_107', 'W324_96_161', 'W975_0_68', 'W975_69_99', 'W975_99_162'}


def label_key(label):
    scene, first, last = label.split('_')
    return int(scene[1:]), int(first), int(last)


def aggregate(rows, expected_labels):
    """Equal interval means after averaging repeats; never frame-weight results."""
    wanted = {(label, mode, repeat) for label in expected_labels
              for mode in ('regular', 'visuv') for repeat in (1, 2, 3)}
    keys = [(r['label'], r['mode'], r['repeat']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != wanted:
        raise ValueError('Incomplete, duplicate or unexpected interval/mode/repeat coverage')
    if any(r['workers'] != 10 or r['seed'] != 42 for r in rows):
        raise ValueError('Do not mix workers or seeds in the matched cohort')
    if any(not math.isfinite(r[m]) for r in rows for m in METRICS):
        raise ValueError('Nonfinite metric')
    groups = []
    for label in sorted(expected_labels, key=label_key):
        item = {'label': label}
        for mode in ('regular', 'visuv'):
            subset = [r for r in rows if r['label'] == label and r['mode'] == mode]
            item[mode] = {m: statistics.mean(r[m] for r in subset) for m in METRICS}
            item[mode + '_sd'] = {m: statistics.stdev(r[m] for r in subset) for m in METRICS}
        item['delta_uv_minus_regular'] = {m: item['visuv'][m] - item['regular'][m] for m in METRICS}
        groups.append(item)
    macro = {mode: {m: statistics.mean(g[mode][m] for g in groups) for m in METRICS}
             for mode in ('regular', 'visuv')}
    wins = {m: sum(g['delta_uv_minus_regular'][m] * (-1 if m == 'LPIPS' else 1) > 0
                   for g in groups) for m in METRICS}
    return {'intervals': groups, 'macro': macro, 'visible_uv_interval_wins': wins}


def collect_runs():
    from generalized_ellipsoid_preparation import training_completed, DEFAULT_VERIFICATION
    from paper_final_baselines import intervals, yaml_audit
    from paper_sphere_baselines import code_snapshot, duration_seconds
    from preparation.jobs import settings, sha, assert_training_worker_policy
    from preparation.storage import code_signature
    from preparation.verification import release_gate

    items = intervals()
    yaml_audit(items)
    selected = set(items) - OMITTED
    release_gate(DEFAULT_VERIFICATION / 'verification.json')
    training_code, preparation_code = code_snapshot(), code_signature()
    files, rows, incomplete, excluded = {}, [], [], []
    for name in ROOT_NAMES:
        root = REPO / 'runs' / name
        path = root / 'manifest.json'
        files[str(path)] = sha(path)
        doc = json.loads(path.read_text())
        context = doc['context']
        if (context['code'] != training_code or context['preparation_code'] != preparation_code
                or context['gate'] != sha(DEFAULT_VERIFICATION / 'verification.json')
                or context['executable'] != sys.executable
                or context['versions'] != {k: importlib.metadata.version(k) for k in context['versions']}
                or context['environment'] != {k: os.environ.get(k) for k in context['environment']}):
            raise ValueError(f'Changed provenance/environment: {root}')
        for job in doc['jobs']:
            if job['label'] in OMITTED:
                excluded.append(job['id'])
                continue
            item = items[job['label']]
            uid = f'segment-{item["segment"]}_frames_{item["first"]:03d}_{item["stop"]-1:03d}_step1'
            if job['uid'] != uid:
                raise ValueError('Full segment/frame identity mismatch')
            for field in ('config', 'geometry'):
                if sha(job[field]) != job[field + '_sha256']:
                    raise ValueError(f'Changed {field}: {job["id"]}')
                files[job[field]] = job[field + '_sha256']
            assert_training_worker_policy(job, item['segment'], settings(job['config'])['num_workers'])
            good = []
            for attempt in sorted((root / job['id']).glob('attempt_*')):
                values = training_completed(attempt, job)
                if values is None:
                    incomplete.append(str(attempt))
                else:
                    good.append((attempt, values))
            if len(good) != 1:
                raise ValueError(f'{job["id"]}: expected one valid completion, got {len(good)}')
            attempt, values = good[0]
            config = settings(attempt / 'training_config.yaml')
            checkpoints = list((attempt / 'checkpoints').glob('*'))
            if len(checkpoints) != 1 or not checkpoints[0].is_file() or checkpoints[0].stat().st_size <= 0:
                raise ValueError(f'Expected final-only checkpoint: {attempt}')
            for evidence in [attempt / 'training_config.yaml', attempt / 'exit_000.txt',
                             attempt / 'initial/background_diagnostics/ellipsoid_geometry.txt'] + [
                    attempt / 'tracker' / f'train_epoch_Mean{m}.csv' for m in METRICS]:
                files[str(evidence)] = sha(evidence)
            rows.append({'label': job['label'], 'uid': uid, 'mode': job['mode'],
                         'repeat': job['repeat'], 'seed': config['seed'], 'workers': config['num_workers'],
                         'frames': job['frame_count'], 'minutes': duration_seconds(attempt / 'exit_000.txt') / 60,
                         **values, 'path': str(attempt)})
        print(f'Validated {name}', flush=True)
    summary = aggregate(rows, selected)
    summary.update(incomplete_attempts=incomplete, excluded_manifest_jobs=excluded,
                   matched_static_baseline_manifests=[str(p) for p in
                       sorted((REPO / 'runs').glob('autowork_paper_final_baselines_*/manifest.json'))],
                   provenance_sha256=files)
    return rows, summary


def markdown(summary):
    def triple(values):
        return f'{values["MPSNR"]:.3f} / {values["SSIM"]:.4f} / {values["LPIPS"]:.4f}'
    count = len(summary['intervals'])
    lines = ['# Static/final ellipsoid comparison', '',
             f'{count} intervals, two texture layouts and three seed-42 repeats: {count * 6} validated trainings.',
             'Means below weight intervals equally, after averaging their three repeats.',
             'These are training-nondeterminism repeats, not independent-seed trials.', '',
             '| Treatment | PSNR / SSIM / LPIPS |', '| --- | --- |']
    for mode, name in [('regular', 'Ellipsoid regular'), ('visuv', 'Ellipsoid Visible-UV')]:
        lines.append(f'| {name} | {triple(summary["macro"][mode])} |')
    lines += ['', 'Visible-UV wins by interval (PSNR, SSIM, LPIPS): ' +
              str(summary['visible_uv_interval_wins']), '',
              '| Interval | Regular PSNR / SSIM / LPIPS | Visible-UV PSNR / SSIM / LPIPS | ΔPSNR | ΔSSIM | ΔLPIPS |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for g in summary['intervals']:
        d = g['delta_uv_minus_regular']
        lines.append(f'| {g["label"]} | {triple(g["regular"])} | {triple(g["visuv"])} | '
                     f'{d["MPSNR"]:+.3f} | {d["SSIM"]:+.4f} | {d["LPIPS"]:+.4f} |')
    lines += ['', 'Positive ΔPSNR/ΔSSIM and negative ΔLPIPS favor Visible-UV.', '',
              'Omitted intervals: ' + ', '.join(sorted(OMITTED, key=label_key)) + '.',
              'W324 96–161 is deferred for resources; the other four await depth restoration.',
              'Historical/four-worker runs are not substituted or pooled.',
              'Saved calibration, geometry, epoch-80 metrics, complete final-frame coverage and final-only checkpoints passed validation.',
              'This audit does not rerender checkpoints or decode every final PNG.', '',
              'Matched static baseline manifest count: ' + str(len(summary['matched_static_baseline_manifests'])) + '.',
              'Do not infer an ellipsoid advantage over sphere or plane from unmatched historical runs.',
              'Use identical interval sets for the forthcoming five-treatment comparison.', '',
              'Incomplete earlier attempts (retained, not counted):', '']
    lines += ['- ' + p for p in summary['incomplete_attempts']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_static_final_ellipsoids_2026-10-20'))
    args = parser.parse_args()
    if not args.output_root.name.startswith('autowork_') or args.output_root.exists():
        parser.error('Use a new autowork_ output directory; existing outputs are never overwritten')
    rows, summary = collect_runs()
    args.output_root.mkdir(parents=True)
    with (args.output_root / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    report = markdown(summary)
    (args.output_root / 'comparison.md').write_text(report)
    print(report)
    print('Outputs:', args.output_root)


if __name__ == '__main__':
    main()
