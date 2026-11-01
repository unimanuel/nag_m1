"""Preliminary matched background tables with separate frame and repeat SDs.

Read-only CSV/YAML/manifest analysis: no NAG imports, images, models or training.
Dependencies: NumPy, PyYAML and PyMuPDF. Generated reports stay outside Git.
"""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
METRICS = ('MPSNR', 'SSIM', 'LPIPS')
TREATMENTS = ('plane', 'sphere_regular', 'sphere_visuv', 'ellipsoid_regular', 'ellipsoid_visuv')
NAMES = ('Plane', 'Sphere', 'Sphere UV', 'Ellipsoid', 'Ellipsoid UV')
GROUPS = ('ego', 'final')
PENDING = ('W203_60_107', 'W975_0_68', 'W975_69_99', 'W975_99_162')
OMITTED = set(PENDING) | {'W324_96_161'}
SPHERE = 'nag.model.view_dependent_background_image_sphere_scene_node_3d.ViewDependentBackgroundImageSphereSceneNode3D'
PLANE = 'nag.model.view_dependent_background_image_plane_scene_node_3d.ViewDependentBackgroundImagePlaneSceneNode3D'
ELLIPSOID = 'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D'
UV = {'enabled': True, 'margin_uv': [.05, .05], 'interior_samples_per_axis': 9}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_key(label):
    scene, start, stop = label.split('_')
    return int(scene[1:]), int(start), int(stop)


def last_row(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    if not rows or float(rows[-1]['step']) != 80:
        raise ValueError(f'Expected final epoch 80: {path}')
    return rows[-1]


def frame_statistics(text, frame_count):
    """Strictly parse NumPy's saved 1-D vector, without silently dropping tokens."""
    if not text.strip().startswith('[') or not text.strip().endswith(']'):
        raise ValueError('Expected a bracketed per-frame vector')
    tokens = text.strip()[1:-1].split()
    try:
        values = np.asarray([float(t) for t in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError('Malformed or truncated per-frame vector') from exc
    if len(values) != frame_count or frame_count < 2 or not np.isfinite(values).all():
        raise ValueError('Wrong frame count or nonfinite per-frame metric')
    return float(values.mean()), float(values.std(ddof=0))


def complete_means(path, frame_count):
    """Same success/frame-coverage/final-scalar contract as the paper launchers."""
    try:
        if not (path / 'exit_000.txt').read_text().startswith('Exit code: 0.'):
            return None
        frames = sorted(int(p.name.split('_')[0]) for p in (path / 'final/complete').glob('*_t.png'))
        if frames != list(range(frame_count)):
            return None
        means = [float(last_row(path / 'tracker' / f'train_epoch_Mean{m}.csv')['value']) for m in METRICS]
        return means if np.isfinite(means).all() else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


def manifests(repo):
    rows = []
    for scene in ('W125', 'W141', 'W203', 'W324_without_96_161', 'W344', 'W952', 'W975'):
        rows.append(('final', 'baseline', repo / 'runs' / ('autowork_paper_final_baselines_' + scene)))
    for root in ('autowork_final_ellipsoid_W125', 'autowork_final_ellipsoid_W141',
                 'autowork_final_ellipsoid_W203_ready', 'autowork_final_ellipsoid_W324_10workers',
                 'autowork_final_ellipsoid_W344_10workers', 'autowork_final_ellipsoid_W952_10workers',
                 'autowork_final_ellipsoid_W975_ready_10workers'):
        rows.append(('final', 'ellipsoid', repo / 'runs' / root))
    rows += [('ego', 'sphere', repo / f'runs/autowork_paper_sphere_baselines_part{n}of5') for n in range(1, 6)]
    rows += [('ego', 'plane', repo / f'runs/autowork_paper_plane_baselines_part{n}of2') for n in range(1, 3)]
    rows += [('ego', 'legacy', repo / f'runs/2026-09-25_ego_poles_repeatability_part{n}of4') for n in range(1, 5)]
    return rows


def resolved_value(value, repo):
    while isinstance(value, dict):
        value = value['value']
    return None if value is None else str((repo / value).resolve())


def normalized_config(c, treatment, repo):
    c = json.loads(json.dumps(c))
    expected_type = PLANE if treatment == 'plane' else SPHERE if treatment.startswith('sphere') else ELLIPSOID
    if (c['background_plane_type'] != expected_type or c['seed'] != 42 or c['num_workers'] != 10
            or c['max_epochs'] != 80 or c.get('deterministic', False)
            or c.get('checkpoint_path') or c.get('checkpoint_tracker_path')):
        raise ValueError('Unexpected saved treatment/training settings')
    if c['background_plane_kwargs'].get('color_mask_resolution') != [960, 1920]:
        raise ValueError('Not the selected nominal texture budget')
    uv = c['background_plane_kwargs'].pop('sphere_visible_uv_texture', None)
    if uv != (UV if treatment.endswith('visuv') else None):
        raise ValueError('Unexpected Visible-UV layout')
    if treatment.startswith('ellipsoid'):
        if (not c.get('background_ellipsoid_path') or c.get('background_ellipsoid_scale_margin') != 0
                or c.get('background_ellipsoid_texture_orientation') != 'legacy'
                or c.get('background_ellipsoid_max_radius') is not None
                or c.get('background_ellipsoid_radii_override') is not None):
            raise ValueError('Not the selected fixed legacy ellipsoid treatment')
    elif c.get('background_ellipsoid_path') or c.get('background_sphere_radius_override') is not None:
        raise ValueError('Unexpected geometry override')
    if treatment.startswith('sphere') and not c.get('background_sphere_radius_from_visible_plane_centers'):
        raise ValueError('Unexpected sphere-radius policy')
    for k in ('data_path', 'images_path', 'masks_path', 'depths_path', 'boxes_path',
              'boxes_object_id_mask_mapping_path', 'mask_ids_filter_path'):
        c[k] = resolved_value(c.get(k), repo)
    for k in ('name', 'output_path', 'experiment_datetime', 'background_plane_type',
              'background_ellipsoid_path', 'background_ellipsoid_scale_margin',
              'background_sphere_radius_from_visible_plane_centers', 'save_intermediate_checkpoints'):
        c.pop(k, None)
    return c


def collect(repo=REPO, manifest_rows=None, omitted=None, yaml_loader=None):
    records, provenance, incomplete, normalized = [], {}, [], {}

    def evidence(path):
        provenance[str(path)] = sha(path)

    if omitted is None:
        omitted = OMITTED
    for group, family, root in manifests(repo) if manifest_rows is None else manifest_rows:
        manifest = root / 'manifest.json'
        doc = json.loads(manifest.read_text())
        evidence(manifest)
        count = 0
        for job in doc['jobs']:
            label = job['label']
            if label in omitted or (family == 'legacy' and job['orientation'] != 'legacy'):
                continue
            treatment = job.get('treatment') or ('plane' if family == 'plane' else
                ('sphere_' if family == 'sphere' else 'ellipsoid_') + job['mode'])
            for field in ('config', 'geometry', 'reference_geometry'):
                digest = job.get(field + '_sha256')
                if digest is None and field == 'config':
                    digest = doc.get('context', {}).get('files', {}).get(str(Path(job[field]).resolve()))
                if digest is not None:
                    if sha(job[field]) != digest:
                        raise ValueError(f'Changed source {field}: {job["id"]}')
                    evidence(Path(job[field]))
            good = []
            for path in sorted((root / job['id']).glob('attempt_*')):
                values = complete_means(path, job['frame_count'])
                if values is None:
                    incomplete.append(str(path))
                else:
                    good.append((path, values))
            if len(good) != 1:
                raise ValueError(f'Expected one complete attempt: {job["id"]}; got {len(good)}')
            path, means = good[0]
            c = (yaml_loader or yaml.safe_load)((path / 'training_config.yaml').read_text())['NAGConfig']
            sl = c['frame_indices_filter']
            _, start, last = label.split('_')
            segment = Path(resolved_value(c['data_path'], repo)).name
            if (sl['start'], sl['stop'], sl.get('step') or 1, job['frame_count']) != (
                    int(start), int(last) + 1, 1, int(last) - int(start) + 1):
                raise ValueError('Wrong interval/frame identity')
            if not re.fullmatch(r'segment-\d+', segment) or segment[8:11] != label[1:4]:
                raise ValueError('Wrong full segment identity')
            canonical = normalized_config(c, treatment, repo)
            key = group, label
            if key in normalized and normalized[key] != canonical:
                raise ValueError(f'Unmatched effective configurations: {label}, {treatment}')
            normalized[key] = canonical
            frame_sd = []
            gaps = []
            for m, mean in zip(METRICS, means):
                csv_path = path / 'tracker' / f'train_epoch_{m}.csv'
                array_mean, sd = frame_statistics(last_row(csv_path)['value'], job['frame_count'])
                if not np.isclose(array_mean, mean, rtol=1e-6, atol=1e-6):
                    raise ValueError(f'Per-frame vector and saved mean differ: {csv_path}')
                frame_sd.append(sd)
                gaps.append(abs(array_mean - mean))
                evidence(csv_path)
                evidence(path / 'tracker' / f'train_epoch_Mean{m}.csv')
            for p in (path / 'training_config.yaml', path / 'exit_000.txt'):
                evidence(p)
            records.append(dict(group=group, label=label, segment=segment, treatment=treatment,
                repeat=job['repeat'], frames=job['frame_count'], seed=c['seed'], workers=c['num_workers'],
                mean=means, frame_sd=frame_sd, rounding_gap=gaps, path=str(path),
                physical_path=str(path.resolve()), manifest=str(manifest)))
            count += 1
        print(f'[verified] {root.name}: {count} runs', flush=True)
    return records, dict(files_sha256=provenance, incomplete_attempts=incomplete,
                         supplementary_sha256=sha(repo / 'static/pdfs/nag-supplementary.pdf'))


def aggregate(records, expected_counts=None):
    expected_counts = expected_counts or {'ego': 7, 'final': 20}
    keys = [(r['group'], r['label'], r['treatment'], r['repeat']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate run identity')
    groups = {}
    for group, count in expected_counts.items():
        labels = sorted({r['label'] for r in records if r['group'] == group}, key=label_key)
        if len(labels) != count:
            raise ValueError(f'Expected {count} intervals for {group}')
        wanted = {(group, l, t, n) for l in labels for t in TREATMENTS for n in (1, 2, 3)}
        if {k for k in keys if k[0] == group} != wanted:
            raise ValueError(f'Incomplete treatment/repeat coverage: {group}')
        intervals = {}
        for label in labels:
            intervals[label] = {}
            for t in TREATMENTS:
                rows = sorted([r for r in records if (r['group'], r['label'], r['treatment']) ==
                               (group, label, t)], key=lambda r: r['repeat'])
                if len({(r['segment'], r['frames']) for r in rows}) != 1:
                    raise ValueError('Conflicting segment/frame identity')
                a = np.asarray([r['mean'] for r in rows]); s = np.asarray([r['frame_sd'] for r in rows])
                if (a.shape != (3, 3) or s.shape != (3, 3) or not np.isfinite(a).all()
                        or not np.isfinite(s).all() or (s < 0).any()
                        or any(r['seed'] != 42 or r['workers'] != 10 for r in rows)):
                    raise ValueError('Invalid values, seeds or workers')
                intervals[label][t] = dict(mean=a.mean(0).tolist(), frame_sd=s.mean(0).tolist(),
                                           repeat_sd=a.std(0, ddof=1).tolist())

        def reduce(values):
            return {t: {field: np.mean([v[t][field] for v in values], axis=0).tolist()
                        for field in ('mean', 'frame_sd')} for t in TREATMENTS}

        scenes = {scene: reduce([v for l, v in intervals.items() if l.split('_')[0] == scene])
                  for scene in sorted({l.split('_')[0] for l in labels}, key=lambda s: int(s[1:]))}
        groups[group] = dict(intervals=intervals, scenes=scenes, interval_macro=reduce(list(intervals.values())),
                             scene_macro=reduce(list(scenes.values())))
    if {r['group'] for r in records} != set(expected_counts):
        raise ValueError('Unexpected cohort')
    return groups


def cell(item, metric, ascii_only=False):
    precision = 3 if metric == 0 else 5
    sign = '+/-' if ascii_only else '\u00b1'
    return f'{item["mean"][metric]:.{precision}f} {sign} {item["frame_sd"][metric]:.{precision}f}'


def text_report(groups, records, provenance, report_date):
    lines = ['PRELIMINARY RESULTS - background treatments', 'Report date: ' + report_date, '',
        'Scope: 405 completed matched trainings; 7 ego + 20 final intervals; 9 parent scenes.',
        'Five treatments, three fresh seed-42 nondeterminism repeats per interval.',
        'PRELIMINARY: four restored-depth final intervals are not included yet:',
        ', '.join(PENDING),
        'Their depth is restored; their matched trainings remain pending/not incorporated.',
        'W324_96_161 remains separately excluded for resource/runtime reasons.', '',
        'DEFINITIONS',
        'Mean: official final epoch-80 tracker scalar; averaged over the three repeats.',
        'Frame SD: population SD (divisor F) of per-frame metric values within EACH run.',
        'Displayed +/-: average of those three frame SDs, NOT an across-repeat SD or a confidence interval.',
        'Run SD: sample SD (divisor 2) of the three official run means; reported separately.',
        'Aggregate mean and frame SD: equal-weight average of interval statistics.',
        'Parent-scene rows: equal-weight interval statistics within that parent.',
        'Scene-balanced robustness: equal-weight parent-scene rows.',
        'Do not pool frames or mix between-interval difficulty into the frame SD.',
        'This follows the inter-frame concept of NAG supplementary C.1/Table 3;',
        'our population-SD convention and three-repeat averaging are explicit reporting choices.',
        'Frame SD is quality variation, not a direct flicker metric. Low SD alone need not mean good quality.',
        'Metrics are reconstruction metrics, not held-out/novel-view evidence.',
        'Higher PSNR and SSIM, lower LPIPS are better. PSNR unit: dB.', '',
        'AGGREGATE TABLES: mean +/- average inter-frame SD']
    for group, g in groups.items():
        for weighting, key in [('equal intervals', 'interval_macro'), ('equal parent scenes', 'scene_macro')]:
            lines += ['', group.upper() + ' - ' + weighting,
                      'Treatment | PSNR | SSIM | LPIPS']
            for t, name in zip(TREATMENTS, NAMES):
                lines.append(name + ' | ' + ' | '.join(cell(g[key][t], m, True) for m in range(3)))
        for scope in ('scenes', 'intervals'):
            lines += ['', group.upper() + ' - COMPLETE ' + scope.upper(),
                      'Identity | Treatment | PSNR +/- frame SD | SSIM +/- frame SD | LPIPS +/- frame SD']
            for label, values in g[scope].items():
                for t, name in zip(TREATMENTS, NAMES):
                    lines.append(label + ' | ' + name + ' | ' + ' | '.join(cell(values[t], m, True) for m in range(3)))
        lines += ['', group.upper() + ' - ACROSS-REPEAT SAMPLE SD (not frame SD)',
                  'Interval | Treatment | PSNR run SD | SSIM run SD | LPIPS run SD']
        for label, values in g['intervals'].items():
            for t, name in zip(TREATMENTS, NAMES):
                lines.append(label + ' | ' + name + ' | ' + ' | '.join(f'{v:.9g}' for v in values[t]['repeat_sd']))
    lines += ['', 'COMPLETE RUN-LEVEL APPENDIX',
              'Group | Interval | Treatment | Repeat | Frames | PSNR | PSNR frame SD | SSIM | SSIM frame SD | LPIPS | LPIPS frame SD']
    ordered = sorted(records, key=lambda r: (GROUPS.index(r['group']), label_key(r['label']),
                                           TREATMENTS.index(r['treatment']), r['repeat']))
    for r in ordered:
        numbers = [x for pair in zip(r['mean'], r['frame_sd']) for x in pair]
        lines.append(' | '.join(map(str, (r['group'], r['label'], r['treatment'], r['repeat'], r['frames'])))
                     + ' | ' + ' | '.join(f'{v:.9g}' for v in numbers))
        lines.append('  Source: ' + r['path'])
        if r['physical_path'] != r['path']:
            lines.append('  Archive/junction target: ' + r['physical_path'])
    lines += ['', 'RETAINED INCOMPLETE ATTEMPTS - NOT COUNTED'] + provenance['incomplete_attempts']
    lines += ['', 'SOURCE MANIFESTS (complete per-file hashes in summary.json)']
    for p, h in provenance['files_sha256'].items():
        if p.endswith('/manifest.json'):
            lines.append(h + '  ' + p)
    lines += ['', 'No training, checkpoint loading, image rerendering or pixel-error generation was performed.',
              'CSV vector precision is rounded by the original logger; vector means were checked against official scalars.',
              'Largest absolute scalar/vector mean discrepancies (PSNR, SSIM, LPIPS): ' +
              str(np.max([r['rounding_gap'] for r in records], axis=0).tolist())]
    return '\n'.join(lines) + '\n'


class PDF:
    """Small vector-table writer: black text, light cells, overflow checks."""
    def __init__(self, report_date, banner='PRELIMINARY - four restored-depth intervals pending',
                 footer='PRELIMINARY', title='PRELIMINARY background results: mean and inter-frame SD',
                 subject='405 matched trainings; restored-depth intervals not incorporated'):
        import pymupdf
        self.lib = pymupdf
        self.doc = pymupdf.open()
        self.date = report_date
        self.banner, self.footer = banner, footer
        self.title, self.subject = title, subject

    def page(self, title, subtitle=''):
        self.p = self.doc.new_page(width=842, height=595)
        self.p.draw_rect(self.lib.Rect(28, 15, 814, 37), color=None, fill=(1, .95, .78))
        self.p.insert_text((34, 30), self.banner, fontsize=10, fontname='hebo')
        self.p.insert_text((30, 62), title, fontsize=17, fontname='hebo')
        self.y = 75
        if subtitle:
            self.para(subtitle, 9)

    def para(self, text, size=10):
        width = 782
        words, line, lines = text.split(), '', []
        for word in words:
            candidate = (line + ' ' + word).strip()
            if self.lib.get_text_length(candidate, fontsize=size) > width and line:
                lines.append(line); line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        if self.y + len(lines) * (size + 4) > 557:
            raise ValueError('PDF paragraph overflow')
        for line in lines:
            self.y += size + 4
            self.p.insert_text((30, self.y), line, fontsize=size, color=(0, 0, 0))
        self.y += 6

    def table(self, headers, rows, widths=None, size=9, height=23):
        widths = widths or [142] + [128] * 5
        if abs(sum(widths) - 782) > .1:
            raise ValueError('Table widths must sum to 782')
        for i, row in enumerate([headers] + rows):
            if self.y + height > 557:
                raise ValueError('PDF table overflow')
            x = 30
            for text, w in zip(row, widths):
                rect = self.lib.Rect(x, self.y, x + w, self.y + height)
                self.p.draw_rect(rect, color=(.65, .65, .65), width=.35,
                                 fill=(.86, .91, .96) if i == 0 else (1, 1, 1) if i % 2 else (.96, .97, .98))
                font = 'hebo' if i == 0 else 'helv'
                fs = size
                while self.lib.get_text_length(str(text), fontname=font, fontsize=fs) > w - 8 and fs > 7:
                    fs -= .2
                if self.lib.get_text_length(str(text), fontname=font, fontsize=fs) > w - 8:
                    raise ValueError('PDF cell text overflow: ' + str(text))
                self.p.insert_text((x + 4, self.y + (height + fs) / 2 - 2), str(text),
                                   fontsize=fs, fontname=font, color=(0, 0, 0))
                x += w
            self.y += height
        self.y += 10

    def save(self, path):
        for i, p in enumerate(self.doc):
            p.insert_text((30, 579), f'{self.footer} | {self.date} | page {i+1}/{len(self.doc)}',
                          fontsize=8, color=(0, 0, 0))
        self.doc.set_metadata({'title': self.title, 'subject': self.subject})
        self.doc.save(path, garbage=4, deflate=True)
        self.doc.close()


def pdf_report(path, groups, records, report_date):
    p = PDF(report_date)
    p.page('Background comparison: preliminary results',
           '405 trainings | 27 intervals | 9 parent scenes | all five treatments | three seed-42 repeats')
    for group in GROUPS:
        p.para(group.upper() + ': interval-balanced mean +/- averaged inter-frame SD', 11)
        values = groups[group]['interval_macro']
        p.table(['Treatment', 'PSNR (dB), higher', 'SSIM, higher', 'LPIPS, lower'],
                [[name] + [cell(values[t], m) for m in range(3)] for t, name in zip(TREATMENTS, NAMES)],
                [206, 192, 192, 192], height=24)
    p.para('The +/- values above describe frame-to-frame reconstruction-quality variation, not training-repeat uncertainty.', 10)
    p.para('The four restored-depth intervals are excluded from this fixed preliminary snapshot. W324 96-161 remains excluded for resources.', 9)

    p.page('Definitions, aggregation and pending coverage')
    for text in [
        'Official means are final epoch-80 tracker scalars. Frame SD is calculated separately within every run over all selected frames, with population divisor F. We average the three run means and the three frame SDs for each interval.',
        'Across-repeat SD is a separate statistic: sample SD of the three run means, divisor 2. It measures same-seed nondeterminism, not independent-seed robustness. Its tables are explicitly separate from frame-SD tables.',
        'Primary aggregates weight intervals equally. Parent-scene rows average their interval statistics; the scene-balanced robustness table weights parents equally. We never pool frames across intervals, average frame vectors before taking SD, or call an averaged SD a confidence interval.',
        'This follows the inter-frame reporting concept in NAG supplementary C.1, Tables 3 and 11. The population-SD convention and averaging over three repeats are explicit choices here; the paper captions do not specify their denominator.',
        'Lower frame SD can also reflect uniformly poor reconstructions or easier frames. It is not a direct flicker or novel-view metric. Near-static final and ego cohorts are reported separately; not all final intervals are stationary.',
        'The main cohort uses legacy ellipsoid poles and zero ellipsoid runtime margin, ten workers and matched per-interval configurations. Historical intermediate-checkpoint storage differences do not change metrics. Ego sampler/update counts vary by interval; final uses the approved common template.',
    ]:
        p.para(text)
    p.table(['Excluded interval', 'Reason'], [[s, 'Depth restored; matched trainings pending/not incorporated'] for s in PENDING]
            + [['W324_96_161', 'Resource/runtime exclusion; not a missing-input exclusion']], [206, 576], height=22)

    p.page('Scene-balanced robustness: same statistics, different weights')
    for group in GROUPS:
        p.para(group.upper() + ': equal weight per parent scene', 11)
        v = groups[group]['scene_macro']
        p.table(['Treatment', 'PSNR mean +/- frame SD', 'SSIM mean +/- frame SD', 'LPIPS mean +/- frame SD'],
                [[name] + [cell(v[t], m) for m in range(3)] for t, name in zip(TREATMENTS, NAMES)],
                [206, 192, 192, 192], height=24)
    p.para('Frame SD is still averaged within the stated hierarchy; it is NOT standard deviation across scene means.', 10)

    for m, metric in enumerate(('PSNR', 'SSIM', 'LPIPS')):
        p.page('Parent-scene comparison: ' + metric,
               'Each cell: mean +/- averaged inter-frame SD; intervals are averaged within their parent scene.')
        rows = []
        for group in GROUPS:
            for scene, v in groups[group]['scenes'].items():
                rows.append([group.title() + ' ' + scene] + [cell(v[t], m) for t in TREATMENTS])
        p.table(['Parent scene'] + list(NAMES), rows, height=28)

    for group in GROUPS:
        labels = list(groups[group]['intervals'])
        for m, metric in enumerate(('PSNR', 'SSIM', 'LPIPS')):
            for first in range(0, len(labels), 10):
                chunk = labels[first:first+10]
                p.page(f'{group.title()} intervals: {metric} ({first+1}-{first+len(chunk)})',
                       'TOP: mean +/- averaged inter-frame SD. BOTTOM: across-repeat sample SD only (not temporal variation).')
                vals = groups[group]['intervals']
                p.table(['Interval'] + list(NAMES), [[l] + [cell(vals[l][t], m) for t in TREATMENTS] for l in chunk], height=20)
                p.para('Across-repeat sample SD of the three run means (no +/- frame SD in this table):', 9)
                precision = 4 if m == 0 else 6
                p.table(['Interval'] + list(NAMES), [[l] + [f'{vals[l][t]["repeat_sd"][m]:.{precision}f}'
                                                         for t in TREATMENTS] for l in chunk], height=18)

    ordered = sorted(records, key=lambda r: (GROUPS.index(r['group']), label_key(r['label']),
                                           TREATMENTS.index(r['treatment']), r['repeat']))
    for group in GROUPS:
        subset = [r for r in ordered if r['group'] == group]
        for first in range(0, len(subset), 20):
            p.page(f'Complete run-level appendix: {group} ({first+1}-{min(first+20,len(subset))})',
                   'Every completed run, no best-repeat selection. Each metric cell is that run mean +/- its own inter-frame SD.')
            rows = [[r['label'], NAMES[TREATMENTS.index(r['treatment'])], str(r['repeat'])] +
                    [cell(r, m) for m in range(3)] for r in subset[first:first+20]]
            p.table(['Interval', 'Treatment', 'Repeat', 'PSNR', 'SSIM', 'LPIPS'], rows,
                    [124, 112, 44, 166, 168, 168], height=21, size=8.5)
    p.page('Provenance and limitations')
    for text in [
        'Exactly one valid attempt is selected for every interval/treatment/repeat. Success requires exit code zero, exact final-frame coverage, finite epoch-80 scalars and complete per-frame vectors. Recorded source YAML/geometry hashes and effective per-interval configurations are checked.',
        'summary.json records full source-file SHA256 evidence, logical and resolved archive paths, incomplete attempts and all statistics. runs.csv and the plain-text report contain every one of the 405 selected runs. Directory junctions preserve access to archived sources; no historical cache metadata is rewritten.',
        'The logger rounded its stored vectors. Their means are checked against official scalar means; the latter remain the authoritative reported means. Frame SDs use the saved vectors in float64.',
        'The four restored-depth intervals are deliberately not added piecemeal to this 27-interval snapshot. Their completed five-treatment repetitions should be incorporated together in a later, separately labelled release.',
        'No confidence intervals, significance claims or independent-scene claims are implied by the SD bars. Adjacent intervals and overlapping endpoints belong to the same parent scene.',
        'No training, checkpoint loading, rendering or image-based metric recomputation was performed. The statistical report does not establish novel-view parallax or editing quality.',
    ]:
        p.para(text, 11)
    p.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(
        '/mnt/d/CGProjectData/waymo/diagnostics/autowork_preliminary_background_statistics_2026-10-29'))
    parser.add_argument('--report-date', default='2026-10-29')
    args = parser.parse_args()
    date.fromisoformat(args.report_date)
    if args.output_root.exists() or not args.output_root.name.startswith('autowork_'):
        parser.error('Choose a new autowork_ report directory; existing reports are never overwritten')
    records, provenance = collect()
    groups = aggregate(records)
    # Build before creating the output folder so validation failures never leave reports.
    text = text_report(groups, records, provenance, args.report_date)
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / 'preliminary_background_results.txt').write_text(text, encoding='utf-8')
    summary = dict(status='PRELIMINARY', date=args.report_date, groups=groups, runs=records,
                   pending=list(PENDING), resource_exclusion='W324_96_161', provenance=provenance,
                   frame_sd_ddof=0, repeat_sd_ddof=1)
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (args.output_root / 'runs.csv').open('w', newline='') as f:
        fields = ['group','label','segment','treatment','repeat','frames','path','physical_path']
        fields += [m+'_'+s for m in METRICS for s in ('mean','frame_sd')]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in fields[:8]},
                             **{m+'_'+s:r[s][i] for i,m in enumerate(METRICS) for s in ('mean','frame_sd')}})
    pdf_report(args.output_root / 'preliminary_background_results.pdf', groups, records, args.report_date)
    print('REPORTS:', args.output_root, flush=True)
    print('AGGREGATES:', json.dumps({g:v['interval_macro'] for g,v in groups.items()}), flush=True)


if __name__ == '__main__':
    main()
