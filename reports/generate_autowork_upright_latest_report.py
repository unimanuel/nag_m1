"""Build the latest upright results and chronological recipe-history PDF.

Only completed epoch-80 runs are used. Generated artifacts stay outside Git;
the generator is committed, and the PDF/CSV/provenance files live on drive D.
Requires NumPy, PyYAML, matplotlib and PyMuPDF. No training is launched.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
from generate_sampling_floor_v2_typeset import MathPage
matplotlib.rcParams['figure.max_open_warning'] = 40
METRICS = ('MPSNR', 'SSIM', 'LPIPS')


def completed(folder, frame_count):
    """Same completion contract as ego_repeatability, without training imports."""
    if not (folder / 'exit_000.txt').is_file():
        return None
    frames = sorted(int(p.name.split('_')[0])
                    for p in (folder / 'final/complete').glob('*_t.png'))
    if frames != list(range(frame_count)):
        return None
    result = {}
    try:
        for metric in METRICS:
            with (folder / 'tracker' / f'train_epoch_Mean{metric}.csv').open() as file:
                row = list(csv.DictReader(file, delimiter=';'))[-1]
            value = float(row['value'])
            if float(row['step']) != 80 or not math.isfinite(value):
                return None
            result[metric] = value
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return result


def table(p, headers, rows, widths, size=9):
    height = (len(rows) + 1) * 23
    ax = p.figure.add_axes([36/792, (p.y-height)/612, 720/792, height/612])
    ax.axis('off')
    cells = ax.table(cellText=rows, colLabels=headers, cellLoc='left',
                     colLoc='left', colWidths=widths, bbox=[0, 0, 1, 1])
    cells.auto_set_font_size(False)
    cells.set_fontsize(size)
    for (row, _), cell in cells.get_celld().items():
        cell.set_facecolor('#dce5ef' if row == 0 else ('#f1f4f7' if row % 2 else '#ffffff'))
        cell.set_edgecolor('#b6c2cf')
        cell.set_linewidth(.5)
        cell.get_text().set_color('#080808')
        if row == 0:
            cell.get_text().set_weight('bold')
    p.y -= height + 20
    p.check()

LABELS = ['W191_0_40', 'W191_40_128', 'W191_128_198', 'W254_0_37',
          'W254_37_81', 'W254_86_137', 'W254_139_195']
PROBLEM = ['W191_0_40', 'W254_0_37', 'W254_139_195']
FAMILIES = ['sphere', 'legacy', 'upright', 'upright_fit', 'strong_floor', 'compact']
NAMES = ['Sphere reference', 'Legacy floor / legacy poles', 'Fixed narrow / upright poles',
         'Old upright-aware floor', 'Strong floor 0.8 / upright', 'Compact 0.15 / upright']
NEW_ROOTS = ['autowork_upright_optimizer_validation_regular_34c7664f',
             'autowork_upright_optimizer_validation_regular_aaa12435']
VERSIONS = {'strong_floor': 'upright_strong_floor_t0800_v1',
            'compact': 'upright_compact_w0150_v1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_path(label, version):
    prefix, first, last = label.split('_')
    segment = '1918764220984209654' if prefix == 'W191' else '2547899409721197155'
    root = Path('/mnt/d/CGProjectData/waymo/ellipsoids') / f'segment-{segment}'
    stem = f'frames_{int(first):03d}_{int(last):03d}'
    name = (f'autowork_{stem}_{version}' if version.startswith('upright_')
            else f'{stem}_reprojection_depth_{version}')
    return root / name / 'ellipsoid.json'


def read_geometry(path):
    fit = json.loads(Path(path).read_text())['fit']
    return {key: np.asarray(fit[key], dtype=float)
            for key in ['center', 'radii', 'rotation_columns']}


def config_differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((config_differences(a.get(k), b.get(k), prefix + '/' + k)
                    for k in sorted(set(a) | set(b))), [])
    return [] if a == b else [(prefix, a, b)]


def load_data():
    values, sources = defaultdict(list), []
    historical = HERE / '2026-09-28_upright_comparison.sources.json'
    for record in json.loads(historical.read_text())['runs']:
        label = record['sequence']
        _, first, last = label.split('_')
        path = Path(record['path'])
        result = completed(path, int(last) - int(first) + 1)
        if result is None:
            raise ValueError(f'Incomplete historical run: {path}')
        key = label, record['mode'], record['family']
        values[key].append(np.asarray([result[m] for m in METRICS]))
        sources.append({**record, 'metrics': result,
                        'metric_sha256': {m: sha(path / 'tracker' / f'train_epoch_Mean{m}.csv')
                                          for m in METRICS}})
    reference_manifest = REPO / 'runs/2026-09-28_upright_sampling_floor/manifest.json'
    snapshots = []
    for name in NEW_ROOTS:
        root = REPO / 'runs' / name
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['reference_manifest_sha256'] != sha(reference_manifest):
            raise ValueError('Reference manifest changed')
        snapshots.append({'path': str(manifest_path), 'sha256': sha(manifest_path),
                          'manifest': manifest})
        for job in manifest['jobs']:
            if job['sequence'] not in PROBLEM or job['mode'] != 'regular':
                raise ValueError('Unexpected latest cohort')
            for field in ['geometry', 'config']:
                if sha(job[field]) != job[field + '_sha256']:
                    raise ValueError(f'Training input changed: {job[field]}')
            report = Path(job['geometry']).with_name('sampling_floor_diagnostics.json')
            if sha(report) != job['report_sha256']:
                raise ValueError(f'Preparation report changed: {report}')
            attempts = [(p, completed(p, job['frame_count']))
                        for p in (root / job['id']).glob('autowork_attempt_*')]
            attempts = [(p, v) for p, v in attempts if v is not None]
            if len(attempts) != 1:
                raise ValueError(f'Ambiguous/incomplete latest run: {job["id"]}')
            path, result = attempts[0]
            cfg = yaml.safe_load((path / 'training_config.yaml').read_text())['NAGConfig']
            expected = {'seed': 42, 'max_epochs': 80, 'deterministic': False,
                        'background_ellipsoid_texture_orientation': 'world_y',
                        'background_ellipsoid_scale_margin': 0.0,
                        'background_ellipsoid_radii_override': None,
                        'background_ellipsoid_max_radius': None}
            if any(cfg.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Unexpected latest settings: {path}')
            old = next(r for r in sources if (r['sequence'], r['mode'], r['family']) ==
                       (job['sequence'], 'regular', 'upright_fit'))
            old_cfg = yaml.safe_load((Path(old['path']) / 'training_config.yaml').read_text())['NAGConfig']
            allowed = {'name', 'output_path', 'experiment_datetime', 'background_ellipsoid_path'}
            differences = config_differences(cfg, old_cfg)
            unexpected = [d for d in differences if d[0].split('/')[1] not in allowed]
            if unexpected:
                raise ValueError(f'Uncontrolled config differences: {unexpected}')
            values[job['sequence'], 'regular', job['recipe']].append(
                np.asarray([result[m] for m in METRICS]))
            sources.append({'sequence': job['sequence'], 'mode': 'regular',
                            'family': job['recipe'], 'path': str(path), 'metrics': result,
                            'geometry': job['geometry'], 'geometry_sha256': job['geometry_sha256'],
                            'config_differences': differences,
                            'metric_sha256': {m: sha(path / 'tracker' / f'train_epoch_Mean{m}.csv')
                                              for m in METRICS}})
    reuse = []
    for label in LABELS:
        for mode in ['regular', 'visuv']:
            for family, count in zip(FAMILIES[:4], [1, 3, 3, 1]):
                if len(values[label, mode, family]) != count:
                    raise ValueError(f'Historical cohort count mismatch: {label} {mode} {family}')
        for recipe in VERSIONS:
            if label in PROBLEM:
                if len(values[label, 'regular', recipe]) != 1:
                    raise ValueError('Missing new result')
                continue
            old_path = geometry_path(label, 'sampling_floor_world_y_v2')
            new_path = geometry_path(label, VERSIONS[recipe])
            old, new = read_geometry(old_path), read_geometry(new_path)
            if not all(np.array_equal(old[k], new[k]) for k in old):
                raise ValueError(f'Cannot reuse nonidentical geometry: {label} {recipe}')
            reuse.append({'sequence': label, 'recipe': recipe,
                          'reuse_family': 'upright_fit', 'old_geometry': str(old_path),
                          'new_geometry': str(new_path), 'old_sha256': sha(old_path),
                          'new_sha256': sha(new_path), 'geometry_arrays_exactly_equal': True})
    return values, sources, snapshots, reuse


def label_text(label):
    prefix, first, last = label.split('_')
    return f'{prefix} {first}-{last}'


def mean(values, label, family, mode='regular', reuse=False):
    key = label, mode, family
    if key not in values and reuse and mode == 'regular' and label not in PROBLEM:
        key = label, mode, 'upright_fit'
    return np.mean(values[key], axis=0)


def aggregate(values, labels, family, mode='regular', reuse=False):
    return np.mean([mean(values, label, family, mode, reuse) for label in labels], axis=0)


def fmt(vector):
    return [f'{vector[0]:.3f}', f'{vector[1]:.5f}', f'{vector[2]:.5f}']


def page(title, subtitle):
    return MathPage(title, subtitle)


def results_pages(values):
    p = page('Latest upright ellipsoids: results and recipe history',
             '2 October 2026 | six new regular-UV trainings | W191 and W254 ego intervals')
    p.paragraph('Both new axis-neutral recipes repair the old wide upright-aware fits. Strong floor is better on average, but mostly recovers the earlier fixed-narrow upright quality. Neither establishes an advantage over the legacy-pole reference. Compact wins W254 0-37 but loses to strong floor on the other two intervals.')
    rows = [[name] + fmt(aggregate(values, PROBLEM, family))
            for family, name in zip(FAMILIES, NAMES)]
    table(p, ['Three-interval treatment', 'PSNR up', 'SSIM up', 'LPIPS down'],
          rows, [.55, .15, .15, .15], 9)
    p.paragraph('Macro average = equal weight for W191 0-40, W254 0-37, and W254 139-195. Higher PSNR/SSIM and lower LPIPS are better. These are saved final epoch-80 image metrics, not held-out-scene generalization or neural-training loss values.')
    p.paragraph('All six new jobs passed successful-exit, complete-frame and finite epoch-80 checks. Settings: seed 42, nondeterministic training, upright world-Y poles, regular UV, zero ellipsoid runtime margin. Original YAML and geometry hashes match the recorded manifests. Training configuration matches old upright-aware controls apart from geometry and run metadata.')
    p.paragraph('The two earlier fixed-geometry pole treatments average three same-seed repeats per interval. Sphere, old upright-aware and the new recipes have one run each. No new Visible-UV result is available. The cohort contains only two parent Waymo segments.')
    yield p

    for label in PROBLEM:
        p = page('Individual comparison: ' + label_text(label),
                 'Same interval and regular UV | repeated baselines shown as mean +/- sample SD')
        rows = []
        for family, name in zip(FAMILIES, NAMES):
            a = np.asarray(values[label, 'regular', family])
            cells = fmt(a.mean(axis=0))
            if len(a) > 1:
                sd = fmt(a.std(axis=0, ddof=1))
                cells = [f'{v} +/- {s}' for v, s in zip(cells, sd)]
            rows.append([name, str(len(a))] + cells)
        table(p, ['Treatment', 'n', 'PSNR', 'SSIM', 'LPIPS'], rows,
              [.39, .04, .19, .19, .19], 8)
        rows = []
        for family in ['strong_floor', 'compact']:
            for baseline in ['upright_fit', 'upright', 'legacy']:
                delta = mean(values, label, family) - mean(values, label, baseline)
                rows.append([family + ' minus ' + baseline] +
                            [f'{delta[0]:+.3f}', f'{delta[1]:+.5f}', f'{delta[2]:+.5f}'])
        table(p, ['Signed change', 'PSNR', 'SSIM', 'LPIPS'], rows,
              [.55, .15, .15, .15], 8)
        p.paragraph('Positive PSNR/SSIM changes and negative LPIPS changes favor the new recipe. Baseline names: upright_fit = old wide upright-aware fit; upright = fixed narrow geometry with upright poles; legacy = legacy-floor geometry with legacy poles.')
        yield p

    p = page('Interpretation: repair versus a new best method',
             'Small differences require repeats; local geometry diagnostics do not predict every quality change')
    for family in ['strong_floor', 'compact']:
        delta = aggregate(values, PROBLEM, family) - aggregate(values, PROBLEM, 'upright_fit')
        p.paragraph(f'{family}: mean change against old wide upright = {delta[0]:+.3f} dB PSNR, {delta[1]:+.5f} SSIM, {delta[2]:+.5f} LPIPS. Every interval improves in all three metrics against that old wide treatment.')
    p.paragraph('Strong floor gains only 0.042 dB on average over the previous fixed-narrow upright mean, with almost unchanged SSIM and slightly worse LPIPS. Its W191 and W254 139-195 PSNR values fall inside the previous upright repeat ranges; W254 0-37 is just above that range. These are not replicated significance tests.')
    p.paragraph('Compact exceeds every previous narrow-upright repeat in PSNR and SSIM on W254 0-37 and improves LPIPS, but is worse than the narrow-upright mean in PSNR/LPIPS on W191 0-40 and in all metrics on W254 139-195. Greater compactness is not consistently better.')
    p.paragraph('Both recipes beat the historical sphere on W191 0-40 in all metrics. Both lose to sphere on the two W254 intervals in all metrics. Strong floor also remains worse than the legacy-pole ellipsoid in all three metrics in all three intervals.')
    p.paragraph('Recommended upright candidate: strong floor, with compact retained as an ablation. Compare actual learned flow and spatial errors against fixed-narrow upright and legacy poles before introducing another loss. Removing lateral-major shapes improves these cases but is not proof that such shapes are intrinsically invalid for all scenes.')
    yield p

    p = page('Seven-interval regular-UV summary',
             'Assembled comparison: six new jobs plus explicitly reused unchanged-geometry results')
    table(p, ['Treatment', 'PSNR', 'SSIM', 'LPIPS'],
          [[name] + fmt(aggregate(values, LABELS, family, reuse=True))
           for family, name in zip(FAMILIES, NAMES)], [.55, .15, .15, .15], 9)
    p.paragraph('For compact and strong floor ONLY: the three problem intervals use the new runs. The other four reuse the completed old upright-aware runs. Their center, radii and rotation arrays are verified exactly identical to the newly generated recipe geometries. This is an assembled estimate, not a fresh seven-interval sweep.')
    p.paragraph('Reused intervals: W191 40-128, W191 128-198, W254 37-81, W254 86-137. Equal interval weighting is applied after repeat averaging. These totals must not be presented as seven new trainings or compared with a three-interval average.')
    p.paragraph('The legacy-floor/legacy-pole recipe remains the strongest of these seven-interval treatments on all three macro metrics. Strong floor improves the old upright-aware average, but does not establish overall superiority over sphere: its PSNR is lower, while its SSIM and LPIPS are better.')
    yield p

    for index, metric in enumerate(METRICS):
        p = page('All individual intervals: ' + ('PSNR' if index == 0 else metric),
                 'Regular UV | * = reused old upright-aware result, identical physical geometry')
        rows = []
        for label in LABELS:
            cells = []
            for family in FAMILIES:
                v = mean(values, label, family, reuse=True)[index]
                cells.append(f'{v:.{3 if index == 0 else 5}f}' +
                             (' *' if family in VERSIONS and label not in PROBLEM else ''))
            rows.append([label_text(label)] + cells)
        table(p, ['Interval', 'Sphere', 'Legacy poles', 'Narrow upright', 'Old upright fit',
                  'Strong floor', 'Compact'], rows, [.16] + [.14] * 6, 8)
        p.paragraph('Legacy poles and Narrow upright are means of three same-seed repeats. All other measured cells are single runs. An asterisk identifies reuse, not a newly measured training. Strong floor and compact use upright poles; no new visible-UV results are substituted here.')
        yield p

    p = page('Previous Visible-UV context',
             'Historical cohort only: the six newest trainings used regular UV')
    rows = [[name] + fmt(aggregate(values, LABELS, family, 'visuv'))
            for family, name in zip(FAMILIES[:4], NAMES[:4])]
    table(p, ['Seven-interval Visible-UV treatment', 'PSNR', 'SSIM', 'LPIPS'], rows,
          [.55, .15, .15, .15], 9)
    p.paragraph('Compact and strong-floor Visible-UV cells are intentionally absent. Geometry fitting for both recipes uses canonical regular mapping. A later Visible-UV training reallocates texture density on the same geometry; it does not refit the geometry.')
    p.paragraph('Earlier upright-aware widening degraded all three metrics on all three problematic intervals in both UV modes. The latest regular results repair that loss, but cannot establish the response under Visible-UV. Fixed-geometry upright tests also change the longitude reference, so they do not isolate pole direction alone.')
    yield p

    p = page('How the new geometries changed',
             'Physical radii in metres: fixed local columns [forward-like, lateral, vertical]')
    rows = []
    for label in PROBLEM:
        old = read_geometry(geometry_path(label, 'sampling_floor_world_y_v2'))
        new = [read_geometry(geometry_path(label, VERSIONS[k])) for k in ['strong_floor', 'compact']]
        triplet = lambda g: ' / '.join(f'{x:.2f}' for x in g['radii'])
        rows.append([label_text(label), triplet(old), triplet(new[0]), triplet(new[1])])
    table(p, ['Interval', 'Old upright-aware', 'Strong floor', 'Compact'], rows,
          [.18, .28, .27, .27], 8.5)
    p.paragraph('The old lateral radii of roughly 51-62 m shrink to roughly 26-39 m. All six new geometries have the forward-like radius larger than the lateral radius. Centers also change during fitting; this is not an isolated one-radius ablation. Physical rotation remains fixed.')
    p.paragraph('Neither recipe imposes a forward-greater-than-lateral rule. Generic axis contractions were available as starting candidates. Earlier ablations reached essentially the same shapes without those extra starts, implicating the changed objectives rather than the multistart mechanism.')
    p.paragraph('Four nonproblematic intervals preserve the original reprojection geometry exactly. Their new PLYs remain useful for inspection, but identical geometry does not imply an independent new training result.')
    yield p


def history_pages():
    p = page('Chronology I: enclosing the visible point union',
             'Earliest PCA construction | geometry derived from points, not trained appearance')
    p.paragraph('The first milestone extracted Waymo LiDAR points. The initial ellipsoid union retained points projected into FRONT by either camera-projection slot and transformed selected frames into a shared NAG coordinate system. PCA determines orientation, not radii equal to covariance standard deviations.')
    p.equation(r'\mu=N^{-1}\sum_n P_n,\quad S=N^{-1}\sum_n(P_n-\mu)(P_n-\mu)^{\mathsf{T}},\quad SR=R\Lambda', size=13)
    p.equation(r'y_n=R^{\mathsf{T}}(P_n-\mu),\quad \ell_k=\min_n y_{nk},\quad u_k=\max_n y_{nk}', size=14)
    p.equation(r'm=(\ell+u)/2,\quad b=(u-\ell)/2,\quad c=\mu+Rm', size=14, height=40)
    p.equation(r's=\max_n\|(y_n-m)/b\|_2,\qquad a=sb', height=42)
    p.paragraph('Purpose: replace the infinite/far plane with a finite anisotropic surface reflecting visible scene extent. Uniform scale s encloses every selected point, but is not a minimum-volume fit. Outliers can enlarge the shape; a sparse vertical point distribution can make it flat. Its point-based center can leave cameras poorly positioned inside.')
    p.paragraph('Earlier rendering usually added a 2% radius margin. Raw saved PLY radii and effective rendered radii must therefore be distinguished. The latest reprojection/sampling-floor experiments instead use zero runtime margin. Original enclosure does not minimize ray depth, cross-frame parallax, or texture sampling.')
    yield p

    p = page('Chronology II: camera-centered robust AR3',
             'Replace extreme enclosure with robust extents and camera-clearance priors')
    p.equation(r'c=\bar C,\quad b_k=Q_{0.995}(|[R^{\mathsf{T}}(P-c)]_k|)', size=14)
    p.equation(r'\widetilde b_k=\max(b_k,\max_j b_j/3,15\,\mathrm{m})', size=14, height=42)
    p.equation(r's=\max(1,Q_{0.995}(\|R^{\mathsf{T}}(P-c)/\widetilde b\|)),\quad a=s\widetilde b', size=13)
    p.equation(r'\rho_i=\|A^{-1}R^{\mathsf{T}}(C_i-c)\|,\quad d_{\mathrm{safe}}=a_{\min}(1-\rho_{\max})', size=14)
    p.paragraph('Retain PCA orientation, move the center to the camera mean, then recompute extents relative to that new center. A 99.5% quantile reduces outlier influence. Raise short radii to enforce at most 3:1 aspect ratio and a 15 m minimum; uniformly expand toward maximum camera normalized radius 0.4 and 15 m conservative clearance.')
    p.paragraph('Reason for change: the original surface was flat and could place cameras near an edge, causing unstable intersections and difficult parameterization. This recipe improves camera positioning and shape conditioning, but does not explicitly align the surface with background depths. Quantiles no longer guarantee enclosure of every point.')
    p.paragraph('Uncapped AR3 could still be excessively large. The later sphere-clipped AR3 takes each effective runtime radius as min(1.02 times the safe fitted radius, matching sphere radius). That final clipping can undo preceding clearance and point-coverage guarantees. A 0.4 target before clipping is not a guarantee afterward.')
    yield p

    p = page('Chronology III: controlled geometry and cross sections',
             'Manual ablations isolate center, scale, and thickness; they are not new optimizers')
    rows = [
        ['Original PCA', '70.81 / 37.26 / 5.96', 'Point-centered, thin'],
        ['Centered robust AR3, uncapped', '89.12 / 29.71 / 29.71', 'Camera-centered, thick'],
        ['Sphere-equivalent ellipsoid', '58.62 / 58.62 / 58.62', 'Equal-radius implementation control'],
        ['Halfway controlled geometry', '73.87 / 44.16 / 44.16', 'Interpolation toward sphere'],
        ['Sphere-clipped centered AR3', '58.62 / 29.71 / 29.71', 'Scale ceiling without equal radii'],
        ['Centered all-point enclosure, clipped', '58.62 / 49.01 / 7.01', 'Retain thin all-point extent'],
        ['Narrow / thin', '58.62 / 29.71 / 7.01', 'Change thickness alone against AR3'],
        ['Wide / thick', '58.62 / 49.01 / 29.71', 'Change lateral width against AR3'],
    ]
    table(p, ['W254 86-137 ablation', 'Effective radii (m)', 'Controlled question'], rows,
          [.42, .24, .34], 8)
    p.paragraph('Historical effective dimensions include runtime margin and any sphere cap. Some ablations supplied radii as NAG overrides rather than changing the original stored PLY. Mesh inspection must use the effective geometry for the intended run.')
    p.paragraph('Motivation: centering, clipping and thickness changed together in the first revision. The controlled runs separated them. Thick clipped AR3 was much better than the original flat enclosure; greater lateral width was not consistently advantageous. These findings were interval-specific, not universal shape constraints.')
    yield p

    p = page('Chronology IV: centered all-point enclosure',
             'Remove quantiles and shape constraints to isolate camera centering plus clipping')
    p.equation(r'c=\bar C,\quad b_k=\max_n|[R^{\mathsf{T}}(P_n-c)]_k|', size=14)
    p.equation(r's=\max_n\|R^{\mathsf{T}}(P_n-c)/b\|,\quad a=sb', size=14)
    p.equation(r'a_k^{\mathrm{render}}=\min(1.02a_k,r_S)', size=15)
    p.paragraph('This branch kept PCA orientation and uniform all-point enclosure, changed only the conceptual center to the camera mean, and applied the later sphere cap. It intentionally omitted robust quantiles, minimum-radius/aspect-ratio limits and camera-clearance expansion.')
    p.paragraph('Why: determine whether the stronger AR3 rules were necessary, rather than attributing every improvement to centering. Recomputing extents at the new center can enlarge some radii; it is not a simple translation. Before clipping the construction encloses all selected points, afterward it need not.')
    p.paragraph('The cross-section ablations then recombined its wide/thin radii with AR3 narrow/thick radii. This tested whether vertical flatness or lateral width explained the loss. These were controlled geometry choices, not fitted multi-frame surface objectives.')
    yield p

    p = page('Chronology V: ray-depth proxy and the v4 basis fix',
             'Move from enclosing a volume to fitting a surface near background observations')
    p.equation(r'd_i=(P-C_i)/\|P-C_i\|,\quad t_L=\|P-C_i\|,\quad Q_i=C_i+t_Ed_i', size=14)
    p.equation(r'L_{\mathrm{depth}}=\operatorname{mean}\rho_{0.2}(\log(t_E/t_L))+L_{\mathrm{safety}}', size=14)
    p.paragraph('The depth-proxy/bgmask family uses FRONT observations outside saved foreground masks and stratified sampling. It fits a free center and positive radii along individual observing rays; points are not measured along parallel forward lines. Mask selection is imperfect static-background filtering, not a perfect semantic/motion classifier.')
    p.paragraph('A fixed road-aligned basis and three forward-offset starts feed bounded L-BFGS-B. Local center bounds are [-80,160], [-25,25], [-15,15] m; radius bounds are 1-350 m. The asymmetric forward bound is a search prior, not a Waymo axis requirement.')
    p.paragraph('The earlier v3/first-version basis assumed the wrong world vertical for NAG exports. V4 uses world Y as vertical and projects the trajectory onto the X-Z ground plane to construct local [forward, lateral, vertical] axes. This changes which physical directions receive bounds and initialization, so it changes optimization, not merely labels.')
    p.paragraph('Why: enclosure often put the surface far beyond actual visible geometry. Fitting ray depth aimed to reduce the background displacement the flow network must absorb. Depth-only loss still does not directly measure cross-view pixel parallax or texture allocation; thin surfaces remained a concern.')
    yield p

    p = page('Chronology VI: vertical-height branches and Moderate-Y',
             'Physical world axes, local axes, and texture poles are separate concepts')
    p.paragraph('Initial proxy inspection exposed ambiguity in X/Y/Z labeling. Manual Y and Z enlargement PLYs tested which physical direction viewers perceived as vertical. The moderate-height training changed physical vertical extent while retaining the proxy center, other radii and orientation. These pilot enlargements were not new depth optimizations.')
    p.equation(r'a_{\mathrm{vertical}}^{\mathrm{Moderate\!\! -\!\! Y}}=\max(a_{\mathrm{vertical}}^{\mathrm{raw\ v4}},a_{\mathrm{lateral}}^{\mathrm{raw\ v4}})', size=14)
    p.paragraph('After correcting the basis, the generic v4 Moderate-Y rule raises the vertical radius to at least the lateral radius. This avoids an interval-specific manually chosen vertical height. It is a post-fit change: the surface is not reoptimized after enlargement.')
    p.paragraph('Motivation: a very flat depth-fit shape might distort texture allocation or worsen behavior near cameras. Moderate-Y improved some aggregate results, but the best choice differed by interval. Raw v4 remained competitive on W254 37-81, showing that thickness alone is not a universal quality criterion.')
    p.paragraph('The local road-forward column can approximately align with world X even though camera-local forward uses another axis. Waymo vehicle axes, NAG world coordinates, ellipsoid local axes and texture coordinates must not be conflated. Physical vertical is world Y in these NAG exports; upright texture poles require an additional texture-frame construction.')
    yield p

    p = page('Chronology VII: fixed geometry, texture orientation and seam',
             'A mapping ablation changes neither ellipsoid PLY nor ray intersections')
    p.equation(r's=A^{-1}R^{\mathsf{T}}(X-c),\quad t=B^{\mathsf{T}}s', size=15, height=40)
    p.equation(r'u=(\operatorname{atan2}(t_x,t_z)+\pi)/(2\pi),\quad v=(\pi/2-\arcsin(t_y))/\pi', size=14)
    p.equation(r'b_y=\frac{A^{-1}R^{\mathsf{T}}e_Y}{\|A^{-1}R^{\mathsf{T}}e_Y\|},\quad B=[b_x\ b_y\ b_z]', size=14)
    p.paragraph('Legacy mapping uses B = identity, so its poles follow the second normalized ellipsoid axis; in the road basis that is lateral. Upright world_y aligns physical poles with world Y. It derives longitude from projected world Z, falling back to world X, and completes an orthonormal normalized-space basis.')
    p.paragraph('Fixed-seam world_y changes both polar and longitude conventions. Automatic seam additionally places the longitude cut in the largest sampled empty angular gap, or the least-occupied neighborhood when coverage is complete. It does not rotate the physical ellipsoid.')
    p.paragraph('Why: test whether apparent shape failures were partly texture chart problems. The early six-run comparison held Moderate-Y geometry fixed. Later 84 same-seed trainings compared legacy and upright poles over all seven intervals, both regular and Visible-UV. Upright local density improved, but aggregate rendering remained worse. These tests reject neither every upright mapping nor every seam choice.')
    yield p

    p = page('Chronology VIII: symmetric reprojection-depth fitting',
             'Directly penalize cross-view pixel mismatch rather than only source-ray depth')
    p.equation(r'e_{ij}=\frac{\|\pi_j(Q_i)-\pi_j(P)\|+\|\pi_i(Q_j)-\pi_i(P)\|}{2}', size=14)
    p.equation(r'D(G)=\operatorname{mean}\rho_8(e_{ij})+0.15\operatorname{mean}\rho_{0.2}(\log(t_E/t_L))', size=14)
    p.paragraph('The original reprojection-depth v1 uses background-selected pairs with target-frame sparse-LiDAR visibility support. Temporal and heading-diverse pairs let a turn contribute multiple directions to the same objective; the fitter does not assume two streets or a fixed number of turns.')
    p.paragraph('Only center and log radii optimize. Rotation comes from raw v4 and remains fixed. Four starts use scaled safety-projected raw-v4 radii and a camera-mean candidate. Bounded L-BFGS-B is followed by safety repair; candidates are selected by the repaired objective, not the infeasible pre-repair optimum.')
    p.paragraph('Compared with depth proxy, this directly penalizes a proxy point projecting differently in another camera. The weak depth anchor stabilizes low-baseline regions. Pixel/depth losses remain independent of texture pole placement. Angular camera and surface-coordinate errors were measured in diagnostics, not added as optimized loss terms.')
    p.paragraph('The first two controlled intervals improved, but extending unchanged settings to all seven produced excessive lateral-major radii on three intervals. Small pixel-loss gains could outweigh large shape changes in weakly constrained directions. This motivated scale and sampling regularization.')
    yield p

    p = page('Chronology IX: safety and radius-scale regularization',
             'Original versus new regularizer labels must remain explicit')
    p.equation(r'\rho_i=\|A^{-1}R^{\mathsf{T}}(C_i-c)\|,\quad \rho_{\max}\leq0.65', size=14, height=40)
    p.equation(r'a_{\min}(1-\rho_{\max})\geq6\,\mathrm{m},\quad a_{\min}\geq8\,\mathrm{m},\quad a_{\max}/a_{\min}\leq10', size=14)
    p.paragraph('Original reprojection uses smooth safety penalties and radius-only repair. These guards replace the earlier AR3 0.4 normalized-radius target; they are not the same constraints. Local center bounds are [-80,160], [-30,30], [-20,20] m. There is no sphere clipping or automatic Moderate-Y enlargement.')
    p.equation(r'L_{\mathrm{scale}}=\frac{20}{3}\sum_{k=1}^3\rho_{0.2}(\max(0,\log(a_k/A_{\mathrm{ref}})))', size=14)
    p.paragraph('The new_radius_scale branch adds a penalty above A_ref, the largest radius of the safety-projected raw-v4 initializer. It is axis-neutral: it limits excessive global scale, not lateral versus forward ordering, and is a soft prior rather than a hard cap. Original means radius-excess weight zero.')
    p.paragraph('Why: regularize huge weakly constrained axes without picking winners per interval. It reduced lateral expansion and improved the pathological training cohort, but did not consistently retain the best earlier quality. The later sampling-floor recipes disable this radius-excess term to isolate a different mechanism.')
    yield p

    p = page('Chronology X: sampling floor v1',
             'Add a texture-density constraint that geometric pixel error can overlook')
    p.equation(r'\Delta T\simeq J\Delta p,\quad J=\partial T/\partial p,\quad \sigma_{\min}(J)=\min_{\|d\|=1}\|Jd\|', size=14)
    p.equation(r'r(p)=\frac{\sigma_{\min}(J_E(p))}{\sigma_{\min}(J_S(p))}', height=48)
    p.equation(r'F_\tau(G)=\sum_p w_p\rho_{0.2}(\max(0,\log(\tau/r(p))))', size=15)
    p.equation(r'\min_{c,\log a}D(G)+\lambda F_{0.5}(G),\quad \lambda\in\{5,10,20\}', size=14)
    p.paragraph('T is texture position in texels, not normalized UV. The smallest singular value measures the least-resolved local image direction in texels per pixel. Central differences at +/- half a pixel estimate J with seam wrapping. Weights give equal mass to frames and occupied 12-by-8 image cells.')
    p.paragraph('Fit with constrained SLSQP from the original reprojection ellipsoid. Require D <= 1.02 D(original) + 1e-6, camera safety and valid projections. Optimize at most 24,000 supported pairs, 14,000 depth rays and 4,096 sampling rays, then check full geometric validity and export full sampling diagnostics.')
    p.paragraph('Why: a surface can have acceptable parallax score yet allocate texture poorly. The floor penalizes sampling compression relative to a reference sphere. Legacy v1 uses regular legacy mapping and has no exact-original preservation gate, so even nearly adequate originals can drift. The floor is soft, not guaranteed at every pixel.')
    yield p

    p = page('Chronology XI: sampling-floor v2 preservation',
             'Avoid changing already adequate original reprojection geometry')
    p.equation(r'F_{0.5}^{\mathrm{all\ valid\ source\ rays}}(G_0)=0\quad\mathrm{and}\quad\mathrm{safety\ valid}', size=14)
    p.equation(r'\Longrightarrow\quad G_{\mathrm{out}}=G_0\quad\mathrm{exactly};\quad\mathrm{skip\ SLSQP}', size=14)
    p.paragraph('The preserved original is reprojection-depth v1, not the first PCA enclosure, AR3, raw v4 or Moderate-Y. Check all valid cached sampling rays plus full geometric validity/safety, not only the optimization subsample. If the gate fails, run the unchanged floor-v1 objective.')
    p.paragraph('Why: v1 could move geometries with already adequate density, causing losses on otherwise successful intervals. Preservation protects those cases without sequence-specific rules. It guarantees identical center/radii/rotation, not identical stochastic training metrics.')
    p.paragraph('The established legacy-pole benchmark mixes floor-v1 refinements on four intervals with exact-original v2 preservation on W191 40-128, W191 128-198 and W254 86-137. The later repeatability sweep rendered those fixed geometries with both legacy and upright poles.')
    yield p

    p = page('Chronology XII: upright-aware sampling-floor v2',
             'Recalculate density with upright poles during geometry fitting, not only during rendering')
    p.equation(r'L_{\mathrm{upright}}=D(G)+20F_{0.5}^{\mathrm{world\ Y}}(G)', size=15)
    p.paragraph('Start again from original reprojection-depth v1, not the already narrowed legacy-floor shape. Recompute B(R,a) for each proposed center/radius geometry using world-Y poles and fixed world-Z longitude. Apply the same policy to the sphere reference. Keep preservation, safety and the 2% full-data geometric budget.')
    p.paragraph('Why: upright mapping changes local texel density, so a floor fit made for sideways legacy poles may not be suitable. The validation retrained all fourteen regular/Visible-UV jobs, including geometrically unchanged ones. Canonical fitting remains regular UV; no automatic seam or Visible-UV-specific refit is active.')
    p.paragraph('Outcome: the three pathological intervals widened laterally again. The 0.5-floor penalty became nearly zero and small data-loss gains favored the wide candidates. Previously fitted narrow geometries were feasible under the upright floor, but the objective did not prefer them. Their trained quality was worse than fixed-narrow upright in both modes.')
    p.paragraph('This motivated the latest two generic objectives. Learned-flow checks on existing checkpoints showed lower image-pixel flow gradients for the better narrow cohort, while raw UV-flow gradients rose. Weak local error correlations did not justify immediately optimizing either learned-flow proxy.')
    yield p

    p = page('Chronology XIII: compact 0.15',
             'Reward smaller volume without assigning special status to a named axis')
    p.equation(r'V(G)=\frac{4\pi}{3}a_1a_2a_3,\quad \log(V/V_0)=\sum_k\log(a_k/a_{k,0})', size=14)
    p.equation(r'L_{\mathrm{compact}}=D(G)+20F_{0.5}^{\mathrm{world\ Y}}(G)+0.15\log(V/V_0)', size=14)
    p.paragraph('A smaller volume decreases the objective; halving volume contributes approximately -0.104. The constant V0 only anchors the reported value. There is no desired volume, axis-wise target, sphere cap, or forward-greater-than-lateral requirement.')
    p.paragraph('Retain the 0.5 exact-original preservation gate. Adequate originals are returned unchanged even if compactness could reduce them. Deficient originals undergo SLSQP 5/10/20 continuation with the same safety and 2% data budget. Generic feasible axis contractions provide starting candidates, not imposed shape rules.')
    p.paragraph('Why: weakly observed expansion should not be rewarded solely for a tiny geometric-loss improvement. Volume regularization makes that trade-off explicit. In the three problem intervals, favorable contractions mostly reduced the lateral radius. Safety prevents unrestricted shrinkage.')
    p.paragraph('Latest evidence: all metrics improve versus wide upright, but the stronger contraction is not uniformly preferable. Compact wins W254 0-37 and loses to strong floor on W254 139-195. The 0.15 weight was experimentally selected on this cohort, not proven transferable.')
    yield p

    p = page('Chronology XIV: strong floor 0.8',
             'Increase the density target only for originals deficient at the established 0.5 gate')
    p.equation(r'L_{\mathrm{strong}}=D(G)+20F_{0.8}^{\mathrm{world\ Y}}(G)', size=15)
    p.equation(r'F_{0.5}(G_0)=0\quad\mathrm{and}\quad\mathrm{safe}\ \Longrightarrow\ G_{\mathrm{out}}=G_0', size=14)
    p.paragraph('No log-volume term is active. If the original passes the full 0.5 gate, preserve it even if it would violate 0.8. Otherwise optimize the stronger floor. Raising tau from 0.5 to 0.8 penalizes moderately compressed regions that were free under the older objective.')
    p.paragraph('Why: favor useful upright texture allocation instead of explicitly shrinking volume. Contraction follows from the density trade-off. The 0.8 target remains soft; remaining violations can be accepted when eliminating them costs too much data accuracy or breaks safety. Generic starts and final full-data validation are shared with compact.')
    p.paragraph('Latest evidence: all three pathological intervals improve markedly; the mean is better than compact and close to the prior fixed-narrow upright results. Legacy poles still outperform it in all three metrics on all three intervals. Thus this is a geometry repair, not a universal best upright solution.')
    p.paragraph('Both recipes retain fixed rotation and optimize centers as well as radii. Neither directly optimizes angular correspondence, actual learned flow, Visible-UV density, PSNR, SSIM or LPIPS. The other four interval geometries were preserved exactly.')
    yield p

    p = page('Reference sphere, Visible-UV and Huber: exact conventions',
             'Clarifications that prevent misleading interpretations of the recipes')
    p.equation(r'c_S=\bar C,\quad r_S=\max(\max_i\|C_i-\bar C\|,\max_{O\ \mathrm{visible}}\|O-\bar C\|)+\epsilon', size=13)
    p.paragraph('The actual benchmark sphere uses visible foreground plane centers O, not the generic trajectory-plus-cutoff default. Its radius is computed at initialization and is not a learned parameter. It varies with interval; sampling-floor fitting fixes its saved geometry. A completed neural sphere training is not mathematically needed for the reference Jacobian.')
    p.equation(r'T_{\mathrm{vis}}=(W_{\mathrm{vis}}(u-u_0)/\Delta u,\ H_{\mathrm{vis}}(v-v_0)/\Delta v)', size=14)
    p.paragraph('General density diagnostics include Visible-UV tile bounds and actual dimensions. Current floor fitting explicitly rejects Visible-UV: only regular density is optimized, and both training layouts reuse the fitted geometry. The sampling Jacobian is pre-learned-flow geometry, not the final network mapping.')
    p.equation(r'\rho_\delta(e)=e^2/(2\delta)\ \ (|e|\leq\delta),\qquad \rho_\delta(e)=|e|-\delta/2\ \ (|e|>\delta)', size=14)
    p.paragraph('Huber is a robust loss, not a quality metric. Transitions are 8 pixels for reprojection and 0.2 for log depth, log sampling shortfall and the older radius-scale log excess. Tau = 0.5/0.8 and compactness weight 0.15 are empirical hyperparameters. A relative floor is not an absolute one-texel-per-pixel guarantee.')
    yield p


def provenance_pages(sources):
    p = page('Evidence, reproducibility and remaining limitations',
             'New folders, historical provenance, and reproduction commands')
    p.paragraph('New strong-floor source: runs/autowork_upright_optimizer_validation_regular_34c7664f. New compact source: runs/autowork_upright_optimizer_validation_regular_aaa12435. Each contains three completed seed-42 regular-UV jobs, one attempt per job.')
    p.paragraph('Previous narrow upright and legacy-pole sources: runs/2026-09-25_ego_poles_repeatability_part1of4 through part4of4. Old upright-aware sources: runs/2026-09-28_upright_sampling_floor. Sphere paths are recorded in reports/2026-09-28_upright_comparison.sources.json and in the companion provenance JSON, not inferred from folder titles.')
    p.paragraph('The companion sources JSON records every metric-file hash, exact source path, new training manifest, geometry/configuration hashes, allowed configuration differences, and explicit unchanged-geometry reuse. The companion metrics CSV contains the verified per-run values before aggregation.')
    p.paragraph('Historical recipe descriptions are reconstructed from checked-in code and earlier reports, not from a claim that all variants were retrained under identical modern settings. Early plane/PCA/control results are history, not matched latest numerical baselines. Local geometric scores and a visually forward-major mesh cannot replace training validation.')
    p.paragraph('This study tuned recipes on seven intervals from two segments. Independent scenes and independent-seed repeats remain needed for generalization claims. Same-seed repeats quantify nondeterminism only. No new training, fitting, or checkpoint rendering was launched while generating this report.')
    p.paragraph('Rebuild from the repository: /usr/bin/python3 reports/generate_autowork_upright_latest_report.py (NumPy, PyYAML, matplotlib and PyMuPDF required). Default output is D:/CGProjectData/waymo/diagnostics/autowork_upright_latest_report. Existing PDF reports and training artifacts are never modified.')
    p.paragraph(f'Validated source cohort: {len(sources)} completed runs (98 prior ellipsoid jobs, 14 sphere references, six new upright jobs). All require successful exit, complete final frames, finite saved final metrics and epoch 80.')
    yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/diagnostics/autowork_upright_latest_report'))
    args = parser.parse_args()
    if not args.output_dir.name.startswith('autowork_'):
        parser.error('new output folder must begin with autowork_')
    values, sources, manifests, reuse = load_data()
    pages = list(results_pages(values)) + list(history_pages()) + list(provenance_pages(sources))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'upright_latest_results_and_recipe_history.pdf'
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata['Title'] = 'Latest upright results and chronological ellipsoid recipes'
        metadata['Subject'] = 'Verified aggregate and individual comparisons; formatted optimization mathematics'
        for number, p in enumerate(pages, 1):
            p.text(36, 30, f'Upright results and recipe history | {number}/{len(pages)} | 2026-10-02',
                   8, color='#444444')
            pdf.savefig(p.figure)
            plt.close(p.figure)
    with fitz.open(output) as document:
        if len(document) != len(pages) or any(not p.get_text().strip() for p in document):
            raise ValueError('Empty or missing PDF page')
        for index, p in enumerate(document):
            for block in p.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        if x0 < 24 or x1 > 768 or y0 < 15 or y1 > 592:
                            raise ValueError(f'Text outside safe page bounds on page {index+1}: {span["text"]}')
        document.set_toc([[1, p.figure.texts[0].get_text(), index + 1]
                          for index, p in enumerate(pages)])
        document.saveIncr()
    provenance = {'date': '2026-10-02', 'generator_sha256': sha(__file__),
                  'aggregation': 'equal interval mean, repeats averaged first; explicitly identified reuse only',
                  'metrics': list(METRICS), 'runs': sources, 'new_manifests': manifests,
                  'unchanged_geometry_reuse': reuse}
    output.with_suffix('.sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with output.with_suffix('.metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['sequence', 'mode', 'family', 'path'] + list(METRICS))
        writer.writeheader()
        for row in sources:
            writer.writerow({**{k: row[k] for k in ['sequence', 'mode', 'family', 'path']}, **row['metrics']})
    print(f'Created {output}: {len(pages)} pages, {len(sources)} verified runs, {len(reuse)} exact reuse checks.')


if __name__ == '__main__':
    main()
