"""Rebuild the sampling-floor validation PDF from verified saved results."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'nag'))
from diagnostics.parallax_io import SEQUENCES, diagnostic

spec = importlib.util.spec_from_file_location('pdf_helpers', Path(__file__).with_name('generate_2026-09-18_pathological_vs_nonpathological_report.py'))
pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf)
NAMES = {'sphere': 'Sphere', 'ar3': 'Centered, sphere-clipped AR3',
         'raw_v4': 'Raw v4 depth proxy', 'moderate_y': 'v4 Moderate-Y',
         'reprojection_v1': 'Original reprojection', 'floor': 'Sampling floor (v1 + v2)',
         'reprojection_new_radius_scale_v1': 'Radius-scale reprojection'}
METRICS = ('MPSNR', 'SSIM', 'LPIPS')
PATHOLOGICAL = ['W191_0_40', 'W254_0_37', 'W254_139_195']


class Page:
    def __init__(self, title, subtitle=''):
        self.s = pdf.text(36, 571, title, 18, True, pdf.NAVY)
        self.s += pdf.text(36, 548, subtitle, 9, color=pdf.GREY) + pdf.line(36, 537, 756, 537)
        self.y = 515

    def para(self, text, bold=False):
        if bold:
            self.s += pdf.text(36, self.y, text, 11, True)
            self.y -= 20
        else:
            s, self.y = pdf.wrapped(36, self.y, text, 705, 10, 14)
            self.s += s
            self.y -= 12
        assert self.y > 55, 'Page overflow'

    def table(self, headers, rows, widths, size=8.5, height=23):
        s, self.y = pdf.table(self.y, headers, rows, widths, size, height)
        self.s += s
        self.y -= 22
        assert self.y > 55, 'Table overflow'


def metric_values(path):
    values = []
    for metric in METRICS:
        with (path / 'tracker' / f'train_epoch_Mean{metric}.csv').open(newline='') as f:
            data = list(csv.DictReader(f, delimiter=';'))
        if float(data[-1]['step']) != 80:
            raise ValueError(f'Unexpected final epoch: {path}')
        values.append(float(data[-1]['value']))
    return np.asarray(values)


def triplet(v):
    return f'{v[0]:.3f} / {v[1]:.4f} / {v[2]:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'reports/2026-09-19_sampling_floor_v2_comparison.pdf')
    args = parser.parse_args()
    rows, paths = {}, {}
    source = Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_v1/metrics.csv')
    with source.open(newline='') as f:
        for r in csv.DictReader(f):
            key = (r['sequence'], r['family'], r['mode'])
            values = metric_values(Path(r['path']))
            assert np.allclose(values, [float(r[m]) for m in METRICS], rtol=0, atol=1e-8)
            rows[key], paths[key] = values, Path(r['path'])
    for run in sorted((REPO / 'runs/2026-09-18_reprojection_floor').iterdir()):
        if not run.is_dir():
            continue
        label = re.search(r'W(?:191|254)_\d+_\d+', run.name)[0]
        mode = 'visuv' if run.name.endswith('_visuv') else 'regular'
        version = 'floor2' if 'sampling_floor_v2' in run.name else 'floor1'
        _, start, stop = SEQUENCES[label]
        frames = sorted(int(p.name.split('_')[0]) for p in (run / 'final/complete').glob('*_t.png'))
        assert frames == list(range(stop - start)) and (run / 'exit_000.txt').is_file()
        rows[label, version, mode], paths[label, version, mode] = metric_values(run), run
    reports, geometries = {}, {}
    for label, (segment, start, stop) in SEQUENCES.items():
        version = 'floor2' if (label, 'floor2', 'regular') in rows else 'floor1'
        for mode in ('regular', 'visuv'):
            rows[label, 'floor', mode] = rows[label, version, mode]
        parent = Path('/mnt/d/CGProjectData/waymo/ellipsoids') / f'segment-{segment}'
        prefix = f'frames_{start:03d}_{stop-1:03d}_reprojection_depth'
        folder = parent / (prefix + '_sampling_floor_v' + version[-1])
        reports[label] = json.loads((folder / 'sampling_floor_diagnostics.json').read_text())
        original = json.loads((parent / (prefix + '_v1') / 'ellipsoid.json').read_text())['fit']
        current = json.loads((folder / 'ellipsoid.json').read_text())['fit']
        geometries[label] = original, current
        if version == 'floor2':
            assert reports[label]['zero_floor_preserved']
            for k in ('center', 'radii', 'rotation_columns'):
                assert np.array_equal(original[k], current[k])
            for mode in ('regular', 'visuv'):
                a, b = paths[label, 'reprojection_v1', mode], paths[label, 'floor2', mode]
                ga, gb = [diagnostic(p / 'initial/background_diagnostics/ellipsoid_geometry.txt') for p in (a, b)]
                ga.pop('source_path'); gb.pop('source_path')
                assert ga == gb
                ca, cb = [yaml.safe_load((p / 'training_config.yaml').read_text())['NAGConfig'] for p in (a, b)]
                for k in ('name', 'output_path', 'experiment_datetime', 'background_ellipsoid_path'):
                    ca.pop(k); cb.pop(k)
                assert ca == cb

    def mean(family, mode, labels=SEQUENCES):
        return np.mean([rows[s, family, mode] for s in labels], axis=0)

    pages = []
    p = Page('Sampling-floor ellipsoids: completed validation', '19 September 2026 | Seven ego intervals | PSNR higher, SSIM higher, LPIPS lower')
    p.para('Main result: the sampling-floor recipe has the strongest seven-interval averages across all three metrics among the complete treatments below. It largely repairs the lateral-axis failures, but does not win every interval or metric.')
    p.table(['Treatment', 'Regular: PSNR / SSIM / LPIPS', 'Visible-UV: PSNR / SSIM / LPIPS'],
            [[NAMES[f], triplet(mean(f, 'regular')), triplet(mean(f, 'visuv'))] for f in list(NAMES)[:6]], [224, 248, 248])
    p.para('Against sphere: regular gains +0.145 dB PSNR, +0.0075 SSIM and -0.0106 LPIPS; Visible-UV gains +0.178 dB, +0.0079 and -0.0092. Visible-UV improves all three metrics in six of seven intervals; W254 0-37 remains worse.')
    p.para('Cohort definition and limitations', True)
    p.para('Sampling floor combines four completed v1 intervals with the three new v2 intervals, including the new W254 86-137 repeat even though its regular result is lower. This is not best-run selection. Means give each interval equal weight, not each frame. Metrics are final epoch-80 tracker values; no new held-out evaluation or spatial error analysis was performed.')
    p.para('All six new v2 trainings completed. Their starting geometries are exactly the original reprojection ellipsoids: the preservation rule skipped optimization. Their differences therefore measure repeat-training variation, not a new geometry improvement.')
    pages.append(p)

    p = Page('Treatment recipes and terminology', 'Original PCA ellipsoid and original reprojection ellipsoid are different stages')
    recipes = [
        ('Sphere', 'Mean-camera center; equal radii from maximum camera distance plus scene cutoff, with runtime margin. No LiDAR surface optimization.'),
        ('Original PCA ellipsoid (historical)', 'Union FRONT-projected LiDAR points over the YAML interval in common coordinates. PCA orientation, local extrema, bounding-box center, uniform scale to enclose every selected point. Historical context, not a complete seven-interval metric cohort here.'),
        ('Centered, sphere-clipped AR3', 'PCA orientation; camera-mean center; 99.5% robust extents; expand short axes to enforce 3:1 and minimum-radius constraints; robust scale and camera-safety expansion. Runtime radii are min(1.02 times fitted radius, sphere radius). Final clipping can undo prior camera clearance.'),
        ('Raw v4 depth proxy', 'Free center and three positive radii; fixed road-aligned basis using world Y as vertical. Fit masked static-background LiDAR ray depths, not point enclosure. No automatic Moderate-Y enlargement.'),
        ('v4 Moderate-Y', 'Start from raw v4 and raise physical vertical radius to max(vertical, lateral). Preserve center, orientation and other radii; no second optimization.'),
        ('Original / radius-scale reprojection', 'Optimize center and radii with symmetric cross-frame pixel error plus weak depth anchor and safety. Radius-scale version adds a soft penalty for radii exceeding the safety-projected initializer scale; it is not hard sphere clipping.'),
        ('Sampling floor v1 / v2', 'Refine original reprojection fits with a sphere-relative texture-sampling lower bound and 2% geometric-loss budget. V2 additionally preserves safe zero-penalty original geometries exactly; active-penalty optimization is unchanged.')]
    for title, text in recipes:
        p.para(title + ': ' + text)
    p.para('Regular versus Visible-UV changes texture allocation, not the ellipsoid shape. Current fit/train experiments use legacy texture orientation. Upright poles and automatic seams are separate treatments, not enabled here.')
    pages.append(p)

    p = Page('Mathematical evolution I: enclosure, not surface fitting', 'Notation: c = center, R = orthonormal axes, a = three positive semi-axis radii')
    p.para('Ellipsoid surface: || diag(1/a) R^T (X - c) || = 1. Radius values are semi-axes, not diameters. The normalized camera radius is the same expression evaluated at camera center C; values below 1 place the camera inside.')
    p.para('Original PCA construction', True)
    for text in [
        '1. Compute point mean mu and covariance S = mean[(P - mu)(P - mu)^T]. Eigenvectors of S form R, ordered by decreasing eigenvalue. PCA chooses orientation, not the final radii.',
        '2. In local coordinates y = R^T(P - mu), take componentwise extrema l and u. Set midpoint m = (l + u)/2 and half-extents b = (u - l)/2. The world center is c = mu + R m.',
        '3. Compute s = max over points ||(y - m)/b||. Set a = s b. This uniformly expands the axis-scaled ellipsoid until it contains every selected point. It is not a minimum-volume ellipsoid or a least-distance surface fit.',
        'Consequences: a distant outlier or a long road can enlarge the whole shape. A vertically sparse point cloud yields a thin ellipsoid. Enclosing geometry can be far behind most observed background surfaces; this objective never measures cross-view parallax.'
    ]: p.para(text)
    p.para('Camera-centered AR3 revision', True)
    p.para('Set c = mean(C). Recompute point extents relative to this center: b_k = Q_0.995(|[R^T(P-c)]_k|). Increase small b_k to at least max(b)/3 and the minimum radius, then scale by max(1, Q_0.995(||R^T(P-c)/b||)). Apply camera-safety expansion. The runtime sphere ceiling is a later, separate operation, so it can remove enclosure and clearance guarantees.')
    pages.append(p)

    p = Page('Mathematical evolution II: ray-depth proxy', 'Surface-depth matching replaces an all-points enclosure objective')
    p.para('For a selected LiDAR observation P from camera C_i, use its own unit ray d_i = (P-C_i)/||P-C_i|| and measured range t_L = ||P-C_i||. Intersect C_i + t d_i with the ellipsoid to obtain positive exit distance t_E. Rays are not all parallel to the camera viewing axis.')
    p.para('With u = diag(1/a)R^T(C_i-c), v = diag(1/a)R^T d_i, solve (v.v)t^2 + 2(u.v)t + (u.u - 1) = 0. The positive exit root determines the proxy point Q_i.')
    p.para('Depth objective: L_depth = mean H_0.20(log(t_E / t_L)) + camera-safety penalties.')
    p.para('Implemented scaled Huber: H_delta(r) = r^2/(2 delta) for |r| <= delta; otherwise |r| - delta/2. Small residuals receive quadratic penalties; large residuals grow linearly. Log depth emphasizes relative rather than absolute meter error.')
    p.para('Optimization: c = mean(C) + R h; a_k = exp(z_k). Iterative bounded L-BFGS-B uses three forward-offset initializations. R is fixed, not optimized. In v4 the trajectory is projected onto the X-Z ground plane and world Y is vertical; local axis order is forward, lateral, vertical.')
    p.para('The v4 axis-convention correction is not just relabeling: it changes the fixed basis and which physical directions receive different center bounds and initial offsets. This can change the optimized geometry. Bounds for local h are [-80,160], [-25,25], [-15,15] meters; radii are bounded to 1-350 m. The forward asymmetry is a search prior, not a Waymo requirement.')
    p.para('Selection uses valid FRONT background observations and foreground-mask exclusion. These are useful static-background proxies, not perfect semantic or motion labels. Stratified sampling balances coverage. Moderate-Y subsequently thickens the vertical radius without rerunning the loss.')
    p.para('Limitation: good source-ray depth is only an indirect predictor of cross-view image alignment and texture sampling. A fitted proxy need not contain every LiDAR point.')
    pages.append(p)

    p = Page('Mathematical evolution III: parallax / reprojection', 'The implemented error is symmetric cross-camera pixel displacement, not angular texture movement')
    p.para('For the same static point P in cameras i and j, intersect each viewing ray with the candidate surface: Q_i and Q_j. Project Q_i into camera j and Q_j into camera i. The true point projects to pi_j(P) and pi_i(P).')
    p.para('e_ij = ( ||pi_j(Q_i)-pi_j(P)|| + ||pi_i(Q_j)-pi_i(P)|| ) / 2')
    p.para('L_data = mean H_8(e_ij) + 0.15 mean H_0.20(log(t_E/t_L)). Pixel residuals use image pixels; depth residuals use dimensionless log ratios. Huber is applied after averaging the two directional pixel distances.')
    p.para('Only cached visibility-supported background pairs are fitted. Camera projections require valid positive depth; target LiDAR support rejects unsupported/likely occluded comparisons. Depth anchoring uses distinct source observations. Angular errors are diagnostics, not additional terms in this fitted objective.')
    p.para('Original reprojection: optimize h and log radii using bounded L-BFGS-B from four starts derived from safety-projected raw v4. Keep R fixed. Use safety penalties, then repair candidates and rank by the post-repair objective. Targets: camera normalized radius <= 0.65, conservative clearance >= 6 m, each radius >= 8 m, largest/smallest <= 10. The clearance bound is min(a)*(1-max camera normalized radius), not an exact nearest-surface distance.')
    p.para('Center bounds are [-80,160], [-30,30], [-20,20] m in the fixed local basis. A lateral major axis does not mean R was rotated: the optimizer can make its lateral radius larger than its forward radius.')
    p.para('Radius-scale regularizer', True)
    p.para('Add 20 mean_k H_0.20(max(0, log(a_k/A_ref))), where A_ref is the largest radius of the safety-projected raw-v4 initializer. This penalizes excessive global size without demanding the original axis proportions. It shrank the pathological lateral axes, but was less effective than the sampling floor. Its completed training cohort covers only the three pathological intervals.')
    pages.append(p)

    p = Page('Mathematical evolution IV: texture sampling floor', 'A unified penalty addresses texture compression that reprojection/depth error under-penalizes')
    p.para('Map an image pixel x through its ray-surface intersection into texture texels T(x). Estimate the 2x2 Jacobian J = dT/dx using central differences at +/-0.5 image pixels, wrapping the regular longitude seam. Its smallest singular value sigma_min measures texture texels per image pixel in the most compressed direction.')
    p.para('Use the same camera ray and canonical regular texture mapping on the sphere reference. Define r(x) = sigma_min(ellipsoid) / sigma_min(sphere). A ratio of 0.5 means half the sphere sampling density in the worst local direction; it is not a radius ratio.')
    p.para('L_floor = weighted mean H_0.20(max(0, log(0.5/r(x)))). Weights give equal mass to frames and to occupied 12x8 image cells within each frame. This is a sampling-density constraint, not an angular correspondence loss.')
    p.para('Minimize L_data + lambda L_floor using constrained SLSQP continuation with lambda = 5, 10, 20. Start at original reprojection geometry. No radius-scale prior is active. Require L_data <= 1.02 L_data(original) + 1e-6, safety bounds and valid projections; check the geometric budget again on all supported cached observations after safety repair. Rank feasible candidates under the final weight 20.')
    p.para('The optimization samples at most 4,096 source rays for the floor, 24,000 supported pairs and 14,000 depth rays. Full-source diagnostics expose remaining violations. The floor is a soft cost: it does not guarantee every ray reaches ratio 0.5.')
    p.para('V2 preservation rule', True)
    p.para('Before solving, evaluate the floor on all valid cached source rays. If it is exactly zero and the original passes full-data validity and safety checks, return its center, radii and rotation exactly and skip SLSQP. Otherwise use the unchanged v1 optimization. All three new intervals triggered preservation. This prevents gratuitous geometry drift; it cannot guarantee identical neural-training metrics.')
    p.para('Fitting uses legacy regular mapping even when the eventual training uses Visible-UV. Changing pole orientation would change J and therefore this objective. It must be tested as a separate controlled change.')
    pages.append(p)

    p = Page('How the ellipsoids changed', 'Semi-axis radii in meters, ordered forward / lateral / vertical; not sorted by magnitude')
    lines = []
    for label in SEQUENCES:
        a, b = geometries[label]
        lines.append([label.replace('_', ' '), ' / '.join(f'{x:.2f}' for x in a['radii']),
                      ' / '.join(f'{x:.2f}' for x in b['radii']),
                      f"{np.linalg.norm(np.asarray(a['center'])-b['center']):.3f}",
                      'Exact preservation' if reports[label].get('zero_floor_preserved') else 'Active refinement'])
    p.table(['Interval', 'Original reprojection radii', 'Sampling-floor radii', 'Center shift', 'Action'], lines, [100, 190, 175, 85, 170], 8.5)
    p.para('All these refinements retain the original road-aligned orientation. The three pathological intervals lose their extreme lateral elongation; the largest radius becomes forward again. That is an outcome of sampling quality, not a hard requirement that forward must be largest.')
    p.para('For the three pathological intervals, the fraction of cached source rays below half-sphere sampling falls from 100% to about 5-7%. Full geometric loss rises by only about 0.23-1.03%, within the 2% budget. Recovery is therefore much larger in rendering than the change in the old geometric objective suggests.')
    p.para('Earlier stages: the PCA enclosure often became large and flat; camera-centering, AR3 and clipping made it more sphere-like. Raw v4 allowed a displaced center and thin vertical radius to match observed depth. Moderate-Y thickened only vertical extent. Reprojection safety often increased the short radii but permitted huge lateral growth in three intervals. Sampling-floor refinement specifically reduced that growth.')
    p.para('W254 86-137: original and v2 radii are 36.4956 / 17.1674 / 17.1673 m. Sampling-floor v1 had 36.5152 / 17.1429 / 17.1429 m and an 8.74 cm center shift. V2 restores the exact original; the visual mesh difference is very small.')
    pages.append(p)

    notes = {
        'W191_0_40': 'Sampling floor repairs the original reprojection failure and beats sphere in all metrics. Moderate-Y still has better SSIM and LPIPS; higher PSNR is not an across-the-board visual-quality win.',
        'W191_40_128': 'V2 preserved the original geometry. Both new repeats beat sphere and Moderate-Y in all metrics. Changes versus original reprojection are small and are not caused by a new fit.',
        'W191_128_198': 'V2 preserved the original geometry. Both modes retain large SSIM/LPIPS advantages over sphere. Small changes versus original reprojection are repeat variation.',
        'W254_0_37': 'The clearest remaining weakness: sampling floor greatly improves original reprojection, but sphere still wins all three metrics in both modes. Regular SSIM is almost tied. Inspect spatial error before changing the loss.',
        'W254_37_81': 'Sampling floor has strong PSNR and beats sphere in all metrics. Raw v4 still has better SSIM and LPIPS. This interval had a small nonzero floor penalty, so the v2 no-op rule would not change its optimization.',
        'W254_86_137': 'V2 preserves original reprojection geometry exactly. UV recovers from sampling-floor v1, but the regular repeat drops 0.161 dB relative to original with identical saved geometry/settings. Do not attribute all these differences to UV rounding or shape.',
        'W254_139_195': 'Sampling floor repairs the severe original reprojection failure. UV beats sphere in all metrics; regular has lower PSNR, higher SSIM and nearly tied LPIPS. AR3 UV retains higher PSNR, but sampling floor improves SSIM and LPIPS.'}
    for label in SEQUENCES:
        p = Page(label.replace('_', ' ') + ': individual comparison', 'P = PSNR (dB); S = SSIM; L = LPIPS. Deltas are relative to the same mapping sphere.')
        table = []
        for mode in ('regular', 'visuv'):
            sphere = rows[label, 'sphere', mode]
            for family in NAMES:
                if (label, family, mode) not in rows:
                    continue
                v = rows[label, family, mode]; d = v - sphere
                table.append([NAMES[family], 'UV' if mode == 'visuv' else 'Regular',
                              f'{v[0]:.3f}', f'{v[1]:.4f}', f'{v[2]:.4f}',
                              f'{d[0]:+.3f}', f'{d[1]:+.4f}', f'{d[2]:+.4f}'])
        p.table(['Treatment', 'Mapping', 'P', 'S', 'L', 'Delta P', 'Delta S', 'Delta L'],
                table, [220, 64, 67, 67, 67, 75, 80, 80], 8.4, 22)
        p.para(notes[label])
        p.para('Sampling-floor entry: ' + ('v2 independent repeat, original geometry preserved.' if (label, 'floor2', 'regular') in rows else 'completed v1 active-penalty fit, retained without rerunning.'))
        pages.append(p)

    p = Page('Repeatability, interpretation and next checks', 'All six v2 runs preserve the original geometry; seed 42 does not establish deterministic training')
    table = []
    for label in ['W191_40_128', 'W191_128_198', 'W254_86_137']:
        for mode in ('regular', 'visuv'):
            a, b = rows[label, 'reprojection_v1', mode], rows[label, 'floor2', mode]
            table.append([label.replace('_', ' '), mode, triplet(a), triplet(b), f'{b[0]-a[0]:+.3f}'])
    p.table(['Interval', 'Mapping', 'Original: P / S / L', 'V2 repeat: P / S / L', 'Delta P'], table, [115, 65, 230, 230, 80], 8.5)
    p.para('Across the three pathological intervals, sampling floor gains 0.748 dB regular and 0.941 dB UV versus original reprojection. Across the four non-pathological intervals, it differs by -0.022 dB regular and +0.015 dB UV. The regular difference is dominated by the unchanged-geometry W254 repeat, not a failed preservation rule.')
    p.para('Keep this recipe as the working baseline. Check repeatability on W254 86-137 before interpreting small gains. Compare spatial errors for W254 0-37 against sphere and W191 0-40 against Moderate-Y. Then test world-Y texture poles with geometry held fixed; do not simultaneously retune the fit. No claim of statistical significance or universal superiority is justified by these single runs.')
    p.para('Provenance: new runs are in runs/2026-09-18_reprojection_floor; original reprojection runs in runs/2026-09-12_reprojection_depth_v1. Earlier baseline paths come from waymo/diagnostics/texture_sampling_v1/metrics.csv and are checked against saved tracker values. Per-interval PLY/JSON reports are in waymo/ellipsoids/segment-*/frames_*_reprojection_depth*. The generator checks completion, exact v2 geometry, runtime diagnostics and substantive configuration equality.')
    p.para('Math sources: nag_waymo_processing/ellipsoid.py; nag/reprojection_ellipsoid.py; nag/sampling_floor_ellipsoid.py; nag/texture_sampling_diagnostics.py; derive_ellipsoid_axis_variant.py. Historical PCA construction is explained, but no unsupported seven-interval historical average is invented. Training was not rerun for this report.')
    pages.append(p)
    streams = []
    for index, p in enumerate(pages, 1):
        p.s += pdf.line(36, 42, 756, 42)
        p.s += pdf.text(36, 27, 'Sampling-floor validation | Equal interval weights | Black text on light backgrounds', 8)
        p.s += pdf.text(680, 27, f'{index} / {len(pages)}', 8)
        streams.append(p.s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf.build_pdf(streams))
    print(f'{args.output} ({len(pages)} pages)')


if __name__ == '__main__':
    main()
