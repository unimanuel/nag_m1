"""Generate verified upright comparison tables and vector-typeset mathematics."""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fitz

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'nag/scripts'))
from ego_repeatability import completed, METRICS, V2
from generate_sampling_floor_v2_typeset import MathPage

LABELS = ['W191_0_40', 'W191_40_128', 'W191_128_198', 'W254_0_37',
          'W254_37_81', 'W254_86_137', 'W254_139_195']
WIDE = {'W191_0_40', 'W254_0_37', 'W254_139_195'}
FAMILIES = ['sphere', 'legacy', 'upright', 'upright_fit']
NAMES = ['Sphere reference', 'Legacy fit / legacy poles', 'Legacy fit / upright poles', 'Upright-aware fit / upright poles']


def table(page, headers, rows, widths=None, size=9):
    height = (len(rows)+1)*23
    ax = page.figure.add_axes([36/792, (page.y-height)/612, 720/792, height/612])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=headers, cellLoc='left', colLoc='left',
                 colWidths=widths, bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    for (row, col), cell in t.get_celld().items():
        cell.set_facecolor('#dce5ef' if row == 0 else ('#f1f4f7' if row % 2 else '#ffffff'))
        cell.set_edgecolor('#b6c2cf')
        cell.set_linewidth(.5)
        cell.get_text().set_color('#080808')
        if row == 0:
            cell.get_text().set_weight('bold')
    page.y -= height + 20
    page.check()


def data():
    values = defaultdict(list)
    provenance = []
    for part in range(1, 5):
        root = REPO / 'runs' / f'2026-09-25_ego_poles_repeatability_part{part}of4'
        for job in json.loads((root / 'manifest.json').read_text())['jobs']:
            good = [(p, completed(p, job['frame_count'])) for p in (root / job['id']).glob('attempt_*')]
            good = [(p, v) for p, v in good if v is not None]
            if len(good) != 1:
                raise ValueError(f'Ambiguous/incomplete repeat: {job["id"]}')
            p, v = good[0]
            family = 'legacy' if job['orientation'] == 'legacy' else 'upright'
            values[job['label'], job['mode'], family].append([v[m] for m in METRICS])
            provenance.append(dict(sequence=job['label'], mode=job['mode'], family=family, path=str(p)))
    root = REPO / 'runs/2026-09-28_upright_sampling_floor'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['jobs']) == 14 and not manifest['reused']
    for job in manifest['jobs']:
        good = [(p, completed(p, job['frame_count'])) for p in (root / job['id']).glob('attempt_*')]
        good = [(p, v) for p, v in good if v is not None]
        assert len(good) == 1
        p, v = good[0]
        values[job['label'], job['mode'], 'upright_fit'].append([v[m] for m in METRICS])
        provenance.append(dict(sequence=job['label'], mode=job['mode'], family='upright_fit', path=str(p)))
    source = Path('/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_v1/metrics.csv')
    with source.open() as f:
        for row in csv.DictReader(f):
            if row['family'] != 'sphere':
                continue
            actual = []
            for metric in METRICS:
                with (Path(row['path']) / 'tracker' / f'train_epoch_Mean{metric}.csv').open() as f:
                    last = list(csv.DictReader(f, delimiter=';'))[-1]
                assert float(last['step']) == 80
                actual.append(float(last['value']))
            assert np.allclose(actual, [float(row[m]) for m in METRICS], atol=1e-8, rtol=0)
            values[row['sequence'], row['mode'], 'sphere'].append(actual)
            provenance.append(dict(sequence=row['sequence'], mode=row['mode'], family='sphere', path=row['path']))
    for label in LABELS:
        for mode in ['regular', 'visuv']:
            for family, n in zip(FAMILIES, [1, 3, 3, 1]):
                assert len(values[label, mode, family]) == n
    return {k: np.asarray(v) for k, v in values.items()}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / '2026-09-28_upright_comparison.pdf')
    args = parser.parse_args()
    values, provenance = data()
    mean = lambda label, mode, family: values[label, mode, family].mean(axis=0)
    pages = []
    p = MathPage('Upright texture experiments: results and mathematics', 'Seven ego intervals | 84 pole repeats + 14 upright-aware trainings | epoch 80')
    p.paragraph('The legacy sampling-floor geometry with legacy texture poles remains the best aggregate treatment. Upright-aware refinement widened the three problematic lateral radii and reduced rendering quality relative to the previous upright treatment.')
    rows = []
    for mode in ['regular', 'visuv']:
        for family, name in zip(FAMILIES, NAMES):
            v = np.mean([mean(label, mode, family) for label in LABELS], axis=0)
            rows.append([name, 'Regular' if mode == 'regular' else 'Visible-UV', f'{v[0]:.4f}', f'{v[1]:.6f}', f'{v[2]:.6f}'])
    table(p, ['Treatment', 'Mode', 'PSNR (dB)', 'SSIM', 'LPIPS'], rows, [.43,.17,.14,.13,.13], 8)
    p.paragraph('Higher PSNR and SSIM; lower LPIPS are better. Aggregate = arithmetic mean of seven interval means, not pooled-pixel PSNR or a frame-weighted average. Legacy and previous upright entries average three seed-42 repeats; new upright-aware and historical sphere entries contain one run per interval/mode.')
    p.paragraph('Sphere baselines were not repeated in this experiment. Same-seed repeats measure nondeterminism rather than the full independent-seed distribution. The intervals come from only two parent Waymo segments.')
    pages.append(p)

    p = MathPage('Treatments and controlled questions', 'Mapping orientation, geometry fitting, and Visible-UV allocation are distinct choices')
    p.paragraph('Legacy fit / legacy poles: the established cohort uses sampling-floor v1 on four intervals and exact-original preservation v2 on W191 40-128, W191 128-198, and W254 86-137. Legacy normalized local Y supplies the texture poles, which are lateral in the road-aligned physical basis.')
    p.paragraph('Legacy fit / upright poles: all physical geometry is held fixed. The world-Y poles and fixed world-Z longitude convention replace the legacy mapping. Each regular and Visible-UV treatment was run three times with seed 42. This tests the implemented mapping convention, not pole direction independently of longitude reference.')
    p.paragraph('Upright-aware fit / upright poles: start from the original reprojection-depth v1 ellipsoid before legacy floor refinement. Recompute the upright basis for every candidate geometry and for the sphere reference; preserve feasible zero-floor originals. Fit only regular mapping, then train regular and Visible-UV with the final geometry. All 14 jobs were retrained.')
    p.paragraph('Hypothesis A: upright texture coordinates may improve sampling or learning. Hypothesis B: changing orientation after legacy-aware fitting may cause a mismatch that upright-aware refinement can repair. Diagnostics and trainings address both questions; they do not optimize the texture orientation freely.')
    p.paragraph('Training uses 80 epochs, seed 42, deterministic=False, and zero runtime ellipsoid margin. A storage-only change disables intermediate checkpoints; each new run saves final-80.ckpt. Final metrics and render coverage were verified for all 98 ellipsoid jobs.')
    pages.append(p)

    for index, metric in enumerate(METRICS):
        p = MathPage('Individual interval comparisons: ' + ('PSNR' if index == 0 else metric), 'Each cell is mean +/- sample SD for repeats; single runs have no SD')
        rows = []
        for label in LABELS:
            for mode in ['regular', 'visuv']:
                row = [label.replace('_', ' ', 1).replace('_', '-'), 'Reg' if mode=='regular' else 'UV']
                for family in FAMILIES:
                    a = values[label, mode, family][:, index]
                    precision = 3 if index == 0 else 5
                    row.append(f'{a.mean():.{precision}f}' + (f' +/- {a.std(ddof=1):.{precision}f}' if len(a)>1 else ''))
                rows.append(row)
        table(p, ['Interval', 'Mode', 'Sphere', 'Legacy / legacy', 'Legacy / upright', 'Upright-aware'], rows, [.16,.06,.12,.24,.24,.18], 8)
        p.paragraph('Reg = regular UV; UV = Visible-UV. PSNR/SSIM are better upward; LPIPS is better downward. Comparisons use saved final tracker metrics. The new recipe is one stochastic training per interval/mode and should not be interpreted as a repeated significance test.')
        pages.append(p)

    p = MathPage('Geometry changes and the lateral-axis cases', 'Radii in metres; local axes are forward, lateral, vertical')
    rows=[]
    for label in LABELS:
        seg = '1918764220984209654' if label.startswith('W191') else '2547899409721197155'
        a,b=map(int,label.split('_')[1:])
        base=Path('/mnt/d/CGProjectData/waymo/ellipsoids')/f'segment-{seg}'
        prefix=f'frames_{a:03d}_{b:03d}_reprojection_depth'
        def read(suffix): return json.loads((base/(prefix+suffix)/'ellipsoid.json').read_text())
        original=read('_v1');legacy=read('_sampling_floor_v2' if label in V2 else '_sampling_floor_v1');new=read('_sampling_floor_world_y_v2')
        text=lambda doc: ' / '.join(f'{r:.2f}' for r in doc['fit']['radii'])
        rows.append([label.replace('_',' ',1).replace('_','-'),text(original),text(legacy),text(new)])
    table(p,['Interval','Original reprojection','Legacy floor','Upright-aware floor'],rows,[.19,.27,.27,.27],8)
    p.paragraph('W191 0-40, W254 0-37 and W254 139-195 change substantially. Their new lateral radii remain larger than their forward radii. All six trainings on these intervals are worse than every corresponding previous upright repeat on all three metrics.')
    for mode in ['regular','visuv']:
        delta=np.mean([mean(l,mode,'upright_fit')-mean(l,mode,'upright') for l in WIDE],axis=0)
        p.paragraph(f'Three lateral cases, {mode}: new minus prior upright = {delta[0]:+.4f} dB PSNR, {delta[1]:+.6f} SSIM, {delta[2]:+.6f} LPIPS.')
    p.paragraph('Three other intervals preserve exactly the previous physical geometry; W254 37-81 returns to the original with a very small change. Differences on identical geometry reflect training variation. Lateral-major shape alone is not a general proof of poor quality; center and radii changed together.')
    pages.append(p)

    p = MathPage('Sampling diagnostics and interpretation', '100,608 shared LiDAR-supported source pixels; all 28 mappings measured')
    source=Path('/mnt/d/CGProjectData/waymo/diagnostics/poles_sampling_v1/metrics.csv')
    with source.open() as f: sampling=list(csv.DictReader(f))
    rows=[]
    for label in LABELS:
        def row(mode,orientation):return next(r for r in sampling if (r['sequence'],r['mode'],r['orientation'])==(label,mode,orientation))
        l,u=row('regular','legacy'),row('regular','world_y');lv,uv=row('visuv','legacy'),row('visuv','world_y')
        change=lambda a,b:100*(float(b['texels_per_pixel_min_balanced_mean'])/float(a['texels_per_pixel_min_balanced_mean'])-1)
        rows.append([label.replace('_',' ',1).replace('_','-'),f'{100*float(l["floor_below_half_fraction"]):.2f}%',f'{100*float(u["floor_below_half_fraction"]):.2f}%',f'{change(l,u):+.1f}%',f'{change(lv,uv):+.1f}%'])
    table(p,['Interval','Legacy floor violations','Upright violations','Density gain Reg','Density gain UV'],rows,[.18,.24,.20,.19,.19],8)
    p.paragraph('These diagnostics use the previous fixed geometries. Upright increases minimum-direction density, reduces anisotropy, and improves the lower density tail in every interval/mode. All sampled pixels have valid coverage; none are within 5 degrees of the seam or 10 degrees of a pole. Visible-UV texture counts differ by less than 0.06%.')
    p.paragraph('The original large lateral geometries still violated the upright floor before refinement. The new fits reduced those penalties but stopped with wider radii than legacy refinement. Adequate local sampling does not guarantee easier learned flow or better reconstruction. These samples exclude sky and other regions lacking LiDAR support.')
    p.paragraph('Recommended reference remains the legacy recipe. Retain upright as an ablation. A future geometric diagnostic can examine cross-frame texture displacement and its spatial variation with mapping held fixed; matched sphere and plane repeats remain needed for final benchmark claims.')
    pages.append(p)

    p=MathPage('Mathematics I: physical shape and upright texture basis','Upright poles change the coordinate chart; physical R stays fixed in these fits')
    p.equation(r'X=c+RA s,\quad A=\operatorname{diag}(a_1,a_2,a_3),\quad \|s\|_2=1,\quad s=A^{-1}R^{\mathsf{T}}(X-c)',size=13)
    p.equation(r'u=\frac{\operatorname{atan2}(t_x,t_z)+\pi}{2\pi},\qquad v=\frac{\pi/2-\arcsin(t_y)}{\pi},\qquad t=B^{\mathsf{T}}s',size=14)
    p.paragraph('Legacy B is the identity: its poles lie along the second physical local axis. Upright B is constructed to align physical poles with world Y. Dividing by radii is necessary because the chart is defined on normalized ellipsoid coordinates.')
    p.equation(r'b_y=\frac{A^{-1}R^{\mathsf{T}}e_Y}{\|A^{-1}R^{\mathsf{T}}e_Y\|},\quad f=A^{-1}R^{\mathsf{T}}e_Z',size=14)
    p.equation(r'b_z=\frac{f-(f^{\mathsf{T}}b_y)b_y}{\|f-(f^{\mathsf{T}}b_y)b_y\|},\quad b_x=b_y\times b_z,\quad B=[b_x\ b_y\ b_z]',size=14)
    p.paragraph('World X supplies the fallback if the longitude reference degenerates. The physical pole direction satisfies RA b_y parallel to e_Y. Upright-aware refinement recalculates B(R,A) after every candidate change; it does not optimize B as a free rotation or turn the physical ellipsoid upright.')
    p.equation(r'\widetilde u=\frac{\operatorname{mod}(u-u_0,1)}{\Delta u},\qquad \widetilde v=\frac{v-v_0}{\Delta v}',size=14)
    p.paragraph('Visible-UV maps the selected longitude/latitude rectangle to the allocated tile. Changing B can change both the rectangle and grid shape. The geometry PLY does not encode the texture chart and remains identical when only poles change.')
    pages.append(p)

    p=MathPage('Mathematics II: density floor and sphere reference','Sphere geometry must be available; sphere neural training is not mathematically required')
    p.equation(r'F_i(p)=\mathrm{texel}(\mathrm{UV}(Q_i(p))),\quad J_i(p)=\frac{\partial F_i}{\partial p},\quad \sigma_E=\sigma_{\min}(J_i)',size=14)
    p.equation(r'r=\frac{\sigma_E}{\sigma_S},\quad z=\max(0,\log(0.5/r)),\quad L_{\mathrm{floor}}=\frac{\sum_n w_n H_{0.2}(z_n)}{\sum_n w_n}',size=14)
    p.equation(r'H_\delta(z)=\frac{z^2}{2\delta}\ (|z|\leq\delta),\qquad H_\delta(z)=|z|-\frac{\delta}{2}\ (|z|>\delta)',size=14)
    p.paragraph('This is the normalized Huber convention actually used by the fitter. Half-pixel central differences estimate the image-to-texel Jacobian; minimum singular value measures density in the weakest direction. Weights equalize occupied image cells and frames. The floor is a soft penalty and does not promise zero violations.')
    p.paragraph('The sphere uses its own center/radius and the same regular texture dimensions and orientation policy. In this recipe it supplies only the reference density: it does not directly set ellipsoid radii, center, or clipping. Earlier sphere-clipped AR3 recipes used the sphere differently.')
    p.paragraph('Current preparation reads the sphere geometry from completed sphere-run diagnostics. A trained sphere is a convenience of this implementation. Computing its geometry and reference Jacobians without neural training would be sufficient.')
    p.paragraph('Fixed-geometry diagnostics reported zero upright floor loss on the final legacy geometries. The new experiment instead began before legacy floor refinement, so it could produce different geometry. Zero loss and full safety cause exact-original preservation; otherwise SLSQP applies the continuation.')
    pages.append(p)

    p=MathPage('Mathematics III: geometric loss and constrained refinement','Reprojection/depth terms do not depend on the texture chart B')
    p.equation(r'd_i=\frac{P-C_i}{\|P-C_i\|},\quad Q_i=C_i+t_{E,i}d_i,\quad t_{L,i}=\|P-C_i\|',size=14)
    p.equation(r'e_{ij}=\frac{\|\pi_j(Q_i)-\pi_j(P)\|+\|\pi_i(Q_j)-\pi_i(P)\|}{2}',size=14)
    p.equation(r'L_g=\operatorname{mean}H_8(e_{ij})+0.15\operatorname{mean}H_{0.2}(\log(t_E/t_L))',size=14)
    p.equation(r'\min_{c,\log a}\ L_g+\lambda L_{\mathrm{floor}},\quad \lambda\in\{5,10,20\},\quad L_g\leq1.02L_g^{\mathrm{original}}+10^{-6}',size=13)
    p.equation(r'\rho_i=\|A^{-1}R^{\mathsf{T}}(C_i-c)\|,\quad \max_i\rho_i\leq0.65,\quad \min_k a_k(1-\max_i\rho_i)\geq6\mathrm{m}',size=13)
    p.equation(r'\min_k a_k\geq8\mathrm{m},\qquad \frac{\max_k a_k}{\min_k a_k}\leq10',size=14,height=40)
    p.paragraph('Physical R remains fixed. SLSQP optimizes local center coordinates and log radii, with local center bounds [-80,160], [-30,30], [-20,20] metres and radius bounds 1-350 metres. Full cached-data validation precedes final candidate selection. No sphere clipping or radius-excess penalty is active.')
    p.paragraph('The clearance expression is a conservative bound. No constraint requires the forward axis to be largest. Wider lateral candidates can satisfy both geometric and sampling criteria, but their poorer training results reveal that those criteria do not fully describe reconstruction difficulty.')
    pages.append(p)

    with PdfPages(args.output) as pdf:
        for number,p in enumerate(pages,1):
            p.text(36,30,f'Upright comparison | {number}/{len(pages)} | verified saved epoch-80 results',8,color='#444444')
            pdf.savefig(p.figure)
            plt.close(p.figure)
    doc=fitz.open(args.output)
    assert len(doc)==len(pages)
    assert all(page.get_text().strip() for page in doc)
    doc.close()
    manifest=args.output.with_suffix('.sources.json')
    manifest.write_text(json.dumps(dict(aggregation='equal interval mean; repeat means within each interval',
        metrics=list(METRICS), runs=provenance),indent=2)+'\n')
    print(f'Created {args.output} ({len(pages)} pages); verified 98 ellipsoid and 14 sphere results.')


if __name__=='__main__':
    main()
