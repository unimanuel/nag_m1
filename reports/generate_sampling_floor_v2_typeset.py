"""Typeset the report's mathematics; requires matplotlib and PyMuPDF.

Run after generate_sampling_floor_v2_report.py. The source PDF is never changed.
Equations remain vector graphics with embedded fonts, not raster screenshots.
"""
import argparse
import io
from pathlib import Path
import textwrap

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                            'mathtext.fontset': 'dejavusans'})


class MathPage:
    def __init__(self, title, subtitle):
        self.figure = plt.figure(figsize=(11, 8.5))
        self.y = 508
        self.text(36, 572, title, 17, weight='bold', color='#102e57')
        self.text(36, 548, subtitle, 9, color='#424242')
        self.figure.add_artist(plt.Line2D([36/792, 756/792], [537/612]*2,
                                         transform=self.figure.transFigure, color='#bbbbbb', linewidth=.6))

    def text(self, x, y, value, size=10, **kwargs):
        return self.figure.text(x/792, y/612, value, fontsize=size,
                                va='baseline', color=kwargs.pop('color', '#080808'), **kwargs)

    def paragraph(self, value):
        for line in textwrap.wrap(value, width=112):
            self.text(36, self.y, line)
            self.y -= 14
        self.y -= 10
        self.check()

    def equation(self, value, height=48, size=15):
        self.text(55, self.y - 10, '$' + value + '$', size)
        self.y -= height
        self.check()

    def check(self):
        if self.y < 62:
            raise ValueError('Mathematics page overflows the footer')


def math_pages():
    p = MathPage('Mathematical evolution I: PCA enclosure', 'An enclosing construction, not an iterative best-fit surface')
    p.paragraph('Let c be the center, R the orthonormal axis matrix, and a the positive semi-axis radii. P denotes a LiDAR point. All divisions between vectors below are componentwise.')
    p.equation(r'\mathcal{E}=\left\{X:\left\|\operatorname{diag}(a)^{-1}R^{\mathsf{T}}(X-c)\right\|_2=1\right\}')
    p.paragraph('For the FRONT-selected point union, PCA determines the orientation. The final radii are not the square roots of the covariance eigenvalues.')
    p.equation(r'\mu=\frac{1}{N}\sum_n P_n,\qquad S=\frac{1}{N}\sum_n(P_n-\mu)(P_n-\mu)^{\mathsf{T}},\qquad S R=R\Lambda', size=14)
    p.equation(r'y_n=R^{\mathsf{T}}(P_n-\mu),\quad \ell_k=\min_n y_{nk},\quad u_k=\max_n y_{nk}', height=40)
    p.equation(r'm=\frac{\ell+u}{2},\quad b=\frac{u-\ell}{2},\quad c=\mu+Rm,\quad s=\max_n\left\|\frac{y_n-m}{b}\right\|_2,\quad a=sb', size=14)
    p.paragraph('The uniform scale encloses every selected point. It is not a minimum-volume ellipsoid. Extrema and outliers can enlarge the surface, while vertically sparse points can produce a flat shape. No ray depth, parallax or texture-flow error is minimized.')
    p.paragraph('The camera-centered AR3 revision retains PCA orientation but recomputes robust extents around the mean camera center, increases short radii, and then applies safety expansion and sphere clipping. Details follow on the next page.')
    yield p

    p = MathPage('Centered AR3: robust extent and camera safety', 'Centering and clipping are geometric priors, not parallax optimization')
    p.equation(r'\bar C=\frac{1}{M}\sum_i C_i,\quad c=\bar C,\quad b_k=Q_{0.995}\!\left(\left|[R^{\mathsf{T}}(P-c)]_k\right|\right)', size=14)
    p.equation(r'\widetilde b_k=\max\!\left(b_k,\frac{\max_j b_j}{3},a_{\min}\right),\quad s=\max\!\left(1,Q_{0.995}\!\left(\left\|\frac{R^{\mathsf{T}}(P-c)}{\widetilde b}\right\|_2\right)\right)', size=13)
    p.equation(r'a=s\widetilde b,\qquad \rho_i=\left\|\frac{R^{\mathsf{T}}(C_i-c)}{a}\right\|_2,\qquad d_{\mathrm{safe}}=\min_k a_k\,(1-\max_i\rho_i)', size=14)
    p.paragraph('The normalized camera radius is below one inside the ellipsoid. The clearance expression is a conservative lower bound, not the exact shortest camera-to-surface distance. Uniform safety expansion increases radii to satisfy the configured normalized-radius and clearance targets.')
    p.equation(r'a_k^{\mathrm{render}}=\min\!\left(1.02\,a_k^{\mathrm{safe}},\ r_{\mathrm{sphere}}\right)')
    p.paragraph('The subsequent sphere ceiling can undo earlier clearance and point-enclosure guarantees. The 99.5% quantile deliberately reduces sensitivity to extreme points; it does not promise that every point lies inside.')
    p.paragraph('Sphere reference: center at the mean camera position, with radius computed from the largest camera displacement plus the scene cutoff and runtime margin. It does not fit the LiDAR surface.')
    p.equation(r'c_{\mathrm{sphere}}=\bar C,\qquad r_{\mathrm{sphere}}=\left(\max_i\|C_i-\bar C\|_2+d_{\mathrm{cutoff}}\right)(1+\varepsilon)', size=14)
    yield p

    p = MathPage('Mathematical evolution II: depth proxy', 'Fit distance along each observing ray, rather than enclosing all points')
    p.equation(r'd_i=\frac{P-C_i}{\|P-C_i\|_2},\qquad t_L=\|P-C_i\|_2,\qquad Q_i=C_i+t_Ed_i')
    p.paragraph('Each observation has its own ray; the distances are not measured along parallel camera-forward lines. Transform the ray into normalized ellipsoid coordinates and take the positive exit root.')
    p.equation(r'u=\operatorname{diag}(a)^{-1}R^{\mathsf{T}}(C_i-c),\qquad v=\operatorname{diag}(a)^{-1}R^{\mathsf{T}}d_i', size=14, height=42)
    p.equation(r'(v^{\mathsf{T}}v)t^2+2(u^{\mathsf{T}}v)t+(u^{\mathsf{T}}u-1)=0', height=42)
    p.equation(r'L_{\mathrm{depth}}=\frac{1}{N}\sum_n H_{0.20}\!\left(\log\frac{t_{E,n}}{t_{L,n}}\right)+L_{\mathrm{safety}}', height=48)
    p.equation(r'H_\delta(r)=\frac{r^2}{2\delta}\ \ (|r|\leq\delta),\qquad H_\delta(r)=|r|-\frac{\delta}{2}\ \ (|r|>\delta)', size=14)
    p.paragraph('Scaled Huber is quadratic near zero and linear for large errors. Log depth measures relative range mismatch. Iterative bounded L-BFGS-B fits center offsets and log radii from three starts; orientation is fixed. Valid FRONT observations use foreground-mask exclusion and stratified sampling, not perfect semantic or motion labels.')
    p.equation(r'c=\bar C+Rh,\quad a_k=e^{z_k},\qquad a_{\mathrm{vertical}}^{\mathrm{Moderate\!\! -\!\! Y}}=\max(a_{\mathrm{vertical}},a_{\mathrm{lateral}})', size=13, height=40)
    p.paragraph('V4 fixes world Y as vertical and derives forward from the trajectory projected onto X-Z. This changes the optimization basis, not just labels. Depth center bounds are [-80,160], [-25,25], [-15,15] m; radii 1-350 m. Moderate-Y enlarges vertical extent afterward without refitting.')
    yield p

    p = MathPage('Mathematical evolution III: parallax fitting', 'Symmetric cross-camera pixel displacement plus a weak depth anchor')
    p.paragraph('For a supported static-background point P, intersect its rays from cameras i and j to obtain Q_i and Q_j. Compare the real and proxy points after projection into the other camera.')
    p.equation(r'e_{ij}=\frac{\|\pi_j(Q_i)-\pi_j(P)\|_2+\|\pi_i(Q_j)-\pi_i(P)\|_2}{2}', height=50)
    p.equation(r'L_{\mathrm{data}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}H_8(e_{ij})+\frac{0.15}{N}\sum_n H_{0.20}\!\left(\log\frac{t_{E,n}}{t_{L,n}}\right)', size=14)
    p.paragraph('Huber is applied after averaging the directional pixel errors. Cached pairs require visibility support and valid projections; target LiDAR checks reject unsupported or likely occluded pairs. Angular errors are diagnostics, not additional optimization terms.')
    p.paragraph('Original reprojection fits center and log radii with fixed raw-v4 orientation, four starts and L-BFGS-B. Safety penalties are followed by explicit repair; candidates are ranked after repair. The targets are:')
    p.equation(r'\max_i\rho_i\leq0.65,\quad d_{\mathrm{safe}}\geq6\,\mathrm{m},\quad \min_k a_k\geq8\,\mathrm{m},\quad \frac{\max_k a_k}{\min_k a_k}\leq10', size=14)
    p.paragraph('Local center bounds: [-80,160], [-30,30], [-20,20] m; radii 1-350 m. The lateral axis can become largest without rotating the fixed basis. Radius-scale regularization adds the following soft cost; it is not sphere clipping:')
    p.equation(r'L_{\mathrm{scale}}=\frac{20}{3}\sum_{k=1}^3 H_{0.20}\!\left(\max\!\left(0,\log\frac{a_k}{A_{\mathrm{ref}}}\right)\right)', height=48)
    p.paragraph('The reference scale is the largest radius of the safety-projected raw-v4 initializer. This reduced excessive lateral size but underperformed the sampling-floor repair. Its completed training cohort covers only the three pathological intervals.')
    yield p

    p = MathPage('Mathematical evolution IV: sampling floor', 'Penalize texture compression that reprojection and depth losses can miss')
    p.paragraph('Map image pixel x through the candidate surface into texture texels T(x). Estimate the 2-by-2 Jacobian by central differences at plus/minus half a pixel, with longitude seam wrapping. Its smallest singular value measures the least-resolved local image direction.')
    p.equation(r'J(x)=\frac{\partial T(x)}{\partial x},\qquad r(x)=\frac{\sigma_{\min}(J_{\mathrm{ellipsoid}}(x))}{\sigma_{\min}(J_{\mathrm{sphere}}(x))}', height=52)
    p.equation(r'L_{\mathrm{floor}}=\frac{\sum_x w_x H_{0.20}\!\left(\max\!\left(0,\log\frac{0.5}{r(x)}\right)\right)}{\sum_x w_x}', height=60)
    p.paragraph('The same cameras and canonical regular texture mapping are used for sphere and ellipsoid. Weights balance frames and occupied 12-by-8 image cells. A ratio of one half means half the reference sampling in the worst direction, not half the sphere radius. This is not angular correspondence error.')
    p.equation(r'\min_{h,z}\ L_{\mathrm{data}}+\lambda L_{\mathrm{floor}},\qquad \lambda\in\{5,10,20\}', height=42)
    p.equation(r'L_{\mathrm{data}}\leq1.02\,L_{\mathrm{data}}^{\mathrm{original}}+10^{-6}', height=40)
    p.paragraph('Constrained SLSQP starts at the original reprojection fit; the radius-scale prior is disabled. The same safety limits and valid projections are required. After safety repair, the full-observation geometric budget is checked again and feasible candidates compete at final weight 20.')
    p.paragraph('Sampling caps: 4,096 floor rays, 24,000 supported pairs, 14,000 depth rays. Full-source diagnostics reveal remaining violations. The floor is a soft cost, not a guarantee that every ray reaches the threshold. Fitting uses regular legacy mapping even for subsequent Visible-UV training.')
    yield p

    p = MathPage('V2 preservation and texture orientation', 'Separate exact geometry preservation from neural-training repeatability')
    p.paragraph('Before optimizing, v2 checks the floor over every valid cached source ray, not just the optimization subsample. It also checks full-data projection validity and the safety constraints.')
    p.equation(r'L_{\mathrm{floor}}^{\mathrm{all\ cached\ rays}}=0\quad\mathrm{and}\quad\mathrm{safety\ valid}', height=48)
    p.equation(r'\Longrightarrow\quad(c,a,R)_{\mathrm{v2}}=(c,a,R)_{\mathrm{original\ reprojection}}', height=48)
    p.paragraph('When this condition holds, return the exact original center, radii and orientation and skip SLSQP. Otherwise run unchanged sampling-floor-v1 optimization. All three new intervals triggered preservation. This avoids gratuitous geometry drift, but cannot ensure identical training metrics.')
    p.paragraph('The original preserved here is the original reprojection-depth-v1 ellipsoid, not the first PCA enclosure, AR3 or Moderate-Y. W254 86-137 restored its original geometry after the very small sampling-floor-v1 refinement.')
    p.paragraph('Texture orientation is distinct from geometric orientation. Legacy ellipsoid mapping follows the local axes, so poles need not be vertical. World-Y pole mapping and automatic seam placement remain optional and are not enabled in these runs. Sphere poles are vertical in the current setup.')
    p.paragraph('Visible-UV changes texture allocation rather than shape. Rotating texture poles changes the pixel-to-texel Jacobian and therefore changes the sampling-floor objective. Test pole placement on fixed geometry before combining it with a new fitting recipe.')
    p.paragraph('These equations describe proxy objectives, not guaranteed solutions for image quality. The empirical comparisons and unchanged-geometry repeat caveat are retained in the remaining pages. Bounds and camera priors are implementation choices, not requirements imposed by Waymo.')
    yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / '2026-09-19_sampling_floor_v2_comparison.pdf')
    parser.add_argument('--output', type=Path, default=HERE / '2026-09-19_sampling_floor_v2_comparison_typeset_math.pdf')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Use a separate output; preserve the original PDF')
    source = fitz.open(args.source)
    if len(source) != 15 or 'Mathematical evolution I' not in source[2].get_text() or 'How the ellipsoids changed' not in source[6].get_text():
        raise ValueError('Unexpected source report layout')
    buffer = io.BytesIO()
    with PdfPages(buffer) as document:
        for p in math_pages():
            document.savefig(p.figure)
            plt.close(p.figure)
    math = fitz.open(stream=buffer.getvalue(), filetype='pdf')
    result = fitz.open()
    result.insert_pdf(source, from_page=0, to_page=1)
    result.insert_pdf(math)
    result.insert_pdf(source, from_page=6)
    for i, page in enumerate(result):
        page.add_redact_annot(fitz.Rect(0, 565, 792, 612), fill=(1, 1, 1))
        page.apply_redactions()
        page.draw_line((36, 570), (756, 570), color=(.7, .7, .7), width=.5)
        page.insert_text((36, 585), 'Sampling-floor validation | Typeset mathematics edition | Original comparisons retained', fontsize=8)
        page.insert_text((690, 585), f'{i+1} / {len(result)}', fontsize=8)
    result.set_metadata({'title': 'Sampling-floor validation: comparisons and typeset mathematics',
                         'subject': 'Seven ego intervals, treatment recipes and optimization objectives'})
    result.save(args.output, garbage=4, deflate=True)
    print(f'{args.output} ({len(result)} pages)')


if __name__ == '__main__':
    main()
