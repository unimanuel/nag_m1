# Fixed-geometry texture-flow investigation

This diagnostic compares existing surfaces with their actual **legacy** texture
charts. It does not fit geometry, run training, query learned flow, modify PLYs,
or change texture poles. No checkpoint or `pixel_error` folder is required.

## Run in WSL

Activate the normal `nag1` environment and run from the repository root:

```bash
DRY_RUN=1 bash diagnose_ego_texture_flow.sh
bash diagnose_ego_texture_flow.sh
```

The preflight verifies the full-interval observation caches, camera calibration,
completed epoch-80 metrics, legacy mappings, and the three legacy floor repeats.
It writes nothing. The full run defaults to all seven intervals in W191/W254,
both regular and Visible-UV layouts. It compares sphere, sphere-clipped AR3,
raw v4, Moderate-Y, original reprojection-depth v1, and current legacy floor
(v1/v2 as used in repeatability). It adds the radius-scale candidates where
available. With the current run inventory this is 90 geometry/layout comparisons:
84 core comparisons plus six radius-scale comparisons.

Existing prepared observations live at
`D:\CGProjectData\waymo\diagnostics\parallax_v1`. Nothing is decoded again.
Historical runs are located through the existing benchmark inventory; the floor
cohort is located through the four `2026-09-25_ego_poles_repeatability_part*of4`
manifests. The upright runs are intentionally not included. Floor quality is the
mean and sample SD of three matching legacy repeats, not one chosen repeat.

Optional subset or alternate output:

```bash
SEQUENCES="W191_0_40 W254_0_37 W254_139_195" \
OUTPUT_ROOT=/mnt/d/CGProjectData/waymo/diagnostics/texture_flow_pathological_v1 \
bash diagnose_ego_texture_flow.sh
```

Options can also be passed to the launcher, for example `--modes regular` or
`--plot-pairs 0`. `PYTHON_BIN`, `PREPARED_ROOT`, and `OUTPUT_ROOT` can override
defaults. An existing output directory is never overwritten: choose a fresh
output for another run. A failed run retains completed interval outputs with
`manifest.json` marked `complete: false`; there is no automatic resume.

## Outputs

Default output: `D:\CGProjectData\waymo\diagnostics\texture_flow_v1`.

- `report.html`: readable overview and links to sparse-field figures.
- `metrics.csv`: candidate metrics, coverage/support fractions, matched-subset
  means, distributions, actual texture allocation, and existing quality metrics.
- `by_region.csv`: metrics by baseline, heading change, source frame and image cell.
- `deltas_vs_sphere.csv`: within-interval, within-layout diagnostic and quality deltas.
- `exploratory_correlations.csv`: rank association of sphere-relative diagnostic
  differences with quality differences. These are exploratory, not significance tests.
- `manifest.json`: selected runs, saved geometry/mapping, input-cache provenance,
  code hashes, settings, quality sources, and completion flag.
- Per interval/candidate: `samples.npz`, `summary.json`, and representative
  `field_SOURCE_TARGET.png` plots. Arrays retain signed UV/texel vectors, source
  pixels, camera-pair identities, local-fit support and common-valid masks.

Figures show only observed LiDAR samples. Blank areas are **unobserved**, not
zero error; gray points lack a valid measurement. Colour limits and vector scale
are shared across candidates for the same camera pair and layout. Quiver arrows
represent UV directions; they are not image optical-flow vectors. The examples
prefer large heading changes and large baselines; tables cover all supported pairs.

## Mathematics and interpretation

For one background-mask-selected, target-depth-supported point P, intersect
camera rays Ci→P and Cj→P with the candidate E, giving Qi and Qj. With the saved
surface chart U, measure signed displacement Δu = U(Qj) − U(Qi), taking the
shortest periodic longitude branch. An ideal shared static appearance requires
relative texture correction between those observations. This is not an absolute
canonical-reference flow, and it is not the trained network's actual output.

Within each camera pair, find at most 16 neighboring observations within 80 source
image pixels. Reject neighbors whose depth differs in either camera by more than
1 m + 10% of the smaller depth. These gates reduce mixing of depth layers but
do not prove semantic/static correspondence. At least six samples and a
well-conditioned two-dimensional image design are required.

Fit local affine functions Δu(p) ≈ A p + b and u_source(p) ≈ J p + c, using
Gaussian distance weights and three robust iteratively reweighted least-squares
solves. Report ||A||F, robust RMS non-affine displacement residual, and, when J
is nonsingular and its condition number is below 1000, the local transport
T = I + A J^-1. Singular values of T measure correspondence stretch/compression;
||T−I||F measures departure from identity. None is a measured network Jacobian.

Global UV and texture-texel fits are separate. In the current renderer, Visible-UV
remaps the fixed initial texture lookup, **not** the flow/RGB network's global
UV input. Texel coordinates use the actual saved tile bounds and allocation;
global UV results must not be described as densification-compensated flow.
Resolution units follow the existing diagnostics' W/H budget convention.

Neighbors never mix camera pairs. Longitude charts are locally unwrapped;
ambiguous near-antipodal branches are rejected, and actual seam crossing/proximity
is reported separately. A periodic local estimate does not imply that the neural
network is itself periodic or that seam crossings are harmless.

Weights equalize source frames, then target pairs, then occupied 12×8 image cells.
Percentiles are unweighted. Local fits use common valid observations across
candidates within a layout. Headline `common_*` statistics further use identical
finite pair/coverage/sampling/local-fit subsets as appropriate. Missing coverage
or support never becomes a zero residual; denominators and support fractions
are recorded. Regular and Visible-UV common subsets can differ and must be
interpreted alongside coverage.

The geometry controls remain pixel reprojection, target-camera angle, surface
angle, relative ray-depth error, and minimum/maximum texture sampling density.
Regular mapping also reports the minimum-density ratio to the same-budget sphere
and its below-half fraction (the floor reference). No analogous floor claim is
made from independently cropped Visible-UV layouts.
Do not rank geometry by UV displacement alone: low density can make displacement
small without making a better background. Compare smoothness, non-affine
residual, transport, coverage and density jointly against existing PSNR/SSIM/LPIPS.
This investigation does not yet choose weights or introduce an optimizer loss.
Sparse LiDAR misses sky; masks/visibility do not guarantee static points; cameras
are initial calibration, not learned poses. Only two parent segments are present.
