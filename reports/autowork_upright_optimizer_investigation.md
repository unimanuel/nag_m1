# Upright ellipsoid optimizer investigation (geometry-only candidates)

Date: 2026-10-01. No new NAG trainings were run. All experimental PLY/JSON
folders and diagnostic output roots created here start with `autowork_`.

## Why the former upright fit widened laterally

On W191 0–40, W254 0–37 and W254 139–195, the established upright-aware
sampling-floor fitter found axis-1 radii of 59.64, 51.19 and 62.46 m,
respectively. Axis 1 is the lateral-oriented column of the saved rotation in
these three intervals. Its 0.5-times-sphere floor penalty was almost zero, so
the final score favored the wider solution's slightly smaller LiDAR
reprojection/depth loss. This was **not** evidence that the narrower geometry
was infeasible: evaluating the already trained legacy-floor ellipsoid through
the upright chart gives zero 0.5-floor loss on all three intervals.

The fixed-geometry legacy-floor ellipsoids with upright poles also trained
better than the old upright-aware wide fits in all three problem intervals
(regular and Visible-UV). This is useful motivation but not proof that the new
untrained geometries will share that quality.

## Checkpoint learned-flow evidence

The checkpoint diagnostic now reproduces the **actual** saved callback layer,
not a standalone background query. All 12 jobs in the recovered W254 86–137 /
139–195 legacy cohort passed that parity gate. A separate 12-job upright cohort
compared already completed upright-pole trainings: fixed narrow legacy-fit
geometry versus the old upright-aware wide fit, each at seed 42 and with the
same YAML. All 12 upright jobs passed composite and background-layer parity.

The table gives fixed-narrow minus old-wide changes. Error and flow percentages
refer to the same full background-mask pixel grid and every frame; PSNR is
the saved final training metric in dB. These are **existing** checkpoints,
not the new candidate geometries.

| Interval | UV mode | Background RGB MAE | Image-pixel flow gradient | Raw UV-flow gradient | PSNR |
|---|---|---:|---:|---:|---:|
| W191 0–40 | regular | −6.8% | −14.1% | +21.7% | +0.351 |
| W191 0–40 | Visible-UV | −8.8% | −14.9% | +22.5% | +0.417 |
| W254 0–37 | regular | −8.3% | −14.4% | +15.8% | +0.302 |
| W254 0–37 | Visible-UV | −11.3% | −14.8% | +9.7% | +0.378 |
| W254 139–195 | regular | −8.6% | −17.4% | +35.3% | +0.507 |
| W254 139–195 | Visible-UV | −8.9% | −14.2% | +39.1% | +0.455 |

SSIM and LPIPS also favor fixed-narrow geometry in all six comparisons.
But the within-frame spatial rank correlation between *changes* in image-pixel
flow gradient and RGB error is only about 0.01–0.03. Thus a smaller flow
gradient is a useful cohort-level observation, not a validated local causal
loss for the new optimizer. Raw UV-flow gradient moves in the opposite
direction, so penalizing it would have selected the worse wide treatment.
Detailed fields, plots, parity records and matched comparisons are in
`D:\CGProjectData\waymo\diagnostics\autowork_upright_learned_flow_v1`.

## Two axis-neutral candidate recipes

Let \(G=(c,R,a,b,d)\) be the ellipsoid with center \(c\), fixed source rotation
\(R\), and physical radii \(a,b,d\). Let \(D(G)\) be the established symmetric
cross-frame pixel-reprojection loss plus its 0.15-weighted log-ray-depth anchor.
For each cached FRONT background sample, let \(\sigma_G\) be the smaller singular
value of the image-pixel-to-upright-texture-texel Jacobian. Let \(\sigma_S\) be
the same quantity on the existing sphere reference, rendered with the same
world-Y chart and regular texture resolution. The balanced sampling-floor loss
for threshold \(\tau\) is

\[
F_\tau(G)=\sum_i w_i\,\operatorname{Huber}_{0.2}
\left(\max\left[0,\log\frac{\tau\sigma_{S,i}}{\sigma_{G,i}}\right]\right).
\]

Candidate **compact 0.15** keeps the original \(\tau=0.5\) floor and minimizes
\[
D(G)+20F_{0.5}(G)+0.15\log\frac{abd}{a_0b_0d_0}.
\]
The last term is log-volume relative to the original geometry; it has no
forward/lateral/vertical label. The optimizer uses the established 5/10/20
floor-weight continuation, plus generic contractions of each axis as optional
starts. An ablation found the same final three pathological radii without the
extra starts: the log-volume term, not the starts, changed the solution.

Candidate **strong floor 0.8** adds no compactness term. If the original passes
the established full-observation 0.5 floor exactly, it is retained byte-for-
byte. Otherwise, the same continuation minimizes \(D(G)+20F_{0.8}(G)\). This
raises the required upright texel density only for geometries known to be
under-sampled under the original criterion. An ablation again found essentially
the same three pathological results without extra starts: the stronger floor,
not the starts, changed the solution.

Both recipes retain the original source rotation, the existing full-data
\(D(G)\le1.02D(G_0)+10^{-6}\) budget, and the original camera/radius safety
rules. They add **no** constraint requiring one named axis to be longer than
another and no sequence-ID-specific rule. The pre-existing safety rules still
include a 10:1 maximum aspect ratio and minimum camera clearance; neither new
candidate approaches that aspect-ratio limit.

## Geometry-only results

Radii below are in meters in the saved rotation-column order. For the three
problem intervals, axis 0 is the road/view-aligned principal direction and
axis 1 is the former lateral major axis. All three new candidates have axis
0 longer than axis 1. Four other intervals are **exactly unchanged** from the
original reprojection fit under both recipes.

| Interval | Old upright fit (axis 0, axis 1, axis 2) | Compact 0.15 | Strong floor 0.8 |
|---|---:|---:|---:|
| W191 0–40 | 42.55, 59.64, 17.14 | 43.79, 34.43, 17.14 | 43.44, 37.99, 17.14 |
| W254 0–37 | 35.38, 51.19, 17.14 | 34.48, 26.16, 17.14 | 34.94, 32.32, 17.14 |
| W254 139–195 | 41.52, 62.46, 17.14 | 41.07, 36.84, 17.14 | 41.49, 38.57, 17.14 |

All six new geometries pass the original full-observation camera safety and
2% reprojection/depth budget, and have zero measured 0.5-floor penalty. The
largest full-data loss increase is 1.47% (compact on W254 0–37); the strong-
floor candidate's largest is 0.94%. At the stronger 0.8 threshold, the new
floor penalties are small but generally not identically zero. Pixel
reprojection p90 changes by less than 0.4 px relative to the old upright fit
on each problem interval; smaller is not uniformly better. Required geometric
UV displacement often rises for a narrower surface, so that proxy must not be
mistaken for measured learned-flow difficulty.

The complete matched-observation rows, source hashes, sampling coverage and
camera clearances are in
`D:\CGProjectData\waymo\diagnostics\autowork_upright_optimizer_diagnostics_v2`.
Every sequence has an inspectable `ellipsoid.ply` inside the corresponding
`D:\CGProjectData\waymo\ellipsoids\segment-...\autowork_frames_...` folder.

## Limits and next validation

These are geometry-screening results, not reconstruction-quality results.
The 0.15 volume weight and 0.8 density target were selected while examining
these seven intervals, so a new independent cohort would eventually be needed
to claim generalization. Existing learned-flow checkpoints can help explain
old wide versus fixed-narrow upright treatments, but there is no learned flow
for either **new** candidate until it is trained. Do not infer PSNR, SSIM or
LPIPS from a zero sampling-floor penalty or a forward-major PLY alone.

The user-controlled training launcher is `run_autowork_upright_optimizer_validation.sh`.
It defaults to the three problem intervals, both recipes, and regular plus
Visible-UV (12 trainings); `--modes regular` restricts it to six. Its dry run
prints every command without creating a run directory. The other four
intervals need no new training for these recipes because their geometry is
exactly the existing original upright-pole treatment.
