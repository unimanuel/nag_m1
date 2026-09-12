# Constrained reprojection/depth ellipsoid experiment

This is the first fitting experiment that directly uses the completed
cross-frame geometry diagnostics. It tests only W254 37--81 and W254 86--137:
the first favored the thin raw v4 depth proxy, while the second showed that a
large reduction in fixed-geometry reprojection error did not by itself improve
the trained result.

The experiment does not fit a separate model for a turn, a street direction,
or a fixed number of road branches. Every selected source/target camera pair
uses the same one ellipsoid. The prepared pairs include temporal neighbors,
distant frames, and heading-diverse targets, with the existing target-frame
sparse-depth visibility check. Thus a turn simply contributes observations
from both directions to the same objective.

## Fit objective

For LiDAR point \(P\), source camera \(i\), target camera \(j\), and candidate
ellipsoid \(E\), let \(Q_i\) and \(Q_j\) be the positive intersections of the
source and target camera rays through \(P\) with \(E\). The primary residual is
the symmetric pixel reprojection discrepancy:

\[
r_{\mathrm{px}}(P,i,j)=\tfrac12\left(
\left\|\pi_j(Q_i)-\pi_j(P)\right\|_2+
\left\|\pi_i(Q_j)-\pi_i(P)\right\|_2\right).
\]

The data loss is

\[
L=\operatorname{Huber}_{8\,\mathrm{px}}(r_{\mathrm{px}})
+0.15\operatorname{Huber}_{0.20}\!\left(
\log\frac{\lVert Q_i-C_i\rVert}{\lVert P-C_i\rVert}
\right).
\]

Only source pixels outside the saved NAG foreground masks and pairs with
target sparse-depth support are used. The first term is dominant. The weaker
log-depth term prevents poorly observed directions from drifting freely.

The ellipsoid orientation remains the raw v4 road-aligned orientation; only
its center and three radii are optimized. Texture orientation is explicitly
held to `legacy` in both training modes. This makes the regular/Visible-UV
comparison a texture-allocation ablation rather than a geometry-orientation
experiment.

## Safety constraints

The first combined trial intentionally does not use sphere clipping or
automatic Moderate-Y enlargement. It does impose conservative geometry guards
chosen to exclude raw v4's harmful regime while retaining the scale of the
successful Moderate-Y cases:

| Guard | Value |
|---|---:|
| Maximum camera normalized radius | 0.65 |
| Minimum conservative camera clearance | 6 m |
| Minimum local radius | 8 m |
| Maximum local axis ratio | 10:1 |

Smooth penalties guide the optimizer; a final radius-only safety projection
makes every guard exact. That projection never changes the center or
orientation and never caps a radius by the matching sphere.

## Run

Activate `nag1`, then first validate the commands without writes:

```bash
cd /mnt/c/Coding/CGProject/nag1
conda activate nag1
DRY_RUN=1 bash prepare_W254_reprojection_depth_ellipsoids.sh
```

Generate the two inspection folders:

```bash
bash prepare_W254_reprojection_depth_ellipsoids.sh
```

Before training, inspect the generated PLY and fit diagnostics:

```text
D:\CGProjectData\waymo\ellipsoids\segment-2547899409721197155\
  frames_037_081_reprojection_depth_v1\
  frames_086_137_reprojection_depth_v1\
```

Each folder contains `ellipsoid.ply`, the copied point/camera PLYs,
`ellipsoid.json`, and `reprojection_fit_diagnostics.json`. The latter compares
the safety-projected raw-v4 initialization to the final combined fit. It is a
fixed-geometry diagnostic, not a prediction of final NAG metrics.

If the geometry is acceptable, launch the four controlled training jobs:

```bash
DRY_RUN=1 bash run_W254_reprojection_depth_benchmarks.sh
bash run_W254_reprojection_depth_benchmarks.sh
```

This produces, for each interval, regular and Visible-UV runs named
`ellipsoid_reprojection_depth_v1` and
`ellipsoid_reprojection_depth_v1_visuv`.

After they finish, put the date-named output directory into
`PARALLAX_EXTRA_RUN_ROOTS` and rerun the usual four diagnostic stages. The
inventory recognizes these names as `reprojection_v1`, so their fixed geometry
and rendered spatial errors will be compared against the matching spheres.

## Held-out ego validation

After the initial controlled trial, the same v1 fit can be tested without
changing its objective on the five remaining large-ego intervals: W191 0--40,
40--128, and 128--198; plus W254 0--37 and 139--195.  These are a
generalization benchmark, not a new geometry variant: every interval retains
its raw-v4 road-aligned orientation and uses the exact same 24,000-pair / 14,000
depth-ray budget and safety constraints above.

Prepare and inspect the five new geometry folders first:

```bash
DRY_RUN=1 bash prepare_remaining_ego_reprojection_depth_ellipsoids.sh
bash prepare_remaining_ego_reprojection_depth_ellipsoids.sh
```

Then run the ten held-out training jobs—regular and Visible-UV for every
interval:

```bash
DRY_RUN=1 bash run_remaining_ego_reprojection_depth_benchmarks.sh
bash run_remaining_ego_reprojection_depth_benchmarks.sh
```

The runner refuses to train if any source YAML or generated v1 JSON is absent,
has an unexpected interval, or violates the saved conservative camera-clearance
guard.  It does not overwrite existing result directories; normal NAG
date-stamped output behavior still applies.
