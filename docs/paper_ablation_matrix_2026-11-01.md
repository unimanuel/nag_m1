# Proposed paper ablation matrix — 1 November 2026

Status: **plan, not executed**. The 465-run main benchmark is complete for
7 ego and 24 final intervals. W324 96–161 remains the declared resource/runtime
exclusion. Keep the main recipe and its results frozen; no new optimizer search
or upright-fitting sweep is part of this plan.

## Already covered; do not train again

The full main benchmark provides plane, sphere regular, sphere Visible-UV,
ellipsoid regular and ellipsoid Visible-UV, with three fresh seed-42 repeats
per interval. This already answers the background-treatment and Visible-UV
allocation questions. The repeats measure same-seed nondeterminism, not three
independent initialization seeds. Native foreground, losses, sampling and
training settings are matched within each interval, not identical across all
ego and final intervals.

Pole-orientation comparisons and many earlier geometry experiments also exist.
Use eligible ones in the supplement after checking configurations, runtime
geometry, calibration and repeat provenance. They are not automatically
controlled ablations of the current final recipe. Do not rerun supersampling,
upright-aware fitting, clipping, axis constraints or resolution changes unless
the paper makes a specific claim that needs that experiment.

## Core experiment: isolate reprojection and sampling floor

Use **regular mapping only**, so the geometry objective changes without also
changing Visible-UV allocation. Three recipes form the loss ladder:

| ID | Recipe | Reprojection weight | Depth weight | Sampling floor |
| --- | --- | ---: | ---: | --- |
| D0 | Matched depth-only optimization | 0 | 0.15 | Off |
| D1 | Original reprojection/depth optimization | 1 | 0.15 | Off |
| D2 | Completed paper ellipsoid | 1 | 0.15 | Existing 5 → 10 → 20 continuation |

D0 must use the **same optimizer, cached observations, raw-v4 initializer,
fixed rotation, multistarts, parameter bounds and safety settings as D1**,
changing only the reprojection weight. It is not the old unmodified thin
raw-v4 mesh: that would conflate objective, optimizer and safety differences.
The current production fitter rejects a zero reprojection weight, so D0 needs
an explicitly isolated/versioned ablation interface and regression checks
before fitting. This reporting change does not implement that interface or
loosen the released preparation/training compatibility gate.

Keep these settings fixed across D0/D1/D2:

- Original masks/visibility selection, poses, calibration and FRONT rays;
  256 rays/frame, eight targets, seed 20260912.
- Same supported observations; maximum 24,000 pair and 14,000 depth samples,
  existing Huber transitions, 200 iterations, original regularizer.
- Maximum camera normalized radius 0.65; conservative clearance 6 m;
  minimum radius 8 m; maximum axis ratio 10. These are common safety settings,
  not a new rule ordering the forward and lateral axes.
- Legacy texture poles, zero runtime margin, no sphere clipping, no
  radius-scale or compactness regularizer, and no automatic axis enlargement.
- D2 retains the frozen 0.5 floor threshold, 4,096 sampling rays, 2% geometric
  budget and exact-zero preservation/fallback behavior. Do not silently fix
  floor failures for selected sequences.
- Same nominal texture budget, per-interval YAML, ten workers, foreground
  model, optimization schedule and epoch 80. Fresh seed-42 repeats 01/02/03;
  no resume from training checkpoints and no best-repeat selection.

Reprojection and floor fitting both change center as well as radii. A failure
to improve a floor penalty is an outcome to report, not permission to choose
another geometry after inspecting reconstruction metrics.

## Predeclare five representative development intervals

| Interval | Role |
| --- | --- |
| W191 0–40 | Historically lateral-major ego failure case |
| W254 37–81 | Turning trajectory and strong depth-proxy case |
| W254 86–137 | Well-studied successful reprojection/control interval |
| W203 60–107 | Nearly stationary camera, weak parallax evidence |
| W975 99–162 | Restored final interval with lateral-major/floor-fallback fit |

This gives three ego and two final intervals, including failures and successes
without selecting only the best-looking cases. These sequences have already
informed development; describe this as a **development ablation**, not an
unseen generalization test. The full 31-interval main comparison is reported
separately and must not be replaced with these five intervals' average.

## Job budget and audit before reuse

There are `5 intervals × 3 recipes × 3 repeats = 45` ellipsoid training cells.
All **15 D2 cells already exist** in the main cohort. Plane and regular-sphere
controls also exist for all five intervals. The upper bound is therefore:

- D0: 15 new trainings.
- D1: at most 15 new trainings, reduced only by audited compatible results.
- D2: no new trainings.
- **At most 30 new trainings before the reuse audit.**

Do not promise a reuse count before auditing the original-reprojection runs.
Many historical recipes have only one repeat or different mappings/settings.
Require the exact original regularizer and objective, intended effective
geometry, interval, source data/calibration, complete training settings and
repeat identity. If treatments produce identical effective geometry and
training configuration, document equivalence and reuse explicitly rather
than inventing a geometric difference or selecting the best run.

Fit/export all proposed geometry first, inspect safety/floor diagnostics and
PLYs without manually repairing named intervals, then launch training in
serial batches with a scheduling budget. Derive hours from the existing
matched per-interval durations; do not assume every training costs the same.
Keep ablation outputs/manifests in new `autowork_` namespaces and preserve the
main cohort and all exploratory artifacts.

## Measurements and reporting

For every interval and treatment, report three-repeat PSNR, SSIM and LPIPS
means; across-repeat sample SD; and the separate inter-frame SD convention.
Compare D0 → D1 (adding reprojection) and D1 → D2 (adding floor), with paired
per-interval deltas, individual rows and equal-interval subset averages.
Also report subgroup rows for the three ego/two final intervals; five intervals
are too few for broad independence or significance claims.

Export center, radii, rotation, camera radius/clearance, data objective and
achieved floor-violation fraction. Where the floor stage returns its input,
say so. This separates a recipe's intended penalty from its actual outcome.

Use matched-support RGB error and learned-flow maps on selected checkpoints
to explain results, not as substitute quality metrics. Dense flow is a
backward texture correction, not optical-flow ground truth; lower magnitude
or gradient is not inherently better. Main reconstruction scores alone do not
prove improved novel-view parallax. Additional parallax evaluation should be
driven by the final paper claims and can use existing checkpoints.

## Optional only after the core experiment

A small threshold sensitivity test (for example 0.25/0.5/0.75) on two
predeclared intervals would address whether the chosen floor threshold is
special. It is not required for the core decomposition, and must not become
a per-sequence hyperparameter search. Audit whether the floor actually
activates before spending training time on it. Independent-seed repeats are
a different robustness question; do not relabel the existing seed-42 repeats.

After the core matrix, final result validation and agreed qualitative
evaluation, freeze the training study. Update the paper tables and supplement;
do not automatically expand to every historical recipe or restart the upright
optimizer investigation.
