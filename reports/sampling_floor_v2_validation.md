# Sampling-floor v2: preserve zero-penalty originals

This revision retains the original reprojection-depth geometry **exactly** when
its sampling-floor penalty is zero on every valid cached source ray and the
original satisfies all camera-clearance, normalized-radius, minimum-radius,
axis-ratio and full-data validity checks. It skips SLSQP in that case. Otherwise
the existing sampling-floor continuation runs unchanged: threshold 0.5,
weights 5/10/20, and the 2% geometric-loss budget. No new radius prior is added.
The check covers all cached valid source rays, not every possible image pixel.

The default revision cohort is W191 40–128, W191 128–198, and W254 86–137.
The first two complete the seven-interval coverage; the third replaces the
unnecessary zero-penalty refinement observed in v1. Retain the completed v1
results for W191 0–40 and W254 0–37, 37–81, 139–195. In particular W254 37–81
had a small **nonzero** penalty and is not eligible for the no-op rule.

Run from the repository in the training environment:

```bash
DRY_RUN=1 bash prepare_sampling_floor_v2_validation.sh
bash prepare_sampling_floor_v2_validation.sh
# Inspect the new PLYs, then:
DRY_RUN=1 bash run_sampling_floor_v2_validation.sh
bash run_sampling_floor_v2_validation.sh
```

Outputs are separate sibling folders under the existing segment directories:
`frames_NNN_NNN_reprojection_depth_sampling_floor_v2`. Existing v1 folders are
not overwritten. Preparation refuses existing output folders to avoid mixing
fits. Both regular and Visible-UV training are requested by default; exact
zero-floor originals are skipped in favor of the existing original
reprojection-v1 results, explicitly not counted as new runs. Therefore the
default train script may legitimately schedule zero jobs. Inspect its output.

To independently repeat all three intervals even when geometry is unchanged:

```bash
bash run_sampling_floor_v2_validation.sh --retrain-unchanged
```

That schedules six jobs. For only the sensitive W254 Visible-UV repeat:

```bash
bash run_sampling_floor_v2_validation.sh --sequences W254_86_137 --modes visuv --retrain-unchanged
```

Training names contain `sampling_floor_v2`; runtime margin remains zero and
texture orientation remains legacy. Geometry identity does not imply that a
new stochastic training run will produce identical metrics. Training is not
automatically resumed/skipped by completion detection; select only unfinished
sequences/modes when resuming. This revision does not establish a new measured
quality improvement until any changed geometries have been trained.
