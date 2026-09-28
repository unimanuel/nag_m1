# Upright-aware sampling-floor validation

This is the direct follow-up to the fixed-geometry pole experiment. It tests
whether a world-Y (upright-pole) texture parameterization benefits when the
sampling-floor fitter itself uses that parameterization. It does not overwrite
the established legacy sampling-floor folders or their PLYs.

## Procedure

For each of the seven ego intervals, preparation starts from the saved original
reprojection-depth v1 ellipsoid, not a legacy sampling-floor result. It retains
the existing symmetric cross-frame reprojection and weak log-depth objective,
all safety constraints, the 2% full-data geometric budget, and no radius-excess
term.

The extra floor uses canonical regular UV only. For every candidate geometry,
the normalized-local-to-texture basis is recomputed with NAG's exact `world_y`
rule: physical world +Y supplies the polar direction and the fixed world-Z
reference supplies longitude. Thus changing radii changes the floor mapping.
The sphere reference receives the same world-Y policy and texture resolution.
Automatic seam placement is not enabled.

The preservation rule checks every valid cached source ray. If the original has
zero upright floor loss and meets the full safety/data checks, it is copied
exactly and SLSQP is skipped. Otherwise, the standard 5/10/20 continuation is
used. Every output gets its own PLY mesh, point/camera PLYs, JSON, and floor
diagnostics. Visible-UV is not fitted; it is evaluated only during training.

## Run

From an activated `nag1` environment:

```bash
DRY_RUN=1 bash prepare_upright_sampling_floor_validation.sh
bash prepare_upright_sampling_floor_validation.sh
```

Inspect the PLYs in:

```text
/mnt/d/CGProjectData/waymo/ellipsoids/segment-.../
  frames_NNN_NNN_reprojection_depth_sampling_floor_world_y_v2/
```

Then run the training preflight and cohort:

```bash
DRY_RUN=1 bash run_upright_sampling_floor_validation.sh
bash run_upright_sampling_floor_validation.sh
```

The launcher uses seed 42, nondeterministic epoch-80 training, zero runtime
geometry margin, explicit `world_y`, and writes fresh outputs to:

```text
/mnt/c/Coding/CGProject/nag1/runs/2026-09-28_upright_sampling_floor
```

It creates a manifest before training and retries incomplete jobs without
overwriting prior attempts. Do not change the geometry JSON, YAML, or relevant
code while the cohort is active.

If a fitted geometry is physically identical to the already completed world-Y
repeatability treatment, the launcher validates and reuses its three prior
seed-42 repeats by default. The manifest records those paths. To train all
fourteen regular/Visible-UV jobs regardless of identity, run:

```bash
bash run_upright_sampling_floor_validation.sh --retrain-identical
```

The default may therefore schedule fewer than fourteen new trainings. It never
substitutes legacy-pole results. For a subset, choose a distinct output root:

```bash
bash prepare_upright_sampling_floor_validation.sh --sequences W254_86_137
bash run_upright_sampling_floor_validation.sh --sequences W254_86_137 --output-root runs/2026-09-28_upright_sampling_floor_W254_86_137
```

This is one fixed-seam upright recipe. A negative result would reject this
recipe, not every possible upright texture parameterization.
