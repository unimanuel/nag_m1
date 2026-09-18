# Sampling-floor v1: controlled geometry validation

In the `nag1` WSL environment, from the repository root:

```bash
DRY_RUN=1 bash prepare_sampling_floor_validation.sh
bash prepare_sampling_floor_validation.sh
```

Preparation fits five candidates and stops before training. Inspect the generated
PLYs, then run:

```bash
DRY_RUN=1 bash run_sampling_floor_validation.sh
bash run_sampling_floor_validation.sh
```

The intervals are W191 0-40, W254 0-37, W254 37-81, W254 86-137, and W254 139-195.
Training runs regular and Visible-UV for each: ten jobs. The middle two W254
intervals test preservation of successful fits; the other three test recovery.
Names end in `ellipsoid_reprojection_depth_sampling_floor_v1` and optionally
`_visuv`. Training uses the standard dated NAG run output location.

Outputs live alongside previous ellipsoids:

```
D:\CGProjectData\waymo\ellipsoids\segment-...\frames_NNN_NNN_reprojection_depth_sampling_floor_v1\
```

Each folder contains `ellipsoid.ply`, `ellipsoid.json`, copied front/background
points and camera centers, `sampling_floor_diagnostics.json`, and the existing
reprojection diagnostic JSON/CSV. Preparation prints radii and the world XYZ
unit vector of the largest axis. A lateral-major shape is allowed: the goal is
to remove harmful sampling compression, not force a forward major axis.

All five intervals use the same settings:

- Initialize from the saved original reprojection-v1 geometry, which must match
  the effective geometry of its completed regular run.
- Preserve the original fixed orientation, symmetric reprojection/depth terms,
  sampling seed/counts, bounds and camera safety settings. No sphere radius
  clipping, Moderate-Y enlargement or radius-scale regularizer is added.
- Measure the smaller singular value of the image-to-texture Jacobian relative
  to the saved regular sphere at the same pixels. Both use the same verified
  regular texture resolution and legacy mapping. Visible-UV is used only during
  the paired training run, not to define another geometry.
- Add mean Huber(max(0, log(0.5 / ratio))) with transition 0.2, balanced across
  frames and occupied image cells, on up to 4096 deterministic source rays.
  Penalty weights progress through 5, 10, 20. These are experimental global
  settings, not proven optimum weights or a guarantee of higher rendering quality.
- Use constrained SLSQP for this refinement, keeping feasible iterates and the
  original. Allow at most 2% higher original geometric data loss. After any safety
  repair, verify the budget on all cached supported pairs and all depth rays;
  rank acceptable candidates by the final combined objective. This validation
  uses cached observations, not independent ground-truth rendering quality.
- Export full-source sampling statistics before and after. The floor is a soft
  penalty: the final shape may still have samples below it, especially when the
  geometric budget binds. If no candidate improves the combined objective while
  satisfying the constraints, retain the original and report that explicitly.

There is no additional runtime margin: training uses margin 0 so the inspected
mesh matches the effective training geometry. The original geometry folders
are preserved. Preparation refuses existing output folders to prevent accidental
replacement. Use `--sequences W254_86_137` to select intervals (for example, to
continue remaining fits after interruption); use the same selection with training.
Training aborts on failure, and does not automatically skip completed jobs.
Use `--modes regular` or `--modes visuv` to select just one mapping when resuming.

`PYTHON_BIN` selects Python. `--prepared-root` and `--ellipsoid-root` override
data paths. Existing full-interval `parallax_v1` caches and original-v1 geometry
assets are required; no pixel-error folders or point-cloud regeneration are needed.
