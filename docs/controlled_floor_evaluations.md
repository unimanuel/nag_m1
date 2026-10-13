# Frozen objective and finite-difference evaluation

This is a follow-up to [controlled replay](controlled_floor_replay.md), not a
new fitting recipe. It intercepts the historical v1, historical v2 and
generalized v2 fitters at their first SLSQP call. Normal initialization and v2's
full-ray preservation check execute, but the optimizer never runs. No model,
training, checkpoint, preparation release or historical-file update is involved.

The default interval is W191 0-40. Five identical parameter vectors are supplied
to every version: the reprojection initializer, stored historical floor geometry,
and the three solutions from the preceding replay. Parameters are local center
offsets in meters and log radii; rotation remains fixed. Objective weights 5, 10
and 20, separate data/floor/reprojection/depth terms and all eight historical
inequality constraints are evaluated.

SciPy's existing `approx_derivative` evaluates:

- The actual SLSQP default: forward differences with absolute step
  `sqrt(float64_epsilon)`, approximately `1.490116119e-8`.
- Central differences with absolute steps `1e-6` and `1e-4`, for diagnosis only.

The same parameter perturbations and bounds are checked across versions. Raw
stencil values and all 15-by-6 Jacobians are saved, allowing objective gradients
and constraint derivatives to be inspected separately. The larger central steps
are not assumed to be ground truth, and do not change the production optimizer.

Two fresh-process repeats are performed for each of three versions and two
layouts: untouched NPZ-loaded arrays and C-contiguous copies with every
observation array aligned to 64 bytes. Values, dtypes and strides are unchanged.
All numerical helpers must still match the pinned historical source revisions.
The previous replay's input hashes, runtime/build information, single-core
affinity and single-thread configuration must match. Repeated uncached objective
evaluations are also checked for exact equality.

## Run

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1
DRY_RUN=1 bash run_controlled_floor_evaluations.sh
bash run_controlled_floor_evaluations.sh
```

The 12 default jobs completed in approximately 64 seconds on this machine.
The preceding controlled replay must be complete and unchanged. Default output:

```text
D:\CGProjectData\waymo\diagnostics\autowork_floor_evaluations_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_LAYOUT_VERSION_repeatNN\attempt_001\
    plan.json
    result.json
    console.txt
```

The manifest gives output and parameter names. Per-job results contain full
Jacobian matrices, stencil samples, native initial parameters, the data-budget
limit, input fingerprints, runtime, alignment and timing. Summary comparisons
separate repeatability, between-version offsets and the layout intervention.

An unchanged invocation reuses completed jobs. For independent repetition:

```bash
bash run_controlled_floor_evaluations.sh \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_floor_evaluations_v1_batch2
```

Unowned or conflicting output directories are rejected; incomplete attempts are
preserved. `--dry-run` performs validation without writing files or evaluations.
The standalone source lives outside NAG's preparation signature, so adding this
diagnostic does not invalidate previous preparation caches.

Matching fixed-point evaluations do not guarantee matching entire SLSQP
trajectories, safety repair or recovery of the historical optimum. A failed
release gate is never overridden by this diagnostic.
