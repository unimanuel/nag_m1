# W191 0-40 objective/Jacobian investigation — 2026-10-13

Twelve CPU-only evaluation jobs completed in about 64 seconds, including process
startup; their measured evaluation time totaled 48.1 seconds. No optimization
iterations or training ran. The published fitting recipe, safety limits,
comparison tolerances, historical files and failed release gate are unchanged.

## Experiment

We evaluated the real closures used by historical v1, historical v2 and the
generalized v2 adapter, stopping each fitter immediately before its first SLSQP
call. This preserves initialization and the preservation-check execution path,
rather than reconstructing an approximately equivalent loss independently.

Every version received five identical six-dimensional parameter vectors: the
reprojection initializer, stored historical refined geometry and the historical
v1, historical v2 and generalized v2 solutions from the preceding replay. Center
offsets are in the fixed ellipsoid basis, and radii are encoded logarithmically.

We compared the objectives at continuation weights 5/10/20, their individual
data/floor components and all eight constraint values and derivatives. We used
SciPy's actual default SLSQP forward-difference step, approximately `1.49e-8`,
and diagnostic central steps `1e-6` and `1e-4`. Every perturbation vector matches
across versions. These alternative steps were not installed in the fitter.

Each version/layout combination ran twice in a fresh process. Layouts were
native NPZ arrays and 64-byte-aligned observation-array copies preserving every
value, dtype and stride. Inputs and recorded runtime match the preceding replay,
including its single-core, single-thread execution controls and numerical builds.

## Results

All within-version/layout repeats were exactly equal for values and Jacobians.
Repeated uncached evaluations also matched exactly. Native input-array alignment
was identical across versions in this new harness, including source points at
48 bytes modulo 64; differences in intermediate allocations remain possible.

Maximum between-version differences across all five vectors and both repeats:

| Comparison/layout | Base evaluation difference | Default forward-Jacobian difference | Central `1e-6` difference | Central `1e-4` difference |
| --- | ---: | ---: | ---: | ---: |
| Native historical v1 / v2 | 0 | 0 | 0 | 0 |
| Native historical / generalized v2 | 1.72930e-17 | 1.19209e-7 | 1.77636e-9 | 8.88178e-12 |
| 64-byte-aligned, every version pair | 0 | 0 | 0 | 0 |

The native base-value offset was solely in the floor loss at the generalized
solution. Combined objective values at the five unperturbed vectors matched;
small differences appeared at finite-difference perturbations. The largest
default-step gradient discrepancy was in the weight-10 combined objective.
Data/reprojection/depth evaluations and all constraint values and derivatives
matched exactly across versions, including both layouts.

For the generalized arm, the alignment intervention removed its small floor
and derivative offsets; the historical arms did not change. With aligned
observation arrays, all three versions agreed exactly at **every recorded base
vector and stencil sample**, for all tested steps. This is stronger evidence of
execution/allocation-sensitive floating-point evaluation than the preceding
correlation between alignment and fitted geometry. It does not isolate a
specific NumPy operation or intermediate allocation as the cause.

## Why this can matter to SLSQP

Forward differences divide a subtraction of nearly equal objective values by
approximately `1.49e-8`. A few last-bit changes in those function values can
therefore yield a noticeably larger change in the estimated derivative. The
cross-version derivative spread becomes much smaller with the larger diagnostic
steps, as expected for roundoff amplification; larger steps also introduce
truncation error and are not automatically superior.

At the initializer, the weight-20 default-forward gradient differs from the
central-`1e-4` estimate by up to `8.57e-5`; the two central estimates differ by
up to `5.85e-7`. At the stored refined geometry, the corresponding differences
are approximately `4.99e-6` and `5.19e-6`. These are sensitivity checks, not
gradient-accuracy proofs. The loss includes sampling calculations and piecewise
robust/floor functions, so no central estimate is presumed exact.

The refined geometries lie on active constraint boundaries: maximum camera
normalized radius is at 0.65, and conservative clearance
is essentially 6 m. Consequently, their nonzero ordinary objective gradients
do not indicate failed convergence. Small trajectory differences near active
constraints can produce different tiny displacements while barely changing the
final objective.

## Conclusion and next validation

No objective or constraint discrepancy suggesting a changed mathematical recipe
was found. The evidence supports roundoff and derivative/execution sensitivity
as a mechanism for the earlier submillimeter geometry differences. It does not
prove that these particular sampled derivative offsets alone explain the full
trajectory differences.

The next focused validation is a complete replay with controlled observation
alignment, keeping the original SLSQP step and recipe unchanged. Compare final
geometries between versions and against the stored optimum under the existing
tolerances. Fixed-point agreement does not establish that this full replay will
pass, so generalized preparation remains unreleased. We have not changed input
alignment in production, increased finite-difference steps, or widened tolerances.

## Evidence and validation

```text
D:\CGProjectData\waymo\diagnostics\autowork_floor_evaluations_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_LAYOUT_VERSION_repeatNN\attempt_001\result.json
```

The input fingerprint is identical to the preceding replay. Six focused unit
tests, shell syntax and read-only frozen-input preflight pass. The new tests
exercise interception without optimization, restoration on failure, aligned
copies, known analytic derivatives and comparison rejection for altered inputs
or parameter stencils. See [the run guide](../docs/controlled_floor_evaluations.md)
for an independent repeat command. No generated diagnostic artifacts are added
to Git.
