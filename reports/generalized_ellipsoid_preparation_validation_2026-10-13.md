# Generalized preparation validation — 2026-10-13

Status: implementation is committed, but numerical release acceptance is
blocked. No full training or download was run. No historical ego inputs,
observation-cache metadata, geometry files, result folders or manifests were
rewritten. The requested tolerances have not been widened.

## Immutable fixtures and executable gate

Before the observation interface was changed, the seven historical ego
intervals were frozen in:

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_v1\ego_fixtures.json
```

The manifest records source configurations, observation caches, calibration and
input snapshots, decoded-cloud content hashes, raw-v4 fits, sphere references,
reprojection fits and the final four-v1/three-v2 geometries. Old metadata is
read-only. New metadata and all replay assets use separate `autowork_` folders.

The completed replay report, including stage paths, per-array shapes, maximum
geometry deviations, objective differences and preservation decisions, is:

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_release_v2\verification.json
```

Both cached-input and native-initialization end-to-end comparisons are required
for all seven intervals. Centers and radii use `rtol=1e-6`, `atol=1e-4 m`;
rotation entries use `atol=1e-6`, with no relative tolerance. Preserved geometry
must match exactly. Any failed or missing comparison keeps the production
preparation/training gate closed.

The legacy floor kernel is the exact file at Git commit `07cbd74`, SHA-256
`5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603`.
This preserves the historical refinement implementation and the generic v2
full-ray exact-zero preservation rule, without later upright/compact options.

## Numerical replay

The first failed interval is W191 0-40. Its cached and end-to-end observations
and reprojection geometry match exactly. End-to-end raw-v4 geometry and the
native sphere reference at historical diagnostic precision also match.
Sampling-floor refinement differs by at most 0.000488994 m in center and
0.000529597 m in radii, with unchanged rotation. Its full-observation weighted
data-objective difference is +2.82496e-7. These small differences still fail the
specified tolerance; they are not waived because their objectives are close.
The combined geometric-plus-floor objective differs by only +1.37078e-9,
and the initial objective and initial floor statistics match exactly.

Separate development probes executing the exact historical v1 source also
showed sub-millimeter deviation from the saved W191 0-40 optimum. This is
evidence that replay sensitivity is not confined to the generalized descriptor
interface. It does not establish a unique numerical cause or authorize a
different acceptance threshold.

All seven intervals completed both layers. The table below applies to both
cached and end-to-end replay: each layer produced the same final deviations.
Every one of the 11 observation arrays matches exactly, including dtype,
ordering, frame IDs, selected source points, intrinsics and visibility classes.
Every reprojection geometry matches exactly. Every regenerated raw-v4 geometry
and native sphere reference at legacy precision also matches exactly.
A separate read-only comparison also confirmed exact canonical mapping metadata
for all seven initialization-only references. The current verifier explicitly
checks that metadata, since an isotropic texture-resolution change could cancel
in a relative sampling-floor ratio and escape a geometry-only check.

| Interval | v2 action | Maximum center difference (m) | Maximum radius difference (m) | Full data-objective difference | Both layers |
| --- | --- | ---: | ---: | ---: | --- |
| W191 0-40 | refine | 0.000488994 | 0.000529597 | +2.82496e-7 | FAIL |
| W191 40-128 | preserve | 0 | 0 | 0 | PASS |
| W191 128-198 | preserve | 0 | 0 | 0 | PASS |
| W254 0-37 | refine | 0 | 0 | 0 | PASS |
| W254 37-81 | refine | 0 | 0 | 0 | PASS |
| W254 86-137 | preserve | 0 | 0 | 0 | PASS |
| W254 139-195 | refine | 0.000123441 | 0.000101238 | +1.35057e-7 | PASS |

Every rotation difference is zero. W254 139-195 passes because the criterion
includes `rtol=1e-6` in addition to the absolute 0.0001 m allowance. The same
three original geometries are preserved exactly, and all four other intervals
undergo refinement. W254 37-81 is not wrongly preserved: its original full-ray
floor loss is small but strictly positive. The checker exits with failure and
records `released: false`.

After completing this replay, final launcher review added reference-availability
inventory and explicit descriptor-derived training paths/calibration, plus
completed-run input validation and a clearer missing-gate error. None of the
observation, native initialization, raw-v4 or numerical fitter sources changed.
Nevertheless, the current default is a new
`autowork_generalized_preparation_release_v3` namespace, not a rewritten v2
certificate. It remains uncertified. Unit tests cover the final launcher changes;
no current-code seven-interval pass is claimed.

## Static-scene smoke

A diagnostic-only, isolated W125 124-149 preparation completed every geometry
stage. Maximum camera baseline was **0.000355155 m**, so weak parallax was
reported without changing the fitter or weights. Its observations contained
41,046 target-depth-supported pairs and 3,140 unknown pairs.

| Quantity | Result |
| --- | --- |
| Final center, NAG world meters | `[3.066121, -10.371636, -11.307897]` |
| Final radii, local-axis meters | `[43.337914, 57.522177, 17.142857]` |
| Maximum camera normalized radius | `0.65` |
| Conservative camera clearance | `6.000000000119 m` |
| Minimum axis radius | `17.142857 m` |
| Maximum axis ratio | `3.355460` |
| Full original geometric data loss | `0.055282717504` |
| Full final geometric data loss | `0.056389371323` |
| Native sphere diagnostic radius | `73.6549 m` |

The final data loss is within the historical 2% budget, including the existing
`1e-6` numerical allowance. Initialization-only export invokes dataset and model
construction, not Trainer/build/train/render-video/checkpoint methods. The
smoke did not authorize static training or alter a historical geometry.

Inspection files, including the blue ellipsoid, red sphere, camera centers and
point-cloud PLYs, are under:

```text
D:\CGProjectData\waymo\diagnostics\autowork_static_preparation_smoke_v1\
  segment-12511696717465549299_frames_124_149_step1\
    sampling_floor_v2\attempt_001\
```

## Automated validation and inventory

All **84** relevant unit/regression tests pass:

- 28 generalized preparation tests: deduplication/full identity/strides,
  conflicting settings, missing inputs, static cameras, native mapping,
  initialization-only call boundaries, reference mismatch, immutable retries,
  source-hash pinning, exact preservation and tiny positive full-ray loss,
  reference availability, descriptor-derived training paths and rejection of
  completed training from another dataset root or calibration.
- 12 existing sampling-floor tests.
- 4 existing reprojection tests.
- 40 paper sphere/plane/final baseline tests.

Bash syntax checks pass for the five requested launchers and their shared
wrapper. Read-only audit discovers all 25 matched final intervals. Four W125
intervals currently have the cloud inputs needed for geometry; the remaining
21 need point-cloud extraction, and four of those also have known TIFF gaps.

| Interval | Missing depth frames |
| --- | --- |
| W203 60-107 | 97-99 |
| W975 0-68 | 6-9, 58-68 |
| W975 69-99 | all selected frames |
| W975 99-162 | 99 |

The scripts neither repair depth nor skip selected frames. Full segment IDs
resolve downloads from the local Waymo training list; a read-only command
preview was tested. Actual download/extraction remains user-run.

## Handoff boundary

The five scripts are present, but `prepare_final_ellipsoids.sh` and
`run_final_ellipsoid_benchmarks.sh` deliberately reject a failed full gate.
Audit and cloud-command preview are safe to run now. Do not bypass the gate or
edit the verification JSON to authorize training. Resolving replay acceptance
requires restoring sufficient numerical reproducibility or an explicit user
decision about the comparison criterion; this implementation changes neither.

See [the usage guide](../docs/generalized_ellipsoid_preparation.md) for commands,
provenance, retries, reference validation and output locations.
