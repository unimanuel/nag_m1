# Generalized legacy ellipsoid preparation: approved-policy release

Status: **released**. All seven ego intervals passed fresh cached-input and
end-to-end verification (14/14) under the user-approved numerical-equivalence
policy. The actual production `release_gate` accepted the completed certificate
against current source, recipe, fixture and policy hashes. A fresh stationary
scene preparation and training-launcher dry-run also passed. No training ran.

## Acceptance versus recipe

On 2026-10-13 the user approved increasing the optimized center/radius absolute
comparison tolerance from 0.1 mm to 1 mm, retaining `rtol=1e-6`. Rotation entries
still use `atol=1e-6`, `rtol=0`. Preserved geometries remain exactly equal.

This changes release acceptance only. Observation selection, random seeds,
mask/visibility logic, native sphere initialization, raw-v4 interpreter, loss
weights, optimizer settings, continuation, safety repair, texture orientation,
runtime margin and preservation decisions are unchanged. The legacy v2 kernel
remains byte-identical to source commit `07cbd74`. There is no input-alignment
intervention, compactness/radius-scale prior, sphere clipping, automatic
enlargement or upright fitting.

The certificate records policy `numerical_equivalence_1mm_v1`. It additionally
requires:

- Exact observation field sets, dtypes, ordering, selected points/frame IDs,
  intrinsics and visibility arrays; exact diagnostic-precision sphere geometry
  and canonical mapping metadata.
- Close original/final combined objectives and full-observation component
  losses: `rtol=1e-6`, `atol=1e-6`.
- Exact full-observation count/validity agreement; pixel/angle/log-depth
  distribution summaries with `rtol=1e-6` and absolute allowances `1e-3` pixels,
  `1e-4` degrees and `1e-4` log-depth units.
- Full-ray sampling statistic agreement: `rtol=1e-5`, `atol=1e-6`.
- Unchanged maximum camera normalized radius 0.65, minimum conservative
  clearance 6 m, minimum radius 8 m, maximum axis ratio 10, and 2% full-data-loss
  budget with the existing kernel's floating-point allowances.

Nonfinite geometry or diagnostics, changed counts, safety failures, missing
layers, stale source/recipe/fixture signatures or an old policy cannot release
preparation. No automatic tolerance adjustment or historical-report relabeling
is implemented.

## Seven-interval result

The two verification layers produced identical differences in each row. Errors
are maximum absolute components, not vector lengths.

| Interval | v2 action | Center difference (mm) | Radius difference (mm) | Full data-objective difference | Cached | End-to-end |
| --- | --- | ---: | ---: | ---: | --- | --- |
| W191 0-40 | Refine | 0.488994 | 0.529597 | +2.82496e-7 | PASS | PASS |
| W191 40-128 | Preserve | 0 | 0 | 0 | PASS | PASS |
| W191 128-198 | Preserve | 0 | 0 | 0 | PASS | PASS |
| W254 0-37 | Refine | 0 | 0 | 0 | PASS | PASS |
| W254 37-81 | Refine | 0 | 0 | 0 | PASS | PASS |
| W254 86-137 | Preserve | 0 | 0 | 0 | PASS | PASS |
| W254 139-195 | Refine | 0.123441 | 0.101238 | +1.35057e-7 | PASS | PASS |

All eleven observation arrays, regenerated raw-v4 geometries and reprojection
geometries match exactly. Native diagnostic-precision sphere references and
canonical regular mappings match exactly. Every rotation matches exactly.
The same three intervals preserve their geometry exactly and the other four
refine. W254 37-81's small positive full-ray floor loss is not treated as zero.

Maximum full-observation data-objective difference is `2.82496e-7`; maximum
final combined-objective difference is `1.92020e-9`. Maximum final pixel
distribution-summary difference is `0.000183419` pixels. All additional
diagnostic, validity, safety and geometric-budget checks pass. Existing ego
geometry files, observation caches, results and manifests were not overwritten;
no ego retraining is required solely for this preparation generalization.

## Certificate and provenance

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_release_v4\verification.json
```

Certificate SHA-256:
`4bab1ae08a38a1df68a68d0724e62411965f7b2be0020a1c5f2a5f091593b9b6`.

Immutable fixture manifest SHA-256:
`ca6e0f36a2757dba20b78ffdc54bb2ee555e7c9ea371f73838af6379ee67ddb8`.

Current-code verification includes all numerical stages; the earlier strict
v2 certificate and replay artifacts remain untouched. The uncertified v3
namespace was not promoted or reused as a release. New cached/end-to-end stage
artifacts, PLYs and their hash-backed manifests are isolated below the v4 root.
The acceptance implementation is commit `498f9d9`; subsequent documentation
does not alter the code signature certified by this report.

## Stationary-scene integration smoke

The production `prepare_final_ellipsoids.sh` launcher, using the newly valid
gate, completed W125 124-149 in a fresh diagnostic namespace. Maximum camera
baseline was `0.000355155 m`; weak parallax was reported without changing the
fitter or weights. Observations contain 6,656 sampled source points, 41,046
supported pairs and 3,140 unknown pairs.

| Quantity | Result |
| --- | --- |
| Final center, NAG world m | `[3.066012, -10.371600, -11.308392]` |
| Final radii, local-axis m | `[43.337823, 57.522747, 17.142857]` |
| Native diagnostic sphere radius | `73.6549 m` |
| Final maximum camera normalized radius | `0.65` |
| Conservative clearance | `6.00000000091 m` |
| Minimum radius | `17.142857 m` |
| Maximum axis ratio | `3.355494` |
| Full original data loss | `0.055282717504` |
| Full final data loss | `0.056389371353` |
| Full pair/depth validity | `100% / 100%` |

The final data loss is within the unchanged 2% budget including `1e-6` allowance.
All stages completed, sphere/ellipsoid meshes and inspection assets were
exported, native reference metadata reports no trainer/training, and no
checkpoint folder exists in the smoke outputs. This demonstrates preparation
integration, not static training quality.

```text
D:\CGProjectData\waymo\diagnostics\autowork_static_preparation_release_v4\
  segment-12511696717465549299_frames_124_149_step1\
    sampling_floor_v2\attempt_001\ellipsoid.ply
```

A `DRY_RUN=1` training-launcher invocation validated that preparation's input,
environment, code and stage-artifact signatures and planned six matched jobs:
regular and Visible-UV ellipsoids, three seed-42 repeats each, legacy mapping,
zero margin and final-only checkpoints. No jobs ran and no training directory
was created. A separate preparation dry-run accepted the default v4 gate
without creating files.

## Automated checks and remaining input blockers

All 87 relevant unit/regression tests pass: 31 generalized preparation, 12
sampling-floor, four reprojection, and 40 matched paper-baseline tests. This
includes rejection tests for changed objectives, observation counts, visibility,
safety, rotations, exact preserved arrays, old acceptance policies and incomplete
reports. Shell syntax, whitespace checks and read-only immutable-fixture
preflight pass. Initialization tests prove trainer, training and checkpoint
methods are not called.

The final audit still discovers 25 matched intervals. Four W125 intervals
currently have all preparation inputs; 21 require point clouds. Four of those
also retain missing-depth blockers:

| Interval | Missing depth frames |
| --- | --- |
| W203 60-107 | 97-99 |
| W975 0-68 | 6-9, 58-68 |
| W975 69-99 | All selected frames |
| W975 99-162 | 99 |

The release does not repair those inputs, skip frames or authorize automatic
downloads or training. Ready intervals can proceed using the existing scripts;
download/extraction and training remain user-run. See [the preparation guide](../docs/generalized_ellipsoid_preparation.md).
