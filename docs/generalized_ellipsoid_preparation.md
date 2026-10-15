# Configuration-driven legacy ellipsoid preparation

This pipeline discovers compatible Waymo intervals from YAML, not a built-in
scene list. It keeps the legacy texture mapping and the publication recipe:
corrected-Y-up raw-v4 initialization, original reprojection/depth fitting, then
the exact-zero v2 sampling-floor preservation/refinement policy. No upright,
compactness, radius-scale, clipping or manual vertical enlargement is enabled.

Existing ego files, caches, renders and manifests are never overwritten.
The historical ego numerical fixtures were frozen before refactoring under:

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_v1\ego_fixtures.json
```

## Compatibility gate and current limitations

Preparation and training are deliberately blocked unless the full seven-ego
comparison passes **both** cached-input and end-to-end verification for the
current code/recipe. The checker compares all observation arrays exactly,
checks the v2 preservation decisions, and uses `rtol=1e-6`, `atol=1e-3 m`
for fitted centers/radii and `atol=1e-6` for rotations. Preserved arrays require
exact equality. The 1 mm absolute allowance was explicitly approved by the user
on 2026-10-13 after isolated historical-source and numerical sensitivity tests.
It changes acceptance only, not the fitter, recipe or historical geometries.
No automatic tolerance adjustment is implemented.

The versioned certificate also requires close combined/data/component objectives
(`rtol=1e-6`, `atol=1e-6`), matching full-observation counts and validity, and
close pixel/angle/log-depth distributions. Their absolute allowances are
`1e-3` pixels, `1e-4` degrees and `1e-4` log-depth units, respectively, with
`rtol=1e-6`. Full-ray sampling statistics use `rtol=1e-5`, `atol=1e-6`.
All original safety limits and the 2% full-observation data-loss budget still
apply; a close geometry with changed losses, visibility or unsafe cameras fails.
Observation arrays, diagnostic-precision sphere references, mapping metadata and
preserved geometry keep their exact requirements.

The current approved-policy certificate passed all seven intervals in both
layers (14/14), and the production `release_gate` accepted its current code,
recipe, fixture and policy hashes:

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_release_v4\verification.json
```

Every observation array, raw-v4 fit, reprojection fit and diagnostic-precision
sphere/mapping reference matches exactly. The same three geometries are
preserved exactly; the four others are refined. Maximum final-floor differences
are 0.489 mm in center and 0.530 mm in radius. Full data-objective differences
are at most `2.825e-7`, and final combined-objective differences at most
`1.921e-9`. Existing ego results and geometry files remain unchanged; this
release does not require retraining them. Missing static inputs remain separate
blockers, not waived by the numerical release.

The earlier strict-tolerance replay report is retained read-only:

```text
D:\CGProjectData\waymo\diagnostics\autowork_generalized_preparation_release_v2\verification.json
```

It checks all seven intervals in both layers: six pass, W191 0-40 fails strict
floor-refinement replay. Every stored observation array, raw-v4 geometry,
native diagnostic-precision sphere reference and reprojection geometry matches
exactly. W254 139-195 has small final-floor deviations within the specified
combined relative/absolute tolerance; the other successful floor geometries
match exactly.
Canonical mapping metadata was additionally checked read-only and matches
exactly for all seven native references. The current verifier checks it
explicitly rather than relying on an unchanged relative density ratio.

Final audit/training-launcher hardening was made after that replay, without
changing observation building, sphere initialization or numerical fitting.
Its subsequent `autowork_generalized_preparation_release_v3` namespace was
not certified. The current approved-policy verification namespace is a fresh
`autowork_generalized_preparation_release_v4`.
The v2 report and all its artifacts remain unchanged; no metadata is rewritten
to pretend the latest code was verified by an earlier revision.

A failed report is not a training authorization. The checker collects the
remaining independent comparisons but exits unsuccessfully and leaves
`released: false`. Review the report and the validation note in `reports/`
before attempting static training.

Development found that rerunning even the exact historical v1 fitting source
can differ from its stored W191 0-40 result by fractions of a millimeter. The
strict gate therefore distinguishes identical observation inputs from numerical
optimizer replay. The historical results remain valid stored experiments;
this new preparation release is not silently certified if its approved-policy
comparisons fail. The full gate still has to pass both layers for all seven
intervals on current code; an old failed report is never relabeled as a pass.

The diagnostic-only static smoke used W125 124-149, with a maximum camera
baseline of 0.000355 m. It completed native sphere initialization, observations,
raw-v4, reprojection and floor refinement, without training. Its final maximum
camera normalized radius was 0.65, conservative clearance was 6 m and data loss
remained inside the 2% budget. Its files are isolated under
`D:\CGProjectData\waymo\diagnostics\autowork_static_preparation_smoke_v1`;
this smoke is not a substitute for the release gate or a training authorization.

After the approved-policy gate passed, the production preparation launcher also
completed a fresh W125 124-149 smoke under
`D:\CGProjectData\waymo\diagnostics\autowork_static_preparation_release_v4`.
It used native initialization and the unchanged recipe, passed safety and the
2% data-loss budget, and generated no checkpoints or trainings. A training
dry-run against that preparation validated all stage signatures and planned six
matched jobs (regular/Visible-UV, three seed-42 repeats) without writing training
files. See the [approved-policy release report](../reports/generalized_preparation_release_2026-10-13.md).

The pipeline pins the historical legacy-v2 kernel from commit `07cbd74`, with
source SHA-256 `5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603`.
That source contains the original refinement and generic preservation check.
It is isolated from the later upright/compact implementation, which remains
unchanged. The new wrapper labels the policy v2 uniformly; it never selects a
kernel based on a scene name.

## Commands

### Optional mask-filter validation fix (15 October 2026)

The common ellipsoid training launcher now validates the loader's effective
optional mask-filter path. If `mask_indices_filter.json` is absent, NAG saves
`mask_ids_filter_path: null`; this is valid, not missing required dataset data.
If the file exists, its exact expected path is still required. Required input
paths, unexpected filter paths, calibration, geometry and training settings
retain their strict checks. Do not create a dummy filter file: that would be a
dataset change rather than a validator correction.

This applies to every scene using `run_final_ellipsoid_benchmarks.sh`, not just
W141. The plane/sphere final launcher already supports an absent optional
filter and needs no path-handling change.

The reviewed patch changes validation and signature plumbing only. An explicit
[SHA-256 ledger](compatibility/optional_mask_validation_v1.json) records the
before/after hashes of its four Python files. Code signatures use their earlier
compatibility identities only when **all four current files** match those exact
reviewed hashes. Unlisted code changes and subsequent edits remain detectable;
no fitter, model, data, configuration or recipe is exempted. The CLI announces
the allowance. Raw source hashes remain available with
`code_signature(validation_compatible=False)` and in the ledger. Existing
certificates, preparation outputs and training manifests are not rewritten or
relabeled as newly replayed numerical experiments.

After the fix, rerun the original scene command with the same interval selection
and output root. A completed first training is validated and skipped, not
retrained. For example:

```bash
bash run_final_ellipsoid_benchmarks.sh \
  --intervals W141_2_53 W141_53_101 W141_102_173 W141_174_197 \
  --training-root "$PWD/runs/autowork_final_ellipsoid_W141" \
  --max-hours 7
```

The invocation budget remains a between-training elapsed-time check, not a
guaranteed limit on an active training. Prefix the command with `DRY_RUN=1` for
read-only preparation/provenance validation; a dry run does not itself validate
or skip completed attempts.

Validation of this patch passed 93 CPU unit/regression tests. Read-only checks
accepted the existing full 14/14 certificate, W141's completed first training
and all 19 completed W125 trainings present at the time. The W141 CLI dry run
validated its 24-job manifest; the real resume loop skipped the completed job
and stopped at a tiny scheduling budget with all training calls prohibited.
The final plane/sphere launcher's code/environment preflight also passed.
No geometry was regenerated, no historical manifest/certificate was rewritten,
and no NAG training was run to verify this validation-only correction.

### Standard preparation commands

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Read-only inventory of all 25 matched final intervals:
bash audit_generalized_ellipsoid_preparation.sh

# Full numerical verification; initialization/fitting only, no training:
bash verify_ego_ellipsoid_preparation.sh
```

Verification is resumable. Successful entries are reused after fixture/code
validation. Conflicting revisions are rejected rather than overwriting an
earlier report; use a new `--verification-root` beginning with `autowork_`
when deliberately validating a code revision. Pass that same root to prepare
and train. `--verify-level cached` or `end_to_end` performs an individual layer,
but neither layer alone unlocks preparation. `--intervals W254_37_81` is useful
for a single diagnostic, not a substitute for the seven-interval gate.

`DRY_RUN=1` or `--dry-run` performs read-only checks. It does not initialize
models, fit geometry, download data, train, or create output files.
Audit also distinguishes missing references, supplied/prepared candidates and
the need for native initialization. A candidate's availability is not a claim
that it passed the reuse checks; validation occurs before reuse.

### Point clouds

The local Waymo training list resolves **full** segment IDs to zero-based scene
indices. No numeric download IDs are hand-maintained in these scripts.

```bash
# Preview the exact download/extraction commands first:
DRY_RUN=1 bash prepare_final_pointclouds.sh

# User-run: requires your existing gcloud/Waymo access.
bash prepare_final_pointclouds.sh
```

This uses the processing repository's installed `nagwp` CLI and existing raw
and point-cloud directories on D:. Its native completed-data reuse checks are
retained. The raw-v4 fit runs with that repository's `.venv/bin/python`, not
the NAG interpreter; interpreter/library versions are recorded in provenance.

No download was performed during implementation.

### Geometry and inspection

After the strict gate passes and the selected inputs are available:

```bash
# Existing W125 clouds make it the first candidate preparation group:
bash prepare_final_ellipsoids.sh --intervals W125_0_40 W125_40_93 W125_93_124 W125_124_149
```

The default input directory is `config/waymo/autowork_paper-final`, not the
archival final directory containing incompatible double-resolution variants.
For other configurations use either `--config-dir DIR` or
`--config-paths REGULAR.yaml VISIBLE.yaml`. The interval selector accepts full
identities such as `segment-12511696717465549299_frames_000_040_step1` or an
unambiguous display label. Two different full segments sharing a short prefix
cannot be selected ambiguously.

Default outputs are under:

```text
D:\CGProjectData\waymo\ellipsoids\autowork_generalized_legacy_v1\
  segment-ID_frames_FIRST_LAST_stepSTEP\
    reference\attempt_001\sphere_reference.json
    observations\attempt_001\...
    raw_v4\attempt_001\...
    reprojection_v1\attempt_001\...
    sampling_floor_v2\attempt_001\
      ellipsoid.json
      ellipsoid.ply                 # blue, zero margin, no clipping
      sphere_red.ply
      front_points.ply
      background_points.ply
      camera_centers.ply
      sphere_reference.json
      sampling_floor_diagnostics.json
      reprojection_fit_diagnostics.json
    result.json
```

Initialization-only sphere export needs native NAG model initialization and may
use the GPU. It deliberately does **not** call `NAGRunner.build`, construct a
Trainer, train, render videos, invoke checkpoint callbacks, or save checkpoints.
The native visible-plane-center radius is computed using the existing
foreground/camera setup. Four-decimal reference geometry is retained for legacy
floor compatibility, with full precision in a separate JSON field.

An optional `--sphere-run-map FILE.json` maps each full job identity to an
explicit completed regular-sphere run. Only matching calibration, interval,
initialization settings and radius/mapping policy are accepted. Absence of a
reference triggers native initialization, not an arbitrary radius fallback.
No trained reprojection-ellipsoid run is needed.

Stages record input snapshots, source-config hashes, code hashes, environments,
recipe settings and artifact hashes. Valid completed stages are reused.
Interrupted attempts remain intact and a retry uses another attempt number.
Unowned nonempty directories and changed-input stages are rejected; select a
new output namespace for an intentional revision. Do not edit code or selected
datasets while fitting. Run preparation sequentially.

### Training, separately

Only after inspecting the meshes and passing the release gate:

```bash
DRY_RUN=1 bash run_final_ellipsoid_benchmarks.sh \
  --intervals W125_0_40 W125_40_93 W125_93_124 W125_124_149

bash run_final_ellipsoid_benchmarks.sh \
  --intervals W125_0_40 W125_40_93 W125_93_124 W125_124_149
```

This selects regular and Visible-UV ellipsoids, three same-seed-42 repeats each:
six trainings per interval, or 24 for these four W125 intervals. The geometry
is identical between layouts, with legacy poles, zero runtime margin, no
radius override/clipping and final-only checkpoints. Configuration and runtime
geometry are checked before counting a completed run.
Dataset, image/mask/depth/box paths and calibration are passed explicitly from
the prepared descriptor, including a supplied `--data-root`; completed runs
with another input root or calibration are rejected.

Default training output is `runs/autowork_final_ellipsoid_benchmarks`.
Use a distinct `--training-root` for each different interval selection; changing
the selection in an existing manifest is rejected. A shared lock prevents two
new ellipsoid queues running concurrently. Do not run other GPU training scripts
at the same time. Completed jobs are skipped and failed attempts are preserved.
`--max-hours 9` is a between-job invocation limit, not a hard timeout of an
active training. Rerun the same selection to finish.

## Missing inputs and nearly stationary cameras

The four known initialization-depth gaps remain separate from cloud extraction:

| Interval | Missing depth TIFF frames |
| --- | --- |
| W203 60-107 | 97-99 |
| W975 0-68 | 6-9 and 58-68 |
| W975 69-99 | all selected frames |
| W975 99-162 | 99 |

Restore compatible original depth files or perform a separately approved,
consistent dataset repair. The scripts never omit those frames or insert dummy
depth. Missing cloud frames must be extracted before fitting.

Maximum camera baseline below 0.1 m produces a **diagnostic** weak-parallax
warning only. Weights, observation selection, safety bounds and fitting remain
unchanged. Fewer than 128 supported pairs/rays remains an explicit error.
Foreground masks are used to prefer background observations; this is not
semantic/static-object tracking and does not guarantee every retained point is
stationary. Other datasets require an adapter supplying compatible calibrated
observations; arbitrary Waymo intervals in the existing export convention are
the supported generalization in this version.
