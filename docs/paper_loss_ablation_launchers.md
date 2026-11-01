# Matched loss-ablation launchers

Prepared 1 November 2026. This implements the
[approved matrix proposal](paper_ablation_matrix_2026-11-01.md); it does not
launch real-data fitting or training during implementation.

## What will run

Regular mapping only, legacy poles, zero runtime margin, three fresh seed-42
repeats, ten workers, epoch 80, final-only checkpoints and the matched source
YAML for each interval. No Visible-UV, upright fitting, clipping, compactness,
radius-scale penalty, axis-order constraint or resolution change is added.

| Interval script suffix | New D0 depth-only | New D1 reprojection/depth | Reused D1 / D2 | Buffered new-training estimate |
| --- | ---: | ---: | ---: | ---: |
| W191_0_40 | 3 | 3 | 0 / 3 | 2.49 h |
| W254_37_81 | 3 | 3 | 0 / 3 | 4.81 h |
| W254_86_137 | 3 | 0 | 3 / 3 | 1.71 h |
| W203_60_107 | 3 | 0 | 3 / 3 | 1.70 h |
| W975_99_162 | 3 | 0 | 3 / 3 | 1.90 h |
| **Total** | **15** | **6** | **9 / 15** | **12.61 h** |

The last three intervals have **exactly equal float32 runtime center, radii
and rotation for D1 and D2**. Their nine D1 cells explicitly alias the matching
main D2 repeats; do not describe these as independent new trainings or a new
geometry difference. This is identity-based reuse, not quality-based selection.

Older regular D1 single runs for W191 0–40 and W254 37–81 have matching saved
settings, but lack frozen training-code/environment manifests and explicit
repeat identities. They are not silently counted as matched new repeats.
All six D1 repeats for those two intervals therefore start fresh.

The original upper bound of 30 new jobs is reduced to **21 new jobs** after
the audit. Estimates use the maximum duration of the three completed main
regular ellipsoid repeats for that interval, multiplied by 1.35, then by the
new job count. They exclude geometry preparation and audit overhead and are
not runtime guarantees for new geometry.

## Run preparation first

Use the existing `nag1` environment and repository. Do not run these GPU
training queues in parallel with other training or GPU flow extraction.

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional: read-only input/provenance/reuse audit; no fitting or output files.
DRY_RUN=1 bash prepare_paper_loss_ablations.sh

# Fit the five new D0 geometries and export PLYs. CPU fitting, no NAG training.
bash prepare_paper_loss_ablations.sh
```

No download, point-cloud extraction, raw-v4 rerun or sphere bootstrap is
needed. The existing raw-v4 initializers, observation NPZs and original D1
fits are reused with immutable fixture or released-stage hash checks.

Inspect the printed D0 PLY paths before training. Do not manually modify or
repair selected ellipsoids; unsafe/invalid full-observation fits fail explicitly.
Retries skip validated completed preparation stages and keep incomplete
attempts. If a completed artifact or its source/recipe changes, stop and
inspect; use a new `autowork_` root rather than overwriting it.

## Run one interval queue at a time

Each of these is a separate invocation, not a command to start concurrently:

```bash
bash run_paper_loss_ablations_W191_0_40.sh
bash run_paper_loss_ablations_W254_37_81.sh
bash run_paper_loss_ablations_W254_86_137.sh
bash run_paper_loss_ablations_W203_60_107.sh
bash run_paper_loss_ablations_W975_99_162.sh
```

Each queue has an **8-hour scheduling budget**: before starting the next job,
it checks elapsed time plus that interval's buffered historical cost. It
never kills an active training. A slow active job can exceed the estimate;
rerun the identical script after a budget stop to finish pending jobs.
The existing main-benchmark launchers' 12-hour defaults are unchanged.

For a shorter invocation, append `--max-hours 4`. After preparation, prefix
any interval command with `DRY_RUN=1` to validate and print remaining commands
without starting training. Training/dry-run/summary require all five D0
preparation stages to be complete.

```bash
# Read-only status of all 45 cells, including the 24 reused cells:
bash run_paper_loss_ablations.sh --summary

# Status for one interval:
bash run_paper_loss_ablations_W191_0_40.sh --summary
```

The driver locks across interval queues, preserves console logs and failed
attempts, starts a new attempt rather than resuming a checkpoint, and skips a
completed attempt only after checking success, all final frames, epoch-80
metrics, settings, calibration, input paths and the exact runtime geometry.
It records launch markers even if training fails before creating a run folder.

## Output and provenance

```text
D:\CGProjectData\waymo\ellipsoids\autowork_paper_loss_ablations_v1\
  manifest.json
  matrix.json                          # all 45 cells; explicit reuse pointers
  W191_0_40\D0\attempt_001\ellipsoid.ply
  ...                                  # also JSON, diagnostics, points, cameras

C:\Coding\CGProject\nag1\runs\autowork_paper_loss_ablations_v1\
  manifest.json                        # full plan and immutable 45-cell matrix
  autowork_W191_0_40_D0_regular_seed42_repeat01\
    attempt_001.launch.json
    attempt_001.console.log
    attempt_001\                       # native NAG run, final-80.ckpt
```

D1 and D2 source meshes already exist; their unchanged JSON/PLY locations and
reused run pointers are recorded in the matrix. No historical manifest or
geometry is rewritten. Preparation/training roots can be overridden with
`--preparation-root` and `--training-root`; keep the exact overrides on every
prepare/train/retry/status command. Output roots must be separate `autowork_`
namespaces. Do not edit source/configuration files mid-batch.

## Numerical isolation and validation

`reports/paper_ablation_kernel.py` pins the released reprojection module's
SHA256 and privately loads an exact copy with only its positive-weight guard
relaxed to permit zero reprojection weight. The production module remains
unchanged and still rejects zero. D0 uses the same sampling, Huber functions,
L-BFGS-B starts/settings, bounds, feasibility penalties and final safety
projection as D1; valid-pair feasibility is retained even though pixel-error
cost has weight zero. The source initializer stays in its historical raw-v4
interpreter provenance; this does not rerun or move that optimization.

Common safety remains rho <= 0.65, conservative clearance >= 6 m, radii >= 8 m
and max/min radius <= 10. D0 is checked on all supported pairs/depth rays
before export. A shape that looks unusual is not automatically rejected or
replaced if these numerical checks pass. D1 uses the exact original stored
fit rather than refitting it in another numerical environment.

The driver checks the existing seven-interval release gate, current matched
training interpreter/libraries/environment, exact original D1 objective, cache
input snapshots and pinned geometry/array files. Old ego cache provenance is
accepted only through the immutable release fixtures; static stages must match
the released code/recipe and their artifact hashes. This is not a general
waiver of historical cache metadata or model-code checks.

No real D0 fitting or training was performed during implementation. Synthetic
tests verify exact positive-mode equivalence with the released fitter, zero
pixel-error contribution in D0, safety/export, stage reuse and conflict
preservation, the 21/24 job split, final-only command arguments and bounded
scheduling. Preparation dry runs validate the real five-interval inputs.

Implementation verification passed 78 combined ablation, reprojection,
generalized-preparation and sampling-floor tests. The real preparation
dry-run passed all five input audits and reported the table above without
creating preparation or training roots. The native NAGConfig parser accepted
all generated configuration overrides, all seven shell launchers passed
syntax checks, and the production reprojection source retained its exact
pinned SHA256. No real D0 geometry or new training result is claimed yet.
