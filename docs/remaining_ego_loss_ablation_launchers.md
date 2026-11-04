# Remaining ego loss ablations

This extends the completed five-interval loss ladder to the four remaining
ego intervals. It does not replace the original batch, alter its launcher or
manifests, refit D1/D2, or introduce another ellipsoid recipe. All numerical,
input, calibration, configuration, release-fixture and environment checks are
delegated to the unchanged original driver and hash-pinned ablation kernel.

## Matrix and estimated training time

Regular mapping only; legacy poles; zero runtime margin; three fresh seed-42
repeats; ten workers; epoch 80; final-only checkpoints; no training resume.
D0 sets the original reprojection weight to exactly zero with depth weight
0.15. D1 uses the saved original reprojection/depth fit. D2 reuses the completed
main sampling-floor recipe. Other optimizer, safety and training settings
remain matched to the original ablation.

| Batch | Interval | New D0 | New D1 | Reused D1 / D2 | Typical / buffered training |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | W191 40–128 | 3 | 0 | 3 / 3 | 1 h 49 min / 2 h 34 min |
| 1 | W191 128–198 | 3 | 0 | 3 / 3 | 55 min / 1 h 17 min |
| 2 | W254 0–37 | 3 | 3 | 0 / 3 | 1 h 39 min / 2 h 17 min |
| 2 | W254 139–195 | 3 | 3 | 0 / 3 | 3 h 12 min / 4 h 19 min |
| **Total** | **4 intervals / 36 cells** | **12** | **6** | **6 / 12** | **7 h 35 min / 10 h 28 min** |

The W191 D1/D2 reuse is based on exact float32 runtime center/radii/rotation
identity, not scores. Each pointer is validated against its intended geometry
and matched settings. The 18 new trainings plus 18 reused cells complete all
seven ego intervals when combined with the original three ego ablations.

Typical timing uses the three historical regular-D2 repeat durations per
interval; buffered timing uses the slowest repeat times 1.35. These are not
runtime guarantees. D0 fitting is separate (roughly another 15–30 minutes for
four intervals based on the preceding fits). The two serial batches estimate
3 h 51 min and 6 h 36 min buffered training, respectively.

## Commands

Use the existing `nag1` environment. Do not run GPU training queues in parallel.
No download, point-cloud extraction, sphere initialization or D1/D2 refit is
required. First audit and prepare all four new D0 meshes:

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Read-only audit: prints reuse and time estimates without creating folders.
DRY_RUN=1 bash prepare_remaining_ego_loss_ablations.sh

# Fits/exports the four D0 ellipsoids; no NAG training. Inspect printed PLYs.
bash prepare_remaining_ego_loss_ablations.sh

# Run these at separate times, or sequentially; never concurrently.
bash run_remaining_ego_loss_ablations_part1of2.sh
bash run_remaining_ego_loss_ablations_part2of2.sh

# Read-only status after preparation, including all reused cells.
bash run_remaining_ego_loss_ablations.sh --summary
```

Each training invocation defaults to an eight-hour scheduling budget. It
checks elapsed time plus estimated next-job cost before starting a training;
it never kills an active training. Reinvoke the same batch after a budget
stop or resolved failure. Completed validated attempts are skipped; failed or
incomplete attempts and console logs are preserved, and retries start fresh.
For a shorter budget append `--max-hours 4`. Training stdout/stderr streams
live to the terminal and per-attempt console log, as in the original driver.

After preparation, prefix either batch with `DRY_RUN=1` to validate the matrix
and print only pending commands without training. All four D0 preparations
must be complete before training/dry-run/summary. For one interval instead:

```bash
bash run_remaining_ego_loss_ablations.sh --intervals W254_139_195
```

## Separate output roots and immutable provenance

```text
D:\CGProjectData\waymo\ellipsoids\autowork_remaining_ego_loss_ablations_v1\
  manifest.json
  matrix.json
  W191_40_128\D0\attempt_001\ellipsoid.json
  W191_40_128\D0\attempt_001\ellipsoid.ply
  ...                          # fitting diagnostics, point/camera PLY assets

C:\Coding\CGProject\nag1\runs\autowork_remaining_ego_loss_ablations_v1\
  manifest.json                # 36 cells, including explicit reuse pointers
  autowork_W191_40_128_D0_regular_seed42_repeat01\
    attempt_001.launch.json
    attempt_001.console.log
    attempt_001\              # native run with final-80.ckpt
```

The extension scopes the original driver's interval selection in memory and
restores it afterward. The old driver, console-compatibility ledger, numerical
kernel, fixtures and completed original batch remain unchanged. The extension
and its four shell launchers receive additional exact SHA256 provenance pins.
Do not edit these files or fitting/training inputs after preparation starts.
Custom roots require both overrides on every prepare/train/status invocation;
the old ablation roots and their descendants are explicitly refused.

The original lock serializes these queues with the existing generalized
training queues. Other independently launched GPU jobs must still be avoided.
Changed inputs or conflicting output manifests are errors, never silently
overwritten or re-labelled as completed results.

## Implementation verification

All 91 combined loss-ablation, remaining-cohort, reprojection, generalized
preparation and sampling-floor tests passed. Use the canonical repository
capitalization (`/mnt/c/Coding/CGProject/nag1`) for archived-data tests: the
existing path-identity checks intentionally distinguish differently cased
absolute path strings even on the Windows-backed filesystem. No path check
was relaxed for this extension.

The real read-only preparation audit passed all four intervals and confirmed
18 new / 18 reused cells without creating either new output root. The original
batch's read-only summary still reports 24 reused, 21 new completed and zero
pending. All four shell launchers passed `bash -n`. No real-data fitting,
training, checkpoint loading or rendering was run during implementation.
