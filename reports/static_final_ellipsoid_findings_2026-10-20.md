# Completed static/final ellipsoids and matched-baseline handoff

Audit date: 20 October 2026. No training was launched by Codex.

## Verified cohort

All **120 eligible trainings** passed strict saved-result validation: 20
intervals, regular and Visible-UV layouts, and three fresh seed-42 repeats.
Each interval/layout has exactly one valid attempt per repeat. The earlier
failed W324 42–96 Visible-UV repeat-2 attempt is retained but not counted;
its successful `attempt_002` is counted. Four-worker and historical experiments
are not pooled into this cohort.

Checks cover successful exit, all expected final complete-frame filenames,
finite epoch-80 PSNR/SSIM/LPIPS, saved effective settings, ten workers, seed 42,
unchanged source YAML/geometry hashes, camera calibration, fitted ellipsoid
center/radii/rotation, legacy poles, zero runtime margin, no clipping/radius
override, and exactly one nonempty final checkpoint per valid run. Training
code, preparation signature, release certificate, interpreter, package versions
and relevant environment match the recorded manifests. This audit did not
rerender checkpoints or decode every PNG; completion checks are not a new
image-quality/perceptual audit.

## Aggregate comparison

Average the three repeats within each interval, then weight all 20 intervals
equally. These are same-seed nondeterminism repeats, not independent-seed trials.

| Treatment | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Ellipsoid regular | 32.5765 | 0.921654 | 0.223991 |
| Ellipsoid Visible-UV | 32.5887 | 0.922090 | 0.221888 |
| Visible-UV minus regular | +0.0121 | +0.000436 | −0.002103 |

Visible-UV wins on 11/20 intervals for PSNR, 11/20 for SSIM and 17/20 for
LPIPS. The most consistent observed benefit is perceptual quality, not a
substantial universal PSNR improvement. Do not claim statistical significance
from these counts or three same-seed repeats.

The largest mean improvement is W952 64–140: +0.294 dB PSNR, +0.00601 SSIM,
and −0.01003 LPIPS. Its regular-layout PSNR repeat SD is 0.322 dB, versus
0.035 dB for Visible-UV: the mean difference must be read alongside training
variability. As a sensitivity diagnostic only, removing this interval changes
the mean PSNR delta to −0.0027 dB. It remains included in every main average.

W975 175–195 improves in all metrics (+0.149 dB, +0.00238 SSIM,
−0.00299 LPIPS). W203 3–58 worsens in all three on average (−0.166 dB,
−0.00095 SSIM, +0.00131 LPIPS). W324 is nearly tied in LPIPS but slightly
worse in PSNR/SSIM with Visible-UV. There is no basis for choosing one layout
as a universal per-interval winner or choosing only favorable repeats.

No matched static plane/sphere baseline manifest exists yet. Therefore this
cohort establishes a texture-allocation comparison on fitted ellipsoids,
**not** superiority over sphere or plane. Older static results use different
samplers, workers, epoch update counts or resolution variants and must not be
silently substituted. Near-static scene quality also does not establish a
multi-view-parallax improvement or identify which objects/background regions
explain a metric difference.

## Full individual results and provenance

Generated artifacts on D: preserve per-run metrics, repeat SDs, interval means,
source artifact hashes, selected attempts and the complete 20-row metric table:

```text
D:\CGProjectData\waymo\diagnostics\autowork_static_final_ellipsoids_2026-10-20\
  comparison.md
  runs.csv
  summary.json
```

The reproducible audit source is
[`analyze_static_final_ellipsoids.py`](analyze_static_final_ellipsoids.py).
It refuses missing/duplicate completions, stale provenance, wrong workers/seeds,
nonfinite metrics and existing output directories. Run it with the same `nag1`
Python interpreter; choose a new `autowork_` output root when repeating:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  /mnt/c/Coding/CGProject/nag1/reports/analyze_static_final_ellipsoids.py \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_static_final_ellipsoids_recheck
```

## Next: 180 matched plane/sphere jobs

Each interval receives plane, regular sphere and Visible-UV sphere, each with
three seed-42 repeats. No plane-UV duplicate exists. All use the established
80 epochs, ten workers, 140 batches × 100,000 rays × 20 timestamps, nominal
960×1920 background grid, unchanged foreground/networks/losses and final-only
checkpoints. Spheres use the visible-foreground-plane-center radius rule.
No double-resolution, upright fitting or new geometry constraints are added.

| Separate resumable queue | Intervals | Jobs |
| --- | --- | ---: |
| `run_paper_final_baselines_W125.sh` | 0–40, 40–93, 93–124, 124–149 | 36 |
| `run_paper_final_baselines_W141.sh` | 2–53, 53–101, 102–173, 174–197 | 36 |
| `run_paper_final_baselines_W203_ready.sh` | 3–58 | 9 |
| `run_paper_final_baselines_W324.sh` | 0–42, 42–96, 161–197 | 27 |
| `run_paper_final_baselines_W344.sh` | 0–51, 52–95, 95–135, 135–197 | 36 |
| `run_paper_final_baselines_W952.sh` | 0–63, 64–140, 141–198 | 27 |
| `run_paper_final_baselines_W975_ready.sh` | 175–195 | 9 |

Run one queue at a time with `nag1` activated. Prefix a command with `DRY_RUN=1`
for read-only planning. An invocation defaults to a 12-hour predictive scheduling
budget, not a timeout; it never kills an active job. Rerun the identical command
until its summary reports all intended jobs complete. Add `--summary` after a
queue has started to inspect its saved results. Do not paste all seven commands
as a single unattended batch unless the combined duration is intended.

W203 60–107 and the three earlier W975 intervals remain deferred until the
original depth TIFFs are restored and audited. W324 96–161 remains separately
deferred for resources, even if its depth is present. The new `_ready` wrappers
use explicit interval selections; they cannot automatically expand after depth
restoration. Later missing-interval batches must use distinct roots rather than
changing an existing cohort's selection. No files were downloaded or repaired.

### Storage before starting

The seven ellipsoid output roots currently occupy approximately 414.6 GB
(386.1 GiB), including preserved failed attempts. C: has approximately 272 GiB
free. A rough 1.5× extrapolation for 180 baseline jobs is 621.9 GB (579.2 GiB).
Background-specific outputs and failures can change this; it is not a hard
storage bound. The full remaining suite is unlikely to fit on current C: free
space without additional archiving. Keep a reserve and check between queues.

Defaults remain on C: to preserve the existing storage preference. Alternatively,
explicitly choose a new D: output root before first launch, for example:

```bash
bash run_paper_final_baselines_W125.sh \
  --output-root /mnt/d/CodingD/CGProjectD/nagRunsFromC/autowork_paper_final_baselines_W125
```

Use that exact same `--output-root` on every retry, dry run and summary for that
queue. Do not switch back to the default C: root mid-cohort: that would start a
second independent queue. All raw data, geometry and completed ellipsoid
references stay where they are. No data has been moved automatically.
If baseline outputs are written to D: while ellipsoid outputs were written to
C:, wall-clock comparisons also contain a storage/I/O difference; do not
attribute that runtime difference solely to the background representation.

## Implementation validation

The strict audit validated all 120 selected completed trainings. All seven
production launchers passed read-only dry runs: exactly 20 distinct eligible
intervals and 180 baseline jobs, with ten workers. The combined analysis,
preparation, fitting and paper-baseline unit suites passed 112 tests. Shell
syntax and whitespace checks passed. No model, training YAML, geometry recipe,
historical result or depth input was changed; no training was launched.
