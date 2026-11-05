# Matched original PCA-enclosure baseline

This adds the first, non-camera-centered PCA construction to the seven-ego
comparison. It is a preparation-method baseline, not another reprojection
loss ablation. The completed D0/D1/D2 cohorts and source geometry stay untouched.

## Exact recipe

Use all points with either saved Waymo FRONT camera association in each
YAML-selected frame, with the existing pose-mismatch correction and common
NAG coordinates. There is no foreground-mask rejection, point subsampling,
quantile trimming or static-point classification for this original baseline.
The source extraction is the existing first-LiDAR-return cloud, not a new
both-returns extraction.

The unchanged processing function computes the covariance eigenvectors in
descending eigenvalue order, the local all-point bounding-box midpoint and
half extents, and the single uniform scale needed to enclose every selected
point. The center is the local-box midpoint transformed to world coordinates,
not the mean camera. There is no later safety expansion, minimum radius,
aspect-ratio cap, clipping, Moderate-Y change, reprojection fit or sampling
floor. PCA axes also determine the legacy texture orientation. This therefore
compares whole preparation methods, including selection/orientation differences;
it does not isolate covariance estimation alone.

Prepare in the historical processing Python environment. Training uses the
matched native NAG environment and per-interval main configuration, regular
mapping, legacy poles, zero runtime margin, three fresh seed-42 repeats, ten
workers, epoch 80, live terminal/log output and final-only checkpoints.

This preserves the original PCA construction, not every setting of the old
exploratory trainings. In particular, historical runs sometimes applied a
2% runtime expansion; this matched baseline uses zero, as do current D0/D1/D2.

## Important: original PCA does not guarantee camera enclosure

The historical W191 40–128 fit leaves 53 of 89 initialized camera centers
outside its surface, with maximum normalized radius about 1.029. This is a
limitation of enclosing observed points, not a reason to silently move or
expand the baseline. Other historical ego PCA fits have their camera centers
inside, but do not satisfy the later D0/D1/D2 clearance safeguards either.

Preparation exports the unchanged PCA geometry and a `pca_diagnostics.json`
with per-frame camera radii and forward ray-intersection coverage on a 73x49
image grid. Grid misses are diagnostic, not an exhaustive all-pixel test or a
claim about visibility behind foreground. Missed rays and short hit distances
may affect reconstruction. No camera or geometry is silently repaired.

Training a selected interval with any outside/on-surface camera requires the
explicit `--allow-outside-cameras` acknowledgement. Read-only audit, preparation,
dry-run and summary do not require it. If a shared-safety PCA treatment is
wanted instead, it must have a distinct recipe/output root; it is not provided
by this original-PCA launcher.

## Run commands

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only audit; hashes clouds and validates matched source runs.
DRY_RUN=1 bash prepare_paper_pca_baselines.sh

# Rebuild all seven PCA geometries; exports inspection assets, no NAG training.
bash prepare_paper_pca_baselines.sh
```

Inspect the printed PLYs and camera/coverage diagnostics before training. The
two batches should be run serially, not alongside other GPU jobs:

```bash
# W191 0–40, 40–128, 128–198: nine trainings, roughly 5.1 h buffered.
# Acknowledge the original outside-camera case only after inspecting it.
bash run_paper_pca_baselines_part1of2.sh --allow-outside-cameras

# W254 0–37, 37–81, 86–137, 139–195: twelve trainings, roughly 7.4 h buffered.
bash run_paper_pca_baselines_part2of2.sh

# Read-only status, available after all seven PCA preparations are complete.
bash run_paper_pca_baselines.sh --summary
```

These are 21 new trainings, not selected historical repeats. Estimates use the
slowest main regular-ellipsoid repeat for each interval times 1.35; original
PCA geometry may have different runtime. Preparation is separate. An eight-hour
scheduling budget checks the estimated next job and never kills active work.
Rerun the same batch after a budget stop or resolved failure; completed validated
attempts are skipped, while failures and logs are preserved and retries start
fresh. Use `--max-hours` to override the budget. After preparation, `DRY_RUN=1`
prints pending training commands without training. `--intervals W191_0_40`
selects individual training/status intervals; preparation always covers all seven.

## Outputs and provenance

```text
D:\CGProjectData\waymo\ellipsoids\autowork_paper_pca_baselines_v1\
  manifest.json
  matrix.json
  W191_0_40\PCA\attempt_001\ellipsoid.json
  W191_0_40\PCA\attempt_001\ellipsoid.ply
  W191_0_40\PCA\attempt_001\front_points.ply
  W191_0_40\PCA\attempt_001\camera_centers.ply
  W191_0_40\PCA\attempt_001\pca_diagnostics.json

C:\Coding\CGProject\nag1\runs\autowork_paper_pca_baselines_v1\
  manifest.json
  autowork_W191_0_40_PCA_regular_seed42_repeat01\attempt_001\...
```

Cloud metadata/NPZs, processing source and launcher files receive SHA256 pins;
the processing interpreter/library identity is frozen. Historical calibration,
main configurations, release fixtures and training-environment checks remain
in force. Source and output conflicts fail explicitly; incomplete attempts are
preserved. No historical geometry, manifests, caches or model code are edited.
Custom input/output roots must be repeated on all commands. The old ablation
roots are refused, and the shared generalized-training lock serializes queues.

## Implementation verification

All 104 combined PCA, ablation, reprojection, generalized-preparation and
sampling-floor tests passed. PCA tests include an exact comparison against
the original covariance/extrema/uniform-scale function, an actual worker
invocation in the historical processing interpreter on synthetic NPZs,
foreground-association selection, outside-camera reporting/acknowledgement,
read-only commands, hash-conflict detection and completed-stage retry checks.

The real seven-interval preparation dry-run passed and reported 21 new runs
without creating either output root. The completed remaining-ego ablation
summary still reports 18 new completed, 18 reused and zero pending. All four
shell launchers passed `bash -n`. No real-data PCA generation, NAG training,
checkpoint loading or rendering was performed during implementation. Use the
canonical `/mnt/c/Coding/CGProject/nag1` spelling for archived-data tests, as
required by the unchanged strict historical path-identity checks.
