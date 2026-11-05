# Matched PCA enclosure with shared safety expansion

This adds **PCA + shared safety** to the seven-ego comparison, following the
approved choice to apply the same safety expansion as D0/D1/D2. It is not an
unmodified-original-PCA treatment or another reprojection loss ablation. The
completed D0/D1/D2 cohorts and any original PCA artifacts stay untouched.

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
not the mean camera. Apply the unchanged `enforce_safety` function afterward
to all seven intervals: maximum camera normalized radius 0.65, conservative
clearance at least 6 m, minimum radius 8 m and maximum axis ratio 10. There is
no clipping, Moderate-Y change, reprojection fit or sampling floor. PCA axes
also determine the legacy texture orientation. This therefore
compares whole preparation methods, including selection/orientation differences;
it does not isolate covariance estimation alone.

Prepare in the historical processing Python environment. Training uses the
matched native NAG environment and per-interval main configuration, regular
mapping, legacy poles, zero runtime margin, three fresh seed-42 repeats, ten
workers, epoch 80, live terminal/log output and final-only checkpoints.

This preserves the original PCA center and axes, not its original radii or
every setting of the exploratory trainings. Historical runs sometimes applied
a 2% runtime expansion; this matched baseline uses zero, as do current D0/D1/D2.
Shared safety expansion is a separate preparation step, not a runtime margin.

## Shared safety rule and inspection assets

The historical W191 40–128 fit leaves 53 of 89 initialized camera centers
outside its surface, with maximum normalized radius about 1.029. Other ego
PCA fits contain their initial cameras but also lack the shared clearance
safeguards. Apply the same expansion to every interval, not a special repair
or a camera-centered construction for the one problematic interval.

Let the original radii be `a`. First set `b = max(a, 8 m)` element-wise,
then `b = max(b, max(b)/10)`. Compute the largest normalized camera radius
`rho` using these provisional radii. Multiply every provisional radius by
`max(1, rho/0.65, rho + 6 m/min(b))`. This is exactly the existing shared
projection, not a new fitting objective. Neither center nor rotation changes;
no radius decreases, so all-point enclosure is preserved.

Preparation exports both `pca_original.json/.ply` and the safety-expanded
`ellipsoid.json/.ply` used for training. `pca_diagnostics.json` records original
and effective camera/grid measurements, shared safety settings and per-axis
radius factors. A 73x49 image grid is diagnostic, not an exhaustive visibility
or foreground-depth test. Any failed safety check, outside effective camera
or missed effective grid ray blocks preparation; there is no bypass flag.
The former `--allow-outside-cameras` option is no longer accepted.

## Run commands

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only audit; hashes clouds and validates matched source runs.
DRY_RUN=1 bash prepare_paper_pca_baselines.sh

# Rebuild all seven PCA + shared-safety geometries; no NAG training.
bash prepare_paper_pca_baselines.sh
```

Inspect the printed PLYs and camera/coverage diagnostics before training. The
two batches should be run serially, not alongside other GPU jobs:

```bash
# W191 0–40, 40–128, 128–198: nine trainings, roughly 5.1 h buffered.
bash run_paper_pca_baselines_part1of2.sh

# W254 0–37, 37–81, 86–137, 139–195: twelve trainings, roughly 7.4 h buffered.
bash run_paper_pca_baselines_part2of2.sh

# Read-only status, available after all seven PCA preparations are complete.
bash run_paper_pca_baselines.sh --summary
```

These are 21 new trainings, not selected historical repeats. Estimates use the
slowest main regular-ellipsoid repeat for each interval times 1.35; original
PCA + shared-safety geometry may have different runtime. Preparation is separate. An eight-hour
scheduling budget checks the estimated next job and never kills active work.
Rerun the same batch after a budget stop or resolved failure; completed validated
attempts are skipped, while failures and logs are preserved and retries start
fresh. Use `--max-hours` to override the budget. After preparation, `DRY_RUN=1`
prints pending training commands without training. `--intervals W191_0_40`
selects individual training/status intervals; preparation always covers all seven.

## Outputs and provenance

```text
D:\CGProjectData\waymo\ellipsoids\autowork_paper_pca_shared_safety_v1\
  manifest.json
  matrix.json
  W191_0_40\PCA\attempt_001\ellipsoid.json
  W191_0_40\PCA\attempt_001\ellipsoid.ply
  W191_0_40\PCA\attempt_001\pca_original.json
  W191_0_40\PCA\attempt_001\pca_original.ply
  W191_0_40\PCA\attempt_001\front_points.ply
  W191_0_40\PCA\attempt_001\camera_centers.ply
  W191_0_40\PCA\attempt_001\pca_diagnostics.json

C:\Coding\CGProject\nag1\runs\autowork_paper_pca_shared_safety_v1\
  manifest.json
  autowork_W191_0_40_PCA_safety_regular_seed42_repeat01\attempt_001\...
```

Cloud metadata/NPZs, processing source and launcher files receive SHA256 pins;
the processing interpreter/library identity is frozen. Historical calibration,
main configurations, release fixtures and training-environment checks remain
in force. Source and output conflicts fail explicitly; incomplete attempts are
preserved. No historical geometry, manifests, caches or model code are edited.
Custom input/output roots must be repeated on all commands. The old original
PCA roots (`autowork_paper_pca_baselines_v1`) and ablation roots are refused,
and the shared generalized-training lock serializes queues. Rerun preparation
for this new recipe; do not copy or relabel an old PCA stage manifest.

## Implementation verification

Validation covers exact original PCA export, an actual worker invocation in
the historical processing interpreter on synthetic NPZs, equality with the
existing safety projection, unchanged center/rotation, monotonic radii and
point enclosure, original/effective mesh export, invalid-safety rejection,
protected roots, read-only commands and immutable completed-stage retries.
Use canonical `/mnt/c/Coding/CGProject/nag1` capitalization for archived-data
tests, as required by unchanged strict historical path-identity checks.

The shared-safety revision passed all 105 combined tests and the real
seven-interval read-only preparation audit (21 planned trainings). A read-only
projection check on all seven historical PCA geometries preserved every
center/rotation exactly, satisfied all four shared safeguards, and produced
maximum camera radii between 0.507 and 0.576 with 6 m conservative clearance.
Those checks did not regenerate point-cloud fits, export real-data meshes,
create either preparation/training root, or run training. All four shell
entrypoints passed `bash -n`.
