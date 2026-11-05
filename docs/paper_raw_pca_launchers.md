# Raw PCA ego baseline, with W191 40–128 deferred

Use these scripts instead of the shared-safety training launchers. They train
the original, unexpanded PCA enclosures for six ego intervals. They do not
apply shared safety, camera centering, minimum radii, ratio limits, clipping,
Moderate-Y changes, reprojection or sampling-floor refinement. The later
minimal-expansion W191 40–128 exception is deliberately not implemented.

## Reuse the untouched original exports

The completed `autowork_paper_pca_shared_safety_v1` preparation retained
`pca_original.json` and `pca_original.ply` before applying any expansion. The
new preparation validates that source manifest and each completed stage, then
copies the **original** geometry and mesh, not `ellipsoid.json/.ply` from the
expanded treatment. Geometry fit metadata and arrays remain exactly unchanged;
only inspection filenames and new provenance metadata are added. No point
cloud refit or shared-safety preparation rerun is needed.

Original sources, expanded meshes, earlier trainings and manifests remain
read-only. The raw baseline receives separate recipe/run identities and
output roots. SHA256 checks, matched configuration/calibration, release-fixture
checks, and the frozen training environment are inherited unchanged. Source
or output conflicts fail explicitly. Preparation retries reuse only validated
completed stages and retain incomplete attempts.

## Declared feasibility exclusion

NAG initialization requires each initial camera's normalized ellipsoid radius
to be strictly below one. The original W191 40–128 fit has a maximum about
1.029 and 53 outside cameras, so its three raw-PCA jobs are excluded from this
batch. This is a geometric initialization failure, not selection by image
quality. Its source geometry and measured exclusion are recorded in the raw
manifest/matrix. No minimal enlargement, outside-camera bypass or renderer
change is applied. An additional unexpected outside-camera or ray-grid failure
blocks the batch instead of silently repairing/excluding another interval.

If the exception is investigated later, label it as a separate minimally
expanded PCA treatment. For the present PCA-versus-other-method aggregate,
use the same six intervals for every treatment; do not compare a six-interval
PCA average directly against seven-interval D0/D1/D2 or sphere averages. The
main seven-interval benchmark remains unchanged.

## Commands

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only source/feasibility audit; creates nothing.
DRY_RUN=1 bash prepare_paper_raw_pca.sh

# Copy validated unexpanded geometry/inspection assets into the raw namespace.
# No fitting, expansion, GPU work, or training.
bash prepare_paper_raw_pca.sh
```

Inspect the printed PLY paths, then run the batches serially:

```bash
# W191 0–40 and 128–198: six new trainings, roughly 2.5 hours buffered.
bash run_paper_raw_pca_part1of2.sh

# All four W254 intervals: twelve new trainings, roughly 7.4 hours buffered.
bash run_paper_raw_pca_part2of2.sh

# Read-only status after all six raw exports are prepared.
bash run_paper_raw_pca.sh --summary
```

There are 18 fresh trainings: regular legacy mapping, zero runtime margin,
three same-seed-42 repeats, ten workers, epoch 80 and final-only checkpoints.
Runtime estimates use the slowest historical regular-D2 repeat times 1.35;
raw PCA may differ. The default next-job scheduling budget is eight hours;
rerun the identical batch after a budget stop. It skips validated completed
attempts and starts a fresh attempt after failures rather than resuming a
checkpoint. Console stdout/stderr streams live and is retained in attempt
logs. Never run GPU queues concurrently.

The original PCA construction still includes its all-point uniform enclosing
scale. No additional runtime expansion is applied: this matched raw baseline
uses zero margin, unlike some historical exploratory trainings that used 2%.
It compares complete preparation methods, including all-FRONT versus later
mask-filtered fitting observations, not covariance estimation in isolation.

After preparation, `DRY_RUN=1` prints pending training commands without
training. `--max-hours 4` overrides the budget, and `--intervals W254_86_137`
selects an included interval for training/status. No flag enables W191 40–128.

## Output roots

```text
D:\CGProjectData\waymo\ellipsoids\autowork_paper_raw_pca_v1\
  manifest.json
  matrix.json                         # explicit W191 40–128 exclusion
  W191_0_40\raw_PCA\attempt_001\ellipsoid.json
  W191_0_40\raw_PCA\attempt_001\ellipsoid.ply
  W191_0_40\raw_PCA\attempt_001\front_points.ply
  W191_0_40\raw_PCA\attempt_001\camera_centers.ply
  W191_0_40\raw_PCA\attempt_001\pca_diagnostics.json

C:\Coding\CGProject\nag1\runs\autowork_paper_raw_pca_v1\
  manifest.json
  autowork_W191_0_40_raw_PCA_regular_seed42_repeat01\attempt_001\...
```

Custom `--source-preparation-root`, `--preparation-root` and `--training-root`
overrides must be repeated consistently on every command. Source roots, old
original/shared-safety/ablation roots, their ancestors and descendants are
refused as output targets. No existing data is deleted or rewritten. Geometry
is frozen throughout NAG training, while inherited camera/foreground refinement
is unchanged; initial enclosure does not guarantee the learned poses stay away
from the surface.

## Implementation verification

All 116 combined raw-PCA, shared-PCA, ablation, reprojection, generalized
preparation and sampling-floor tests passed. The new tests verify exact
original fit/mesh copying (including radii below 8 m), no fitting/expansion or
training calls, immutable retries, rejection of expanded input and unexpected
outside-camera cases, explicit deferral, protected roots and read-only commands.

The real read-only source audit passed six included intervals and recorded
W191 40–128 at normalized camera radius 1.029007 as deferred. It planned 18
trainings without creating either new output root. All four shell entrypoints
passed `bash -n`. No existing PCA/ablation launcher, model code, source artifact
or manifest was changed; no real-data copying, fitting or training was run.
