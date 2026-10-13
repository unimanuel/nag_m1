# Full aligned sampling-floor replay

This diagnostic follows the [fixed-parameter objective/Jacobian test](controlled_floor_evaluations.md).
It runs the complete frozen sampling-floor refinement for W191 0-40 using
64-byte-aligned copies of every cached observation array. Values, dtypes,
shapes, strides, selection/order and fitter settings remain unchanged.

The default batch contains nine fits: three fresh-process repeats of historical
v1, historical v2 and generalized v2. Arm order rotates between repeats. All
implementations load before the observations, and each worker uses the previous
replay's interpreter/build, one CPU core, one numerical thread and fixed Python
hash seed. The unchanged kernel performs its original 5/10/20 continuation,
SLSQP finite differences, candidate retention, safety repair and full-observation
selection. There is no optimizer interception or replacement of its gradient.

It does **not** run NAG training, GPU initialization, checkpoints, raw-v4 or
reprojection fitting. No historical geometry, production source, preparation
cache metadata, release status or comparison tolerance is modified. A passing
diagnostic would still not certify the full seven-interval preparation gate.

## Commands

The preceding controlled replay and its frozen fixtures must exist unchanged.

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1
DRY_RUN=1 bash run_controlled_aligned_floor_replay.sh
bash run_controlled_aligned_floor_replay.sh
```

Timing the first real fit took approximately nine seconds. The default nine-fit
batch is expected to take roughly two minutes on this machine, not hours of
training. A reusable timing-only invocation is available:

```bash
bash run_controlled_aligned_floor_replay.sh --max-new-fits 1
bash run_controlled_aligned_floor_replay.sh
```

The first invocation stops after one newly completed fit. The second reuses it
and runs the remainder. Changing this per-invocation fit cap does not change
the experiment signature. Partial batches never write a complete summary.

Default output:

```text
D:\CGProjectData\waymo\diagnostics\autowork_aligned_floor_replay_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_aligned64_VERSION_repeatNN\attempt_001\
    plan.json
    result.json
    console.txt
```

Each result contains geometry, full fitter diagnostics and continuation history,
runtime, observation alignment, fingerprints and the unchanged comparison to
stored geometry. The summary separates within-version repeatability,
between-version comparisons and deviations from the stored optimum. It also
compares each arm against the previous unaligned replay, explicitly an archived
baseline rather than a simultaneous native-layout arm in this new harness.

Completed attempts are reused only with matching hashes/signatures; incomplete
attempts remain intact, and unowned/conflicting output roots are rejected. All
frozen inputs and previous replay artifacts are hash-validated. Workers reject
changed numerical sources, runtime mismatches, alignment failures or input
mutation. Dry-run does not create output or execute fits.

For genuinely independent repetition, choose a new output root:

```bash
bash run_controlled_aligned_floor_replay.sh \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_aligned_floor_replay_v1_batch2
```

No PLY or training files are generated: this replay checks submillimeter numerical
compatibility, not a new surface recipe. Alignment of observation inputs does not
control every intermediate allocation, so fixed-point equality from the preceding
test does not guarantee equality of all subsequent optimizer trajectories.
