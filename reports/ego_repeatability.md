# Seven-interval training repeatability

## Combined upright-pole experiment (four batches)

Use these instead of the legacy-only launcher for the combined experiment:

```bash
DRY_RUN=1 bash run_ego_poles_repeatability_part1.sh
bash run_ego_poles_repeatability_part1.sh

DRY_RUN=1 bash run_ego_poles_repeatability_part2.sh
bash run_ego_poles_repeatability_part2.sh

DRY_RUN=1 bash run_ego_poles_repeatability_part3.sh
bash run_ego_poles_repeatability_part3.sh

DRY_RUN=1 bash run_ego_poles_repeatability_part4.sh
bash run_ego_poles_repeatability_part4.sh
```

| Batch | Intervals | Trainings | Earlier-rate estimate |
| --- | --- | ---: | ---: |
| Part 1 | W191 40-128; W254 0-37 | 24 | about 10.8 hours |
| Part 2 | W191 128-198; W254 37-81 | 24 | about 11.0 hours |
| Part 3 | W191 0-40; W254 139-195 | 24 | about 10.6 hours |
| Part 4 | W254 86-137 | 12 | about 5.2 hours |

Both use three seed-42 repeats, regular and Visible-UV, and both `legacy`
and `world_y`. The unequal counts minimise the longest batch using historical
training times; the single-interval fourth batch is unavoidable with seven
indivisible intervals. Actual world-Y timing may differ from these estimates.

Only texture coordinates change. Geometry files, runtime margin, training
settings and seed are identical within each pair. No refitting or new PLYs.
`world_y` uses upright physical poles and a fixed world-Z-based longitude
reference; automatic seam placement is deliberately excluded. Visible-UV may
choose a different crop/grid as a consequence of the changed mapping.

Each matched orientation pair runs consecutively; order reverses on the second
repeat. Jobs run sequentially, not concurrently on the same GPU. Separate roots:
`runs/2026-09-25_ego_poles_repeatability_part1of4`,
`runs/2026-09-25_ego_poles_repeatability_part2of4`,
`runs/2026-09-25_ego_poles_repeatability_part3of4`, and
`runs/2026-09-25_ego_poles_repeatability_part4of4`.
All four roots are new so that they cannot accidentally collide with the
superseded two-batch plan.
Rerun the identical wrapper to skip completed jobs and retry incomplete ones.

```bash
bash run_ego_poles_repeatability_part1.sh --summary
bash run_ego_poles_repeatability_part2.sh --summary
bash run_ego_poles_repeatability_part3.sh --summary
bash run_ego_poles_repeatability_part4.sh --summary
```

Summaries separate orientation groups and report paired `world_y - legacy`
metric differences only when both members finished. Positive PSNR/SSIM deltas
and negative LPIPS deltas favor upright poles. Missing pairs are not imputed.
These are fresh cohorts, not silent reuse of the older legacy-only batch.
Existing old manifests remain readable through `--summary`; code hashing will
reject resuming an old batch across this launcher change. Do not modify or
relocate either new batch while it is active. Legacy-only defaults remain
available below; extra seeds are optional, not part of the 84-job experiment.

## Legacy-only experiment

This batch measures the stability of the latest validated sampling-floor
ellipsoids, not fitter randomness. It does not regenerate geometry or switch
texture poles, and does not include sphere or plane trainings.

Default: 7 intervals x 2 mappings x 3 fresh seed-42 trainings = **42 jobs**.
Intervals: W191 0-40, 40-128, 128-198; W254 0-37, 37-81, 86-137, 139-195.
The four active-penalty v1 geometries and three preserved v2 geometries are
the same cohort used by the final comparison report. Legacy texture mapping,
zero ellipsoid runtime margin and 80 epochs remain unchanged. Existing
`deterministic: false` is intentional: first measure current nondeterminism,
rather than change numerical algorithms during the repeatability experiment.

From the activated nag1 environment:

```bash
DRY_RUN=1 bash run_ego_repeatability.sh
bash run_ego_repeatability.sh
```

For a longer batch, choose this **instead of** the default launch:

```bash
DRY_RUN=1 bash run_ego_repeatability.sh --extra-seeds 43 44
bash run_ego_repeatability.sh --extra-seeds 43 44
```

This adds one run per interval/mapping for each extra seed: 70 jobs total.
The summary reports these separately from same-seed runs. Two extra seeds
are exploratory, not a precise estimate of the full seed distribution.
Run jobs sequentially on the same GPU and keep the environment unchanged.
Three repeats give a preliminary range/standard deviation, not a definitive
significance test. Existing historical runs are not silently used as repeats.

Outputs: `runs/2026-09-25_ego_repeatability/JOB_ID/attempt_001`.
The manifest pins job selection, config and geometry hashes, Python training
code, package versions and relevant environment variables. Do not edit code,
configs or geometry during the batch. Resume using the **same command**;
changed batch settings require a different `--output-root`.

Successfully completed jobs are skipped only with success marker, complete
final-frame coverage and finite epoch-80 metrics. A failed job stops the batch;
rerunning preserves the partial attempt and starts a new attempt from scratch.
It does not resume checkpoints. One root is protected against concurrent
launchers with an advisory lock. Do not move output folders until finished.

Read status and metric mean, sample standard deviation, minimum and maximum:

```bash
bash run_ego_repeatability.sh --summary
```

Summary does not launch training and uses the saved manifest regardless of
default seed options. Use `--sequences`, `--modes`, `--repeats` and a separate
`--output-root` for smaller independent batches. Dry run validates all inputs
and prints planned commands without writing files; it does not filter completed
jobs (the real execution does). Preserve all attempts, including failures, for
later review to avoid selecting only favorable outcomes.
