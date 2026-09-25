# Seven-interval training repeatability

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
