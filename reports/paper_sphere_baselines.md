# Matched paper sphere baselines

These launch **42 fresh trainings**: seven ego intervals, regular and Visible-UV
sphere textures, and three seed-42 repeats per interval/layout. Historical sphere
results are references, not silently substituted repeats. Plane training and
upright/ellipsoid fitting are outside this batch.

The same YAMLs used by the completed legacy-pole ellipsoid repeats are hash
checked. Every run uses the visible-foreground-plane-center sphere radius rule,
80 epochs and `deterministic: false`. Same-seed repeats measure nondeterminism,
not independent initialization seeds. Only the final checkpoint is saved.

## Five batches

Historical durations are taken from the selected sphere references' successful
exit records. Estimates include a **35% slowdown allowance**.

| Script | Intervals | Jobs | Historical total | Buffered estimate |
| --- | --- | ---: | ---: | ---: |
| `run_paper_sphere_baselines_part1.sh` | W191 0-40, 128-198 | 12 | 3.66 h | 4.94 h |
| `run_paper_sphere_baselines_part2.sh` | W254 0-37, 86-137 | 12 | 4.19 h | 5.65 h |
| `run_paper_sphere_baselines_part3.sh` | W191 40-128 | 6 | 3.65 h | 4.92 h |
| `run_paper_sphere_baselines_part4.sh` | W254 37-81 | 6 | 3.49 h | 4.71 h |
| `run_paper_sphere_baselines_part5.sh` | W254 139-195 | 6 | 3.46 h | 4.67 h |

Each estimate is below six hours on the historical machine. Actual wall time
cannot be guaranteed. By default an invocation stops **between trainings** if
elapsed time plus the buffered next-job estimate would exceed six hours. It
never kills an active training; an unexpectedly long active run can exceed that
estimate. Rerunning the identical wrapper finishes the remaining jobs. Use
`--max-hours 4` to request an earlier scheduling stop, without changing the cohort.

## Run

Activate the same `nag1` environment used for the ellipsoid repeats. No Google
login, point-cloud extraction or ellipsoid preparation is required.

```bash
cd /mnt/c/Coding/CGProject/nag1
conda activate nag1

DRY_RUN=1 bash run_paper_sphere_baselines_part1.sh
DRY_RUN=1 bash run_paper_sphere_baselines_part2.sh
DRY_RUN=1 bash run_paper_sphere_baselines_part3.sh
DRY_RUN=1 bash run_paper_sphere_baselines_part4.sh
DRY_RUN=1 bash run_paper_sphere_baselines_part5.sh
```

Dry runs validate inputs and print commands without creating folders or training.
Run whichever part fits the current session; run parts **sequentially**, not on
the GPU at the same time:

```bash
bash run_paper_sphere_baselines_part1.sh
bash run_paper_sphere_baselines_part2.sh
bash run_paper_sphere_baselines_part3.sh
bash run_paper_sphere_baselines_part4.sh
bash run_paper_sphere_baselines_part5.sh
```

The blocks above list separate invocations, not a recommended unattended
five-part command exceeding the requested session length. Any part may be run
first. `PYTHON_BIN` selects the interpreter; it must match the completed cohort.

## Outputs, status and retries

For each part, default result root:

```text
C:\Coding\CGProject\nag1\runs\autowork_paper_sphere_baselines_part1of5
C:\Coding\CGProject\nag1\runs\autowork_paper_sphere_baselines_part2of5
C:\Coding\CGProject\nag1\runs\autowork_paper_sphere_baselines_part3of5
C:\Coding\CGProject\nag1\runs\autowork_paper_sphere_baselines_part4of5
C:\Coding\CGProject\nag1\runs\autowork_paper_sphere_baselines_part5of5
```

Child layout: `autowork_W191_0_40_sphere_regular_same_seed_seed42_repeat01/attempt_001`.
Corresponding `visuv` and `repeat02`/`repeat03` folders are separate fresh runs.
The driver reverses regular/visuv order on the second repeat.

```bash
bash run_paper_sphere_baselines_part1.sh --summary
```

Substitute the part number for any of the five summaries. A shared lock prevents
concurrent sphere parts; immutable per-part and shared-cohort manifests pin code,
environment, YAMLs, source sphere diagnostics and the ellipsoid comparison
manifests. Do not edit code/configurations, move results, or change the environment
while the cohort is in progress. An existing nonempty folder without the expected
manifest is rejected. An audited storage-only change relative to the older
ellipsoid repeats disables intermediate checkpoints; model/training changes
outside that known revision are rejected.

Rerun the same script after interruption. Completed jobs are skipped only with
success marker, all final frames, finite epoch-80 metrics and matching sphere
center/radius, texture resolution and UV mode. Failed attempts remain intact;
retry writes a new `attempt_002`, starting from scratch, never from a checkpoint.

The existing paper folder-location TXT describes **results completed before
this new batch**, so it does not yet list these folders as completed results.
