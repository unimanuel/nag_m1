# Matched static-scene DataLoader policy (17 October 2026)

W324 96–161 exhausted the earlier WSL RAM/swap allowance with ten workers.
The user requested four workers for W324 and the remaining static scenes,
while retaining the completed ten-worker W125/W141/W203 recipe. The policy
applies to all five treatments: plane, sphere regular, sphere Visible-UV,
ellipsoid regular and ellipsoid Visible-UV.

| Parent segment | Effective workers | Inclusive frame intervals |
| --- | ---: | --- |
| W125 | 10 | 0–40, 40–93, 93–124, 124–149 |
| W141 | 10 | 2–53, 53–101, 102–173, 174–197 |
| W203 | 10 | 3–58; 60–107 remains blocked by missing depth |
| W324 | 4 | 0–42, 42–96, 96–161, 161–197 |
| W344 | 4 | 0–51, 52–95, 95–135, 135–197 |
| W952 | 4 | 0–63, 64–140, 141–198 |
| W975 | 4 | 175–195 ready; 0–68, 69–99, 99–162 remain blocked by missing depth |

The policy is stored with **full segment IDs**, not display prefixes, in
[`autowork_paper-final_worker_policy.json`](../config/waymo/autowork_paper-final_worker_policy.json).
It applies only to YAMLs in `config/waymo/autowork_paper-final`. Arbitrary
compatible Waymo configurations and all ego YAMLs keep their own settings.

## What changes, and what stays unchanged

The existing preparation YAMLs remain untouched, including their
`num_workers: 10` template value. Launchers explicitly pass `--num-workers 4`
for the four-worker scenes, freeze that override and the policy SHA-256 in
each job, and check the effective saved `training_config.yaml`. Four-worker
job IDs end in `_workers4`. A policy change is rejected before another job
is launched. Ten-worker jobs keep their previous IDs/manifests so completed
W125/W141/W203 outputs can still be validated and resumed unchanged.

This is **not** merely a speed-only change. A CPU-only replay of the existing
random sampler and Lightning worker seeding found different pixel and frame
draws for 44/48 batches with ten versus four workers. The intended sampling
distribution, ray/time counts, epoch count, model, losses, seed, geometry,
legacy texture mapping, zero margin and texture budget are unchanged. Keep
worker settings matched within each interval; document the per-segment policy
in the paper. Four workers reduces loader-memory risk but does not guarantee
that all initialization, training and export stages fit available memory.

No LiDAR extraction, native sphere initialization or geometric fitting needs
repeating. An exact six-file launcher-only compatibility ledger extends the
previous optional-mask validator allowance:
[`final_worker_policy_v1.json`](compatibility/final_worker_policy_v1.json).
It preserves earlier preparation signatures only for the reviewed source
hashes. Model, numerical fitter, dataset and training-kernel edits are not
waived. Both the old ledger and original geometry/certificate/manifests remain
unchanged; the new ledger records actual previous/current hashes separately
from canonical compatibility identities. Explicit per-job worker metadata
distinguishes changed training behavior rather than claiming identical runs.

## Ellipsoid queues: run one at a time

From the activated `nag1` environment at the repository root:

```bash
bash run_final_ellipsoid_W324_4workers.sh
bash run_final_ellipsoid_W344_4workers.sh
bash run_final_ellipsoid_W952_4workers.sh
bash run_final_ellipsoid_W975_4workers.sh
```

These are separate invocations, not one recommended continuous batch. Prefix
any command with `DRY_RUN=1` to inspect commands without training or files.
The queues have 24, 24, 18 and 6 jobs, respectively: three seed-42 repeats of
regular and Visible-UV per ready interval. Repeat the identical command after
a budget stop; completed jobs in that four-worker queue are skipped. The
default nine-hour ellipsoid budget remains an elapsed-time check between jobs,
not a guarantee that an active job finishes within nine hours.

Output roots under `runs/`:

```text
autowork_final_ellipsoid_W324_4workers
autowork_final_ellipsoid_W344_4workers
autowork_final_ellipsoid_W952_4workers
autowork_final_ellipsoid_W975_ready_4workers
```

The user archived W324's earlier jobs as
`runs/autowork_final_ellipsoid_W324_10workers`. Leave that directory intact.
Those runs are not invalid experiments, but they are excluded from the new
four-worker matched cohort. All four W324 intervals are retrained from scratch;
the four completed ten-worker jobs are not counted as four-worker repeats.
Do not point a four-worker command at a ten-worker manifest: it is rejected.

## Future plane and sphere queues

The existing baseline launcher now applies the same policy automatically.
Each interval has nine jobs: plane, sphere regular and sphere Visible-UV,
each with three seed-42 repeats.

Ten-worker scenes:

```bash
bash run_paper_final_baselines_W125.sh
bash run_paper_final_baselines_W141.sh
bash run_paper_final_baselines.sh --intervals W203_3_58
```

Four-worker scenes:

```bash
bash run_paper_final_baselines_W324.sh
bash run_paper_final_baselines_W344.sh
bash run_paper_final_baselines_W952.sh
bash run_paper_final_baselines.sh --intervals W975_175_195
```

Default four-worker baseline roots also end in `_4workers`, such as
`runs/autowork_paper_final_baselines_W324_4workers` and
`runs/autowork_paper_final_baselines_W975_175_195_4workers`. Ten-worker defaults
remain unchanged. Both the audit and dry-run output state effective workers.

The baseline scheduler still uses conservative historical estimates. W324
96–161 has an old single-run estimate above the nine-hour budget; this is not
a reliable prediction for four workers. If scheduling refuses even the next
individual job, inspect timings and explicitly use `--allow-long-job` to allow
one estimated-over-budget job per invocation. Active jobs are never killed.

Do not launch different queues concurrently on the same GPU. Preserve failed
attempts; retries create the next `attempt_NNN` and start from scratch. The
three W975 depth-blocked intervals and W203 60–107 are not silently skipped,
repaired or given a different frame selection. After consistent depth repair,
prepare them normally and launch a separate interval selection; the same
per-segment worker policy applies automatically.

## Implementation validation

The targeted regression suite passed all 101 tests, covering generalized
preparation, final/ego baseline launchers, sampling-floor refinement and
reprojection fitting. All four new shell wrappers passed `bash -n`. Eight
production dry runs planned 72 ellipsoid jobs and 108 plane/sphere jobs with
the correct four-worker flags, without creating their output directories.
The real NAG command-line parser also accepted `--num-workers 4`; its native
sphere-initialization signature was unchanged between four and ten workers.

The existing 14/14 ego compatibility certificate remains accepted. Strict
completed-run validation still accepts all 54 completed W125/W141/W203
ten-worker ellipsoid jobs. A four-worker launch against the archived W324
ten-worker root was correctly rejected, leaving its manifest unchanged.
No preparation, training, checkpoint writing or data repair was performed
for this change. Peak four-worker memory usage, runtime and reconstruction
quality have not yet been measured; lower worker count is a risk reduction,
not a memory guarantee.
