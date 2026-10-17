# Final scenes: ten workers, W324 96–161 excluded (18 October 2026)

All selected static/final intervals use ten DataLoader workers again, matching
the original YAML recipe. W324 96–161 is deferred for runtime/resource reasons
across all five backgrounds: plane, sphere regular/Visible-UV and ellipsoid
regular/Visible-UV. This is not a quality-based exclusion. Keep the same interval
set in paper averages and disclose the exclusion and four missing-depth gaps.

No geometry, observations, source YAMLs, completed runs or manifests change.
The default scheduling budget remains twelve hours; active jobs are not killed.
Four-worker runs remain separate and must not be pooled with ten-worker repeats.
The old `_4workers.sh` wrappers stop with migration instructions, rather than
silently starting ten-worker jobs in misleading four-worker folders.

## Resource evidence and limits

Saved W324 42–96 timings were 75.3 minutes regular / 134.4 minutes Visible-UV
with ten workers, versus 72.6 / 163.8 with four. Thus ten workers did not improve
every job; the Visible-UV difference alone is not a controlled worker-speed
benchmark. There is no proof that worker count caused all of 96–161's delay.

The following counts come from foreground mask values actually present in each
selected interval, not overlap-layer filenames or whole-segment box mappings:

| Interval | Frames | Foreground mask IDs |
| --- | ---: | ---: |
| W324 42–96, completed reference | 55 | 24 |
| W324 96–161, excluded | 66 | 23 |
| W324 161–197 | 37 | 16 |
| W344 0–51 / 52–95 / 95–135 / 135–197 | 52 / 44 / 41 / 63 | 22 / 14 / 12 / 9 |
| W952 0–63 / 64–140 / 141–198 | 64 / 77 / 58 | 19 / 15 / 13 |
| W975 175–195 | 21 | 2 |

Fewer foreground masks support trying ten workers again, but they do not bound
RAM, runtime, initialization caches, image exports or CUDA memory. W952 64–140
has more frames than the excluded interval. Other jobs may still need attention;
no peak-memory benchmark or new training was run to certify their resource use.
The earlier ten-worker OOM also occurred before the WSL memory allowance was
increased. Do not change cache, sampler or foreground settings to hide failures.

## Resume the original W324 ten-worker cohort

Four completed ten-worker jobs for 0–42 and 42–96 remain valid references.
They are archived on D: after the recent storage cleanup. To keep important
paper results on C: and avoid rerunning them, first copy that archive back:

```bash
rsync -rt --ignore-existing \
  /mnt/d/CodingD/CGProjectD/nagRunsFromC/autowork_final_ellipsoid_W324_10workers/ \
  /mnt/c/Coding/CGProject/nag1/runs/autowork_final_ellipsoid_W324_10workers/
```

This is a user-run copy of roughly 22 GB; the D: archive is not removed and
existing C: files are not overwritten. Do not substitute the four-worker root.
Nothing has been restored automatically. If the ten-worker cohort is not
restored, the new empty C: root will train all eighteen eligible jobs afresh.

Then, in `nag1` at the repository root:

```bash
DRY_RUN=1 bash run_final_ellipsoid_W324_10workers.sh
bash run_final_ellipsoid_W324_10workers.sh
```

The full original manifest still contains 24 planned jobs, including six for
96–161. The explicit `--skip-intervals W324_96_161` option schedules none of
those six jobs and never validates them as completed results. Eighteen jobs
are eligible; the four validated archived completions leave fourteen trainings.
The original manifest is not rewritten to claim a different historical cohort.
All attempts, including the failed/incomplete excluded interval, remain intact.

## Other ellipsoid queues, one at a time

```bash
bash run_final_ellipsoid_W344_10workers.sh
bash run_final_ellipsoid_W952_10workers.sh
bash run_final_ellipsoid_W975_10workers.sh
```

These contain 24, 18 and 6 jobs, respectively: three seed-42 repeats of each
regular/Visible-UV treatment. New roots are `autowork_final_ellipsoid_W344_10workers`,
`autowork_final_ellipsoid_W952_10workers` and
`autowork_final_ellipsoid_W975_ready_10workers` under `runs/`.
Existing W125/W141/W203 jobs remain ten-worker jobs and need no rerun.
No point-cloud extraction or ellipsoid preparation is needed.

## Future plane/sphere queues

Use the existing launchers; the shared policy now selects ten workers:

```bash
bash run_paper_final_baselines_W324.sh
bash run_paper_final_baselines_W344.sh
bash run_paper_final_baselines_W952.sh
bash run_paper_final_baselines.sh --intervals W975_175_195
```

The W324 baseline wrapper explicitly selects only 0–42, 42–96 and 161–197,
with 27 jobs and the separate root
`runs/autowork_paper_final_baselines_W324_without_96_161`.
Other ready baseline queues have 36, 27 and 9 jobs respectively. All five
treatments must use ten workers within an interval. The four known depth-blocked
intervals remain blocked; no frames are silently repaired or omitted within an
interval. The default W324 exclusion does not remove its YAML or geometry files.

Run queues sequentially on the GPU. Rerun the identical command after a budget
stop; validated completed jobs are skipped. Four-worker roots are neither
renamed, deleted nor resumed by these scripts. Exact source-hash compatibility
continues to protect existing preparation certificates and run manifests.

## Validation

All 107 targeted regression tests passed, including exact policy checking,
four/ten-worker separation, train-only exclusions, unknown-exclusion rejection,
skip-before-completion-validation behavior and unchanged full-cohort manifests
across retries. Shell syntax checks passed for all nine changed/new launchers;
all four retired wrappers returned their intended exit code 2 without launching.

Eight production dry runs planned 66 eligible ellipsoid jobs and 99 baseline
jobs at ten workers. The archived W324 dry run accepted the original full
manifest and omitted all six 96–161 jobs. Strict real-result validation accepted
all 54 W125/W141/W203 completions plus the four eligible archived W324 completions,
leaving fourteen W324 jobs. Existing preparation provenance and the release
certificate remained accepted. No GPU training, geometric fitting, automatic
archive restoration, result deletion or historical-manifest rewrite occurred.
