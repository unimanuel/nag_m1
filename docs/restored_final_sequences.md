# Restored-depth final intervals: W203 and W975

Prepared 29 October 2026. These are user-run preparation/training launchers;
Codex did not run fitting or training.

## Input confirmation

All 49 previously missing depth TIFFs are now present and decoded successfully.
Both segments use 784 x 518 float32 TIFFs with finite, nonnegative values and
positive maxima. This is a local readability/numerical check, not a claim that
the files were independently authenticated against the download source.
All four preparation-input audits pass, including RGB, masks, calibration,
object boxes/mapping and decoded LiDAR frame-file availability.

| Interval | Restored frames needed by this interval | New baseline jobs | New ellipsoid jobs |
| --- | --- | ---: | ---: |
| W203 60-107 | 97-99 | 9 | 6 |
| W975 0-68 | 6-9, 58-68 | 9 | 6 |
| W975 69-99 | 69-99 | 9 | 6 |
| W975 99-162 | 99 | 9 | 6 |

Frame 99 is shared by two W975 intervals: 50 interval-file occurrences but
only 49 distinct restored files. W324 96-161 is not selected: its exclusion is
for resources, not missing depth. Already completed W203 3-58 and W975 175-195
are not scheduled again.

## Storage and immutable cohorts

The existing baseline scene folders were renamed by the user to:

```text
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W203
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W975
```

At the user's revised request, new results use four independent folders
directly under `C:\Coding\CGProject\nag1\runs`, not inside those scene folders:

```text
runs/
  autowork_paper_final_baselines_W203/    # original completed cohort unchanged
  autowork_paper_final_baselines_W975/    # original completed cohort unchanged
  autowork_restored_baselines_W203/      # 9 new plane/sphere/sphere-UV jobs
  autowork_restored_ellipsoids_W203/      # 6 new ellipsoid/ellipsoid-UV jobs
  autowork_restored_baselines_W975/      # 27 new plane/sphere/sphere-UV jobs
  autowork_restored_ellipsoids_W975/      # 18 new ellipsoid/ellipsoid-UV jobs
```

The existing scene manifests retain only their original completed jobs.
Do not expand them, overwrite them or treat their summaries as summaries of
the restored cohorts. Future full-scene analysis must read the existing scene
manifest plus both restored manifests and join by full segment/frame identity,
treatment and repeat; it must not pool failed attempts or silently count a job twice.

No restored training outputs existed in the earlier nested locations when
these defaults changed. No existing run was moved or migrated. The new
launchers no longer depend on the old scene folders being present. If an
old nested restored path does exist, the launcher stops instead of silently
starting a duplicate top-level queue. To continue that existing cohort,
explicitly pass its path with `--output-root` for baseline training or
`--training-root` for ellipsoid training on every invocation.

All four new geometries, reference spheres, PLY inspection assets and fitting
diagnostics use a separate namespace, preserving earlier prepared artifacts:

```text
D:\CGProjectData\waymo\ellipsoids\autowork_restored_final_legacy_v1\
  segment-SEGMENT_frames_FIRST_LAST_step1\
```

C: had approximately 89 GiB free when checked. Do not assume all 60 new jobs
will fit there. Archive more unrelated results before exhausting C:, and
check free space between invocations. The defaults intentionally honor the
requested top-level C: run folders rather than silently changing storage. Explicit
native `--output-root` (baseline training) or `--training-root` (ellipsoid
training) overrides are possible, but then keep that exact override on every
retry and summary; never start a duplicate cohort at a second location.

## Commands

Activate `nag1` and run from the repository:

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only input audit/plan; no fitting or output creation:
DRY_RUN=1 bash prepare_missing_final_ellipsoids.sh

# Prepare four fits and their inspection assets. No NAG training:
bash prepare_missing_final_ellipsoids.sh
```

Inspect the prepared PLY/JSON files if desired. Then run one training queue
at a time, rerunning the identical command after a budget stop:

```bash
bash run_missing_final_baselines_W203.sh   # 9 jobs
bash run_missing_final_ellipsoids_W203.sh  # 6 jobs; preparation required first
bash run_missing_final_baselines_W975.sh   # 27 jobs
bash run_missing_final_ellipsoids_W975.sh  # 18 jobs; preparation required first
```

Do not paste all four as one unattended batch unless that combined duration
is intended. All native launchers default to a 12-hour scheduling budget and
never terminate an active training. Baselines predict the next-job duration;
the unchanged ellipsoid driver checks elapsed time between jobs and can
overrun the budget by its last active job. Thus a whole scene queue may need
several invocations. Append `--max-hours 6` for a shorter invocation.

Prefix any training command with `DRY_RUN=1` for native read-only planning.
Ellipsoid planning validates the completed preparation stages and therefore
must be done after preparation. Baseline planning can run before preparation.
Native stage retries preserve incomplete attempts; input/recipe mismatches
fail rather than overwriting another cohort or widening compatibility rules.

Baseline progress summaries, after their restored cohorts have started:

```bash
bash run_missing_final_baselines_W203.sh --summary
bash run_missing_final_baselines_W975.sh --summary
```

The native ellipsoid driver has no `--summary` option. Its normal invocation
validates/skips completed jobs and schedules remaining jobs; do not use that
as a read-only summary command. `DRY_RUN=1` plans but does not launch training.

The former `_ready` wrappers still default to the old, pre-rename folder
names. To read **only the original** completed baseline jobs without creating
duplicate old queues, explicitly use the renamed parent roots:

```bash
bash run_paper_final_baselines_W203_ready.sh --summary \
  --output-root "$PWD/runs/autowork_paper_final_baselines_W203"
bash run_paper_final_baselines_W975_ready.sh --summary \
  --output-root "$PWD/runs/autowork_paper_final_baselines_W975"
```

## Unchanged experimental recipe and compatibility

There are 36 baseline jobs (plane, regular sphere, Visible-UV sphere) and
24 ellipsoid jobs (regular and Visible-UV), all three fresh seed-42 repeats.
Ten workers, epoch 80, final-only checkpoints, nominal 960 x 1920 texture
budget, sampling, losses and foreground settings remain unchanged.

Preparation delegates to the released generic pipeline: corrected-Y-up
raw-v4, original reprojection/depth fitting and legacy regular-mapping
sampling-floor v2. The existing raw-v4 interpreter and initialization-only
sphere reference export remain unchanged. No upright poles, clipping,
compactness, radius-scale penalty or automatic axis enlargement is added.
Runtime ellipsoid margin remains zero.

The new shell filenames intentionally do not match the historical
`run_paper_final_baselines*.sh` provenance glob. No numerical, training,
preparation or historical launcher code is changed, so no compatibility-ledger
waiver or historical manifest rewrite is necessary. The shell wrappers only
choose explicit intervals and independent output roots.

## Implementation validation

All 49 restored TIFFs decoded and all four preparation-input audits passed.
The real preparation dry run validated four jobs and the existing release
certificate without creating references or fitting artifacts. Both real
baseline dry runs passed, planning nine W203 and 27 W975 jobs with ten workers
and the requested output paths. The ellipsoid training wrappers passed
native-argument forwarding tests; real ellipsoid planning must wait for the
four new geometry fits and was not claimed as completed here.

The combined preparation, numerical-kernel, baseline and launcher unit suites
passed 122 tests; shell syntax and whitespace checks passed. The revised
launcher tests cover top-level defaults, absent old scene directories, and
duplicate prevention with both explicit native output-option spellings. Before/after
SHA256 checks confirmed both renamed scene manifests and the complete
historical training context remain identical. No restored run folders or new
preparation outputs were created. No real-data fitting or training was run.
