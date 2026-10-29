# Matched non-ego final-sequence paper baselines

Prepared 6 October 2026. No training was started by Codex.

**29 October update:** all 49 missing depth TIFFs have been restored and the
four formerly blocked input audits now pass. The original selected 20-interval
five-treatment cohort is complete (300 trainings). Use the
[restored-interval launchers](../docs/restored_final_sequences.md) to add 60
jobs inside the renamed W203/W975 scene folders, with separate immutable
child manifests. The historical missing-input table below records the former
gaps, not the current readiness. W324 96-161 remains a resource exclusion.

**20 October update:** all 120 selected ellipsoid trainings are complete across
20 intervals. Matched static sphere/plane baselines have not yet been trained.
Use the [current comparison and seven-queue handoff](static_final_ellipsoid_findings_2026-10-20.md)
for results, ready-only W203/W975 wrappers, storage limits and D: output options.

**18 October update:** all selected final scenes again use ten workers for every
background treatment. W324 96–161 is excluded for runtime/resource reasons,
not reconstruction quality, from the default ellipsoid and baseline queues.
Four-worker results remain separate. Source YAMLs and preparation geometry are
unchanged. See [the current restart instructions](../docs/final_ten_worker_restart.md).

## What existed, and why fresh results are needed

`config/waymo/final` contains **25 intervals from seven parent segments**:
25 regular sphere YAMLs, 25 plane YAMLs, 25 double-resolution sphere YAMLs,
and only 10 Visible-UV YAMLs. The plane/regular pairs already differ only in
background class and name. The existing Visible-UV variants use the approved
`[0.05, 0.05]` margins and nine interior samples per axis.

However, these are not fully matched to the recent ego paper cohort:

- Most use 200 x 70,000, 280 x 50,000, 350 x 40,000, or 467 x 30,000
  batches/rays, rather than the selected ego W254 0-37 reference's
  **140 x 100,000**.
- Those products are approximately 14 million sampled rays per epoch, but
  different batch counts mean different numbers of optimizer updates. Equal
  total samples do not make the training recipes equivalent.
- The old final files use four workers and less frequent intermediate plotting
  and geometry exports; the ego template uses ten workers and its existing
  plotting/export cadence.
- Fifteen intervals have no Visible-UV YAML.

A read-only scan of dated run folders found **48 historical run directories**:
42 passed successful-exit, full-frame-output and finite epoch-80 metric checks;
six were incomplete. Only **11 of the 25 intervals** have any completed
historical result. This inventory includes archival double-resolution and
early implementation variants, not 42 matched paper baselines.

The most relevant older locations are:

```text
runs/2026-08-05_sphvisuv_dubres_regular
runs/2026-07-18_dubres_planecenters
runs/2026-08-18_ellipsoid_static_compare
```

None of these metrics is automatically reused as a fresh matched result.
Historical completed plane/sphere runs with a 960x1920 grid contribute only
timing estimates. Double-resolution and incomplete attempts are excluded from
timing selection. Some intervals have no historical timing: their scheduling
estimate is explicitly a heuristic, not a measured runtime.

## New configurations and treatments

The **75 new YAMLs** live in `config/waymo/autowork_paper-final`, preserving
all historical `config/waymo/final` files and earlier manifests unchanged.
They copy the established large-ego W254 0-37 template, replacing only the
segment data path, inclusive frame interval, run name and intended background
selection. Final-only checkpoint storage is explicit.

**Important cross-dataset qualification:** the ego cohort itself does not have
one universal batch/ray combination. W191 0-40, W191 128-198 and W254 0-37 use
140 x 100,000; W191 40-128 uses 156 x 90,000; W254 86-137 uses 200 x 70,000;
W254 37-81 and 139-195 use 280 x 50,000. All other non-dataset YAML settings
match the selected W254 0-37 template. The new final cohort standardizes on
that reference rather than pretending to match every ego interval's sampler
exactly. Within each new interval, all three treatments are fully matched.
Report this batch policy in the paper; do not describe the entire ego/final
collection as having identical optimizer-update counts across all intervals.

Each interval has these three initial treatments:

1. View-dependent plane, using the existing middle-trajectory anchoring.
2. Regular sphere, with radius calculated from visible foreground-plane centers.
3. Sphere Visible-UV, using that same radius policy and the approved allocation.

All use 80 epochs, seed 42, nondeterministic training, 140 batches x 100,000
rays x 20 sampled timestamps, the same losses/phases/networks and foreground
settings, and the nominal `[960, 1920]` background texture budget. No doubled
resolution or upright mapping treatment is introduced. The three repeats per
treatment measure same-seed nondeterminism, not independent-seed variability.

Each interval has nine jobs. All 25 would therefore require **225 trainings**
for these three treatments. There are 21 depth-ready intervals, but W324 96–161
is deferred for resources: the selected 20 intervals require **180 baselines**.
Their 120 ellipsoid/Visible-UV trainings are now complete. The matched baseline
jobs will complete the five-treatment comparison for this selection.

## Readiness and missing inputs

The real NAG loader parsed all 75 YAMLs. A CPU-only `NAGDataset` smoke test
independently verified exact frame indices for all 21 ready intervals and
reproduced the following four failures. No runner/model build, rendering,
checkpoint loading or training was performed.

| Interval | Missing depth TIFF frame indices |
| --- | --- |
| W203 60-107 | 97, 98, 99 |
| W975 0-68 | 6-9 and 58-68 |
| W975 69-99 | 69-99, all selected frames |
| W975 99-162 | 99 |

These files belong under
`data/datasets/waymo/segment-SEGMENT/depth/INDEX.tiff`, not the LiDAR pointcloud
folder. All selected RGB images and masks are present. Boxes are object-level
trajectories, **not** one file per frame. Missing optional mask-filter JSONs
are handled by the existing loader and are not missing-depth failures.

The launcher refuses a selection containing a blocked interval. It does not
drop frames, fill depth gaps, alter the dataset, or silently exclude an interval.
Prefer restoring the original TIFFs from a backup. Do not simply generate
individual replacement images with an arbitrary depth scale: the existing
monocular depth script normalizes across an entire sequence and overwrites
outputs. Repair/regeneration requires a separate, consistent preparation step.

## Commands

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only audit and archival inventory:
bash run_paper_final_baselines.sh --audit-only --historical

# Optional preview: validates inputs and prints commands, without training/files.
DRY_RUN=1 bash run_paper_final_baselines_W125.sh

# Start with W125; one invocation has a twelve-hour scheduling budget.
bash run_paper_final_baselines_W125.sh

# Check status. Rerun the same launcher until all 36 W125 jobs are complete.
bash run_paper_final_baselines_W125.sh --summary
```

The selected complete-input intervals use these seven queues. Run **one at a time**,
not together with another training script:

```bash
bash run_paper_final_baselines_W125.sh
bash run_paper_final_baselines_W141.sh
bash run_paper_final_baselines_W203_ready.sh
bash run_paper_final_baselines_W324.sh
bash run_paper_final_baselines_W344.sh
bash run_paper_final_baselines_W952.sh
bash run_paper_final_baselines_W975_ready.sh
```

Each launcher selects all its intervals and three repeats in all three modes.
W125, W141 and W344 each contain 36 jobs; W324 and W952 contain 27. The W324
wrapper excludes 96–161 across all three baseline treatments. These are
**resumable queues**, not a promise that a whole segment finishes in one
invocation. The driver stops between jobs when elapsed time plus the next
buffered estimate exceeds twelve hours. Rerun the same script to continue.
`--max-hours 6` requests an earlier scheduling stop.

**Historical W324 timing:** the now-excluded 96–161 regular-sphere job took
about 10.15 hours before a 35% allowance. The current wrapper does not schedule
that interval. For any other estimated-over-budget job, the driver still
requires explicit opt-in:

```bash
bash run_paper_final_baselines_W324.sh --allow-long-job
```

That option permits at most one estimated-over-budget start per invocation.
It never kills an active training. Once a treatment has completed a new matched
repeat, subsequent repeats use its observed durations for scheduling rather
than the older estimate; the original timing provenance remains in the manifest.

Until the four depth gaps are repaired, use explicit ready subsets for the
other two segments, each with nine jobs:

```bash
bash run_paper_final_baselines.sh --intervals W203_3_58
bash run_paper_final_baselines.sh --intervals W975_175_195
```

The equivalent `_W203_ready.sh` and `_W975_ready.sh` wrappers above freeze these
same explicit selections and default roots; they do not expand after depth repair.

After consistent input restoration, add only the missing intervals:

```bash
bash run_paper_final_baselines.sh --intervals W203_60_107
bash run_paper_final_baselines.sh --intervals W975_0_68 W975_69_99 W975_99_162
```

Do not later run the entire W203/W975 segment wrapper if you want to avoid
duplicating the already completed subset: a different selection owns a
different output manifest. The segment wrappers are provided for the case
where all inputs are restored before any subset training begins.

## Output integrity and retries

Default segment outputs:

```text
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W125
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W141
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W324_without_96_161
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W344
C:\Coding\CGProject\nag1\runs\autowork_paper_final_baselines_W952
```

An explicit interval selection uses the same prefix followed by the joined
interval labels. Child directories identify interval, treatment, seed and
repeat; attempts are `attempt_001`, `attempt_002`, etc. Completed jobs are
validated and skipped. Failed/partial attempts are retained and the next
attempt starts from scratch, not a checkpoint. Nonempty unowned output folders
and changed manifests are rejected. Only final checkpoints are written.

The cohort freezes code/configuration hashes, interpreter/package/environment
settings and the completed ego reference manifest. Per-selection manifests
record data-file sizes/mtimes and historical timing evidence. A shared lock
prevents two final launchers running together. Do not change code/configs or
selected dataset inputs during a selection. Restoring blocked inputs in a
different, not-yet-started selection is allowed; existing result manifests are
not rewritten. Summary commands do not start training.

## Ellipsoid stage and scientific interpretation

Ellipsoid preparation has been generalized and verified against all seven ego
fixtures. The 20 selected static intervals now use corrected-Y-up raw-v4
initialization, original reprojection/depth fitting and the unchanged generic
legacy regular-mapping sampling-floor v2 preservation/refinement policy.
Native sphere references were exported by initialization only; a completed
sphere training is not required to fit the ellipsoid. All 120 ellipsoid
trainings have passed completion, effective-settings and geometry checks.
No earlier PCA/AR3 fit or unrelated ego geometry was substituted.

The non-ego group is genuinely mostly near-static: for example W125 40-93 has
about 0.013 m cumulative camera motion, and W324 96-161 about 0.008 m. Some
intervals have brief motion (W125 0-40 ~1.99 m, W141 174-197 ~1.97 m, W952
0-63 ~2.86 m). Reprojection fitting will be weakly informative for the near-static
intervals; depth and the recipe's generic safety/stability terms will matter.
This should be documented and inspected before treating the ellipsoid optimizer
as a well-observed parallax fit. Keep the recipe generic, without per-sequence
hand-selected geometry.

These extra segments are useful held-out evidence for background representation,
but a gain in a near-static interval is not by itself evidence of reduced
multi-view parallax/learned-flow strain. Like the ego experiments, this is a
matched **background-treatment** comparison: native initialization, coverage,
parameterization and compute cost may still differ across background classes.
