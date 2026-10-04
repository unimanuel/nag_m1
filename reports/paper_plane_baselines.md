# Matched paper plane baselines and configuration audit

Prepared 4 October 2026. These launch **21 fresh plane trainings**: the seven
ego intervals, three seed-42 repeats each. There is one planar texture layout;
there is no duplicate "plane Visible-UV" treatment. Plane repeats are compared
with both regular and Visible-UV curved backgrounds.

No training was launched during implementation. The read-only configuration
audit and both real-input dry runs passed. Existing YAMLs, completed results,
geometry files, and previous manifests were not edited.

## Two batches

| Script | Intervals | Jobs | Conservative historical estimate | With 35% allowance |
| --- | --- | ---: | ---: | ---: |
| `run_paper_plane_baselines_part1.sh` | W191 0-40, 40-128, 128-198; W254 0-37 | 12 | 6.08 h | 8.21 h |
| `run_paper_plane_baselines_part2.sh` | W254 37-81, 86-137, 139-195 | 9 | 6.35 h | 8.57 h |

For each interval, scheduling uses the **maximum** duration among the three
fresh regular sphere repeats and available successful historical plane runs
in `2026-08-19_ellipsoid_non_center` and `2026-06-11-13-15-combi`. Both older
bbox variants may contribute conservative timing estimates, but their metrics
and settings are never reused as matched baseline results. The selected timing
sources and all candidates are recorded in the new manifests.

Ten hours is an estimate, not a guarantee. The driver stops **between jobs**
when elapsed time plus the buffered next-job cost exceeds ten hours. It does
not kill an unexpectedly slow active training. Rerun the same script to finish
remaining jobs. `--max-hours 8` requests an earlier scheduling stop without
changing the cohort. Run the parts sequentially, not concurrently.

## Run

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1

# Optional read-only audit of all three families:
bash run_paper_plane_baselines.sh --audit-only

# Optional dry-run checks, without output folders or training:
DRY_RUN=1 bash run_paper_plane_baselines_part1.sh
DRY_RUN=1 bash run_paper_plane_baselines_part2.sh
```

Start one part per session:

```bash
bash run_paper_plane_baselines_part1.sh
```

Then, in a separate session:

```bash
bash run_paper_plane_baselines_part2.sh
```

No Google login, point-cloud extraction or ellipsoid preparation is necessary.
Only the final checkpoint is saved. Every repeat starts from scratch, not a
checkpoint. All repeats use seed 42 and `deterministic: false`, matching the
completed sphere and ellipsoid repeat cohorts; these measure nondeterminism,
not independent-initialization-seed variation.

## Configuration consistency: verified, no YAML changes needed

The scope is **all 21 `config/waymo/large-ego` YAMLs used for these seven paper
intervals**, not unrelated static-camera or archival configurations elsewhere.

For every interval, the regular YAML is
`segment-SEGMENT_FIRST_LAST_large-undistorted.yaml`, with `_bgplane.yaml` and
`_sph_visuv.yaml` variants. The two parent segments are 1918764220984209654 and
2547899409721197155.

- Plane versus regular sphere differs **only** in `background_plane_type` and
  `name`. Both are view-dependent backgrounds.
- Visible-UV versus regular differs **only** in `name` and the added
  `background_plane_kwargs.sphere_visible_uv_texture` settings: enabled,
  margins `[0.05, 0.05]`, and nine interior samples per axis.
- All share the nominal background color-grid budget `[960, 1920]` (1,843,200
  texels), not doubled resolution. Visible-UV reallocates that nominal budget;
  integer tile dimensions can introduce rounding.
- Deep comparison verifies all other YAML fields, including data paths,
  undistorted calibration, frame selection, image/mask/depth processing,
  foreground/bounding-box setup, network and encoding configurations, losses,
  learning schedules, epoch count, seed, evaluation and plotting settings.
- Ellipsoid runs use the **same regular/Visible-UV YAMLs**, not separate
  ellipsoid YAMLs. The completed launcher overrides the background class,
  supplies the saved sampling-floor geometry, sets runtime margin to zero,
  and selects legacy poles. Those are intentional treatment differences.

The preflight additionally checks all **42 completed sphere and 42 completed
legacy-pole ellipsoid runs**, not merely the current files. It verifies successful
exits, all final frames, finite epoch-80 metrics, saved effective settings,
YAML/geometry provenance hashes, sphere geometry, interpreter/package versions,
and training-code compatibility. Saved sphere settings are also checked across
repeats and texture layouts. Only expected geometry selection, UV allocation,
run metadata, and the audited intermediate-checkpoint storage policy may differ.

This is a controlled **background-treatment comparison**, not a claim that
every implementation detail is identical. Physical background geometry,
parameterization, ray intersections and native texture initialization differ.
The current plane implementation is anchored at the middle trajectory frame;
no plane implementation change is introduced by these launchers. The same
nominal neural configuration and texture budget do not guarantee identical
sampling density, texture coverage or FLOPs for all geometries.

## Outputs, status and safe retries

```text
C:\Coding\CGProject\nag1\runs\autowork_paper_plane_baselines_part1of2
C:\Coding\CGProject\nag1\runs\autowork_paper_plane_baselines_part2of2
```

Child pattern:
`autowork_W191_0_40_plane_same_seed_seed42_repeat01/attempt_001`.
The other two repeats have `repeat02` and `repeat03` names.

```bash
bash run_paper_plane_baselines_part1.sh --summary
bash run_paper_plane_baselines_part2.sh --summary
```

Interrupted/failed attempts remain intact. Retrying writes `attempt_002`,
starting over, and skips validated completed jobs. Existing nonempty output
folders with missing/different manifests are rejected. A shared plane-batch
lock prevents concurrent parts. Manifests freeze code, environment, source
result/configuration hashes, and the audit/timing evidence across both parts.
Do not edit code/configs, move source results, or change the environment while
the plane cohort is in progress. Existing sphere/ellipsoid manifests are never
rewritten to accommodate the new launcher; their summary commands remain usable.
