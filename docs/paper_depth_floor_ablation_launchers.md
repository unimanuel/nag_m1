# Depth plus sampling floor (no reprojection)

This missing control starts from the **completed matched D0 depth-only fits**
for all seven ego intervals. It applies the same frozen legacy sampling-floor
kernel as D2, but with reprojection weight **exactly zero throughout**. It does
not initialize from D1 or retain a reprojection term during refinement.

The objective is `0.15 * robust_log_depth_loss + w * sampling_floor_loss`, with
continuation weights `w = 5, 10, 20`, threshold 0.5, Huber transition 0.2, and
up to 4,096 sampling rays. The 2% degradation budget applies to **depth-only**
loss, with the historical 1e-6 numerical allowance. Shared safety remains
camera normalized radius <=0.65, conservative clearance >=6 m, minimum radius
>=8 m, and aspect ratio <=10. Center and log radii can refine using SLSQP;
rotation stays fixed. Regular mapping, legacy poles, zero runtime margin,
no clipping, no radius-scale/compactness prior, and no axis enlargement.
The sphere is the existing sampling-density reference, not a clipping bound.

The supported cross-frame pair validity/visibility constraints remain for
comparability with D2. Pixel reprojection residuals are exported as diagnostics
only: their coefficient is zero, not an epsilon. This is therefore a loss
ablation, not removal of the observation-building/visibility pipeline.

## Run from the NAG repository in WSL

Optional read-only audit (no fitting, training, or new output directories):

```bash
DRY_RUN=1 bash prepare_paper_depth_floor.sh
```

Prepare all seven intervals and inspect the exported blue ellipsoid PLYs:

```bash
bash prepare_paper_depth_floor.sh
```

Preparation outputs, each under `<interval>/depth_floor/attempt_001/`:

```text
D:\CGProjectData\waymo\ellipsoids\autowork_paper_depth_floor_v1\
    <interval>\depth_floor\attempt_001\ellipsoid.json
    <interval>\depth_floor\attempt_001\ellipsoid.ply
```

Each attempt also includes front/background points and camera centers, depth
and sampling-floor diagnostics, source hashes, and exact runtime equivalence
metadata. `matrix.json` records three seed-42 repeat cells per interval and
whether they reuse existing D0 runs. Historical files remain untouched.

Then run these **sequentially**, not simultaneously:

```bash
bash run_paper_depth_floor_part1of2.sh
bash run_paper_depth_floor_part2of2.sh
```

| Batch | Intervals covered | Expected new trainings | Buffered estimate |
| --- | --- | ---: | ---: |
| 1 | W191 0–40, 40–128, 128–198 | 3 (128–198) | 1.2 h |
| 2 | W254 0–37, 37–81, 86–137, 139–195 | 6 (37–81, 139–195) | 4.6 h |

Estimates use the slowest completed matched D0 repeat per interval plus 35%;
they are not runtime guarantees. Preparation is separate CPU fitting. The
scheduling budget defaults to eight hours per invocation; use `--max-hours`
to override. A budget stop happens before a new training, not mid-training.
Run the same script again after a stop or interruption; completed valid
attempts are skipped and partial/failed outputs are preserved.

Each changed interval gets **three regular seed-42 repeats**, ten workers,
80 epochs, matched training settings and final-only checkpoints. Output is
streamed to the console and retained in per-attempt logs, using the existing
validated ablation training runner. There is no Visible-UV sweep here.

New trainings go to:

```text
C:\Coding\CGProject\nag1\runs\autowork_paper_depth_floor_v1\
```

## Reuse is based on identity, not quality

The read-only audit of current D0 fits finds exactly zero full-ray floor loss
on W191 0–40, W191 40–128, W254 0–37 and W254 86–137. The generic v2 rule
preserves their arrays exactly, so their twelve D0 trainings are reused.
The three other intervals have positive floor loss and undergo refinement.
If refinement returns the original or the effective float32 geometry is
identical, those D0 repeats are also reused; therefore **nine is an upper
bound**, not a promise to run unnecessary repeats. Reused results must not
be counted as additional independent repetitions.

Both original D0 manifests, preparation stage signatures, inspection assets,
observation/configuration pins, completed epoch-80 results, reference mapping
and current training environment must validate. A conflict stops the script;
it does not overwrite source files, loosen safety or silently choose another
attempt. The new recipe has its own hash-pinned manifests and stage outputs.

Status after preparation (read-only, no training):

```bash
bash run_paper_depth_floor.sh --summary
```

To preview one training queue after preparation:

```bash
DRY_RUN=1 bash run_paper_depth_floor_part2of2.sh
```

The scripts expose `--d0-training-roots`, `--preparation-root`, `--training-root`
and training-only `--intervals` for explicit alternatives. Changed inputs or
code require a fresh `autowork_` namespace. The raw-PCA exclusion of W191
40–128 does **not** apply: its D0 and depth-plus-floor geometries are safe.
