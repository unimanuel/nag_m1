# Controlled sampling-floor replay

This CPU-only diagnostic isolates the W191 0-40 numerical discrepancy without
rebuilding observations, initializing sphere models or running NAG training.
It does not change tolerances, historical data, or the preparation release gate.

Three arms use identical saved observations, the original reprojection
initializer, diagnostic-precision sphere/mapping and frozen optimizer settings:

1. Historical v1 source, exactly as stored at `73e3874`.
2. Historical v2 source at `07cbd74`, including the exact-zero preservation check.
3. The generalized v2 adapter around that same pinned historical kernel.

The shared numerical helper sources must match both historical revisions.
Each arm runs three times in a fresh Python process; order rotates between
repeats. Workers load all arms before reading arrays, use the same interpreter,
one CPU core, one thread per numerical library and `PYTHONHASHSEED=0`. Runtime
build information, actual thread pools, CPU features, array hashes/layout and
alignment are recorded. No GPU/training framework may be loaded by a worker.

```bash
conda activate nag1
cd /mnt/c/Coding/CGProject/nag1
DRY_RUN=1 bash run_controlled_ellipsoid_replay.sh
bash run_controlled_ellipsoid_replay.sh
```

Default output:

```text
D:\CGProjectData\waymo\diagnostics\autowork_controlled_floor_replay_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_ENGINE_repeatNN\attempt_001\
    result.json
    console.txt
```

The experiment uses nine refinements, not nine trainings. The previous stage
timestamps suggest a roughly 10-second refinement, so the focused run should
take around two minutes on this machine, including subprocess setup. Timing is
an estimate. Dry-run validates source/input hashes and probes the runtime but
does not fit or write files. Valid completed jobs are reused; incomplete
attempts remain intact. Changed code, inputs, runtime or selection require
another `--output-root` whose name begins with `autowork_`.

For an independent second batch, use a fresh root rather than reusing the first:

```bash
bash run_controlled_ellipsoid_replay.sh \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_controlled_floor_replay_v1_batch2
```

Optional control interval:

```bash
bash run_controlled_ellipsoid_replay.sh --sequences W191_0_40 W254_139_195 \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_controlled_floor_replay_v1_with_control
```

`summary.json` separates within-version variation, between-version differences
and departures from the stored historical optimum. Identical repeats would
show determinism in this controlled environment, not prove historical runtime
identity or explain a stable offset from the stored optimum. A v1/v2 difference
with v2/adapter agreement would implicate the preservation-check execution path,
not the generalized wrapper. Very similar objectives alone do not waive a
failed geometry comparison. The same original tolerances are reported.

The standalone root-level diagnostic is intentionally outside the NAG fitting
package, so adding it does not invalidate existing preparation code signatures.
Its own source and launcher are hashed in its independent manifest.
