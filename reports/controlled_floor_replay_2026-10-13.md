# Controlled W191 0-40 refinement replay — 2026-10-13

Nine CPU-only refinements completed locally: three fresh-process repeats each
of historical v1, historical v2 and the generalized v2 adapter. Refinement time
totaled about 91 seconds, with subprocess setup bringing the batch to roughly
two minutes. No training, GPU model initialization, checkpoint saving, historical
file modification or release-gate change was performed.

## Controls

- All jobs used identical frozen observation arrays, ordering, strides/dtypes,
  saved reprojection initializer, sphere reference, mapping and optimizer settings.
  Their input fingerprints are identical.
- NumPy, SciPy, Python, numerical-library build/thread-pool information, CPU
  affinity and recorded runtime environment match across all nine jobs.
- Each worker used one CPU core, one numerical thread and `PYTHONHASHSEED=0`.
  Worker order rotated between repeats; every job ran in a fresh process.
- All workers loaded every implementation before loading observations.
- Historical source commits are `73e3874` (v1) and `07cbd74` (v2). Shared
  numerical helper files match both revisions exactly. Generalized v2's pinned
  kernel matches historical v2's source byte-for-byte.
- The existing geometric comparison remains `rtol=1e-6`, `atol=1e-4 m` for
  centers/radii and `atol=1e-6` for rotations. No threshold was widened.

## Results versus the stored historical optimum

Each row applies to all three repeats of that version.

| Version | Maximum center difference (mm) | Maximum radius difference (mm) | Combined-objective difference | Existing tolerance |
| --- | ---: | ---: | ---: | --- |
| Historical v1 | 0.137329 | 0.127346 | -2.42378e-9 | FAIL |
| Historical v2 | 0.050848 | 0.051375 | -1.55126e-10 | PASS |
| Generalized v2 | 0.133672 | 0.129591 | -1.94177e-10 | FAIL |

Every rotation matches exactly. Within each version, **all three geometries
and combined objectives are exactly identical**, not merely within tolerance.
Thus this particular controlled setup does not show random run-to-run variation.

| Version pair | Maximum center difference (mm) | Maximum radius difference (mm) | Absolute combined-objective difference |
| --- | ---: | ---: | ---: |
| Historical v1 / historical v2 | 0.122890 | 0.129152 | 2.26866e-9 |
| Historical v1 / generalized v2 | 0.146991 | 0.141160 | 2.22961e-9 |
| Historical v2 / generalized v2 | 0.184520 | 0.180967 | 3.90514e-11 |

These between-version differences fail the existing strict geometric comparison,
despite virtually identical objectives. Historical v2 happening to pass the
stored-reference comparison does not release generalized preparation or training.

## Interpretation and limits

The exact historical v1 source does not recover the stored optimum under this
controlled present-day execution setup. Therefore a mismatch to that stored
optimum is not, by itself, proof of a generalized-fitter mathematical change.
Nor does this test prove recreation of the complete historical execution context.

Historical v2 and generalized v2 have identical kernel source and shared
mathematics, yet produce stable tiny offsets here. Their initial combined
objectives are `32.00445088584462` and `32.004450885844655`, respectively—a
difference of about `3.55e-14`. This is already a machine-precision discrepancy
before iterative refinement, despite identical numerical input fingerprints.

The recorded source-point array alignment was 16 bytes modulo 64 in the
historical arms and 48 bytes modulo 64 in the generalized arm. Other allocation
details and intermediate-array alignment are not controlled. This is a useful
lead: memory layout, vectorized floating-point evaluation and finite-difference
gradient sensitivity could explain the offsets. The observed correlation does
**not** establish that alignment is the cause. There is no demonstrated input,
sampling, loss-weight or precision mismatch.

The generalized offset in this isolated harness is about 0.13 mm, whereas the
earlier full preparation replay gave about 0.53 mm. The different execution
contexts produce different tiny solutions near the same objective. Calling this
random sampling or treating the original geometry as an exactly replayable
unique minimum would overstate the evidence.

The next narrow diagnostic would hold allocation/alignment more tightly or
compare objective evaluations and finite-difference gradients at the same
parameters across implementations. Any geometry-tolerance revision remains an
explicit user decision, not an automatic result of this report.

## Files and independent repetition

Machine-readable evidence and per-job logs are in:

```text
D:\CGProjectData\waymo\diagnostics\autowork_controlled_floor_replay_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_ENGINE_repeatNN\attempt_001\result.json
```

An invocation using the same root reuses completed jobs. For a genuinely new
batch, use:

```bash
conda activate nag1
bash run_controlled_ellipsoid_replay.sh \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_controlled_floor_replay_v1_batch2
```

Eight new diagnostic unit tests pass; Bash syntax and read-only preflight pass.
The script is outside the NAG fitting package and does not invalidate its
existing code signatures. See [the replay guide](../docs/controlled_floor_replay.md).
