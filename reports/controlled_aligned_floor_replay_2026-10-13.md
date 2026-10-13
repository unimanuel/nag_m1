# Full W191 0-40 aligned refinement replay — 2026-10-13

Nine CPU-only refinements completed: three fresh-process repeats each of
historical v1, historical v2 and generalized v2 with all cached observation
arrays aligned to 64 bytes. Fit time totaled 86.6 seconds; first timing,
resumption, subprocess setup and completion took approximately two minutes.
No training ran, and no production fitting setting, tolerance or historical
artifact was changed.

## Controls

- Each worker loads all implementations before observations, then copies every
  observation array to 64-byte-aligned C-contiguous storage. Values, dtype,
  shape, strides and ordering remain identical before and after alignment.
- The input fingerprint matches the preceding native replay and derivative
  experiment exactly. Workers check that fitting does not mutate those inputs.
- Python, NumPy, SciPy, CPU affinity, library builds and actual single-thread
  thread pools match the frozen native replay. All nine recorded runtimes match.
- Historical numerical source revisions and shared helpers are unchanged.
  Historical v2 and the generalized adapter use byte-identical kernel source.
- The real optimizer runs the original 5/10/20 continuation and default SLSQP
  finite differences, followed by safety repair and full-observation candidate
  selection. This is not the preceding closure-interception experiment.
- Geometry comparisons retain `rtol=1e-6`, `atol=1e-4 m` for center/radii and
  `atol=1e-6` for rotations. No release approval is issued by this diagnostic.

## Results versus the stored historical geometry

Every row applies to all three repeats. Errors are maximum absolute coordinate
or radius-component deviations, not vector lengths.

| Version | Center difference (mm) | Radius difference (mm) | Combined objective difference | Existing tolerance |
| --- | ---: | ---: | ---: | --- |
| Historical v1, aligned | 0.143964 | 0.103480 | -2.41357e-9 | FAIL |
| Historical v2, aligned | 0.467235 | 0.500668 | +1.32700e-9 | FAIL |
| Generalized v2, aligned | 0.104903 | 0.025824 | -7.74397e-10 | FAIL |

All three rotations exactly match stored geometry. Each version's three
center/radii/rotation arrays and final combined objectives are exactly identical.
Generalized v2's radius comparison passes, but a center component remains just
outside the existing mixed relative/absolute tolerance.

| Between-version comparison | Center difference (mm) | Radius difference (mm) | Absolute objective difference |
| --- | ---: | ---: | ---: |
| Historical v1 / historical v2 | 0.608779 | 0.604148 | 3.74057e-9 |
| Historical v1 / generalized v2 | 0.131444 | 0.116165 | 1.63917e-9 |
| Historical v2 / generalized v2 | 0.477335 | 0.487983 | 2.10139e-9 |

All between-version comparisons fail the unchanged geometric tolerance, although
their final objectives are effectively equal. Each of the three continuation
stages reports successful SLSQP termination in all nine jobs. Final supported
pair/depth validity is 100%; maximum camera normalized radius is 0.65 and
conservative clearance is at least 6 m.

## Comparison with the preceding native replay

| Version | Native center error vs stored (mm) | Aligned center error (mm) | Native radius error vs stored (mm) | Aligned radius error (mm) |
| --- | ---: | ---: | ---: | ---: |
| Historical v1 | 0.137329 | 0.143964 | 0.127346 | 0.103480 |
| Historical v2 | 0.050848 | 0.467235 | 0.051375 | 0.500668 |
| Generalized v2 | 0.133672 | 0.104903 | 0.129591 | 0.025824 |

Alignment does not reliably improve agreement: generalized radii are closer,
while historical v2 changes from passing its stored-reference comparison in
the native harness to failing it in this aligned harness. This is an archived
comparison between two separately instrumented harnesses, not a simultaneous
native/aligned fit ablation under an otherwise identical import/allocation
history. It cannot attribute every difference specifically to alignment.

## What this establishes

**Input alignment alone is not a sufficient compatibility fix.** Matching
fixed-parameter evaluations in the preceding diagnostic did not guarantee
matching evaluations throughout optimization, or matching final selected
geometries. Intermediate allocation, evaluation order/cache history and
finite-difference sensitivity remain uncontrolled.

The first measured data loss and sampling-floor loss are exactly equal in all
three aligned arms, giving the same initial weight-20 score
`32.004450885844655`. The kernel's reported original score, evaluated again when
returning the fit, is `32.004450885844705` for historical v2, about `4.97e-14`
higher. Other arms report the first value. This flags machine-level sensitivity
later in the execution history; it does not isolate whether encoding or loss
evaluation caused it. Final continuation iteration counts are 23/12/19 for v1,
23/12/19 for v2 and 23/12/18 for generalized v2.

The final objective differences remain around a few billionths, despite stable
submillimeter coordinate differences. There is still no evidence here of
changed observations, selection, loss weights, safety policy or mathematical
recipe. Exact reproduction of the original stored solution is not demonstrated
even by its historical source.

## Next step and boundaries

Do not install input alignment as a proven production fix or widen tolerances
automatically. The full preparation gate remains blocked. A further targeted
diagnostic would compare the first divergent SLSQP objective/constraint calls
and parameter encodings, rather than testing more arbitrary alignments. It
would need to acknowledge that tracing itself changes allocation context.

If numerical evaluation is subsequently stabilized, repeat the original
seven-interval compatibility gate. Accepting a numerical-equivalence tolerance
instead of recovering the existing strict geometry comparison would be a
separate explicit user decision, not an outcome silently applied here.

## Artifacts, rerun and validation

```text
D:\CGProjectData\waymo\diagnostics\autowork_aligned_floor_replay_v1\
  manifest.json
  summary.json
  autowork_W191_0_40_aligned64_VERSION_repeatNN\attempt_001\result.json
```

The default launcher reuses completed fits. For an independent fresh batch:

```bash
conda activate nag1
bash run_controlled_aligned_floor_replay.sh \
  --output-root /mnt/d/CGProjectData/waymo/diagnostics/autowork_aligned_floor_replay_v1_batch2
```

Four new unit tests and fourteen preceding diagnostic regression tests pass.
Bash syntax, whitespace checks and read-only frozen-data preflight pass. A
one-fit limit/resume was exercised on real data without changing the experiment
signature; the resumed batch produced all nine records and its summary. No
generated diagnostic artifacts are committed. See [the launcher guide](../docs/controlled_aligned_floor_replay.md).
