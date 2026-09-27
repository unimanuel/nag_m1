# Fixed-geometry pole sampling diagnostic

Run from the activated NAG environment, after completing all four repeat batches:

```bash
DRY_RUN=1 bash compare_ego_poles_sampling.sh
bash compare_ego_poles_sampling.sh
```

Default output: `D:\CGProjectData\waymo\diagnostics\poles_sampling_v1`.
An existing output directory is never overwritten. Use `--output PATH` for a new
measurement. `--sequences W254_86_137` limits the diagnostic to one interval.
`PYTHON_BIN` can select a different environment interpreter.

The script reads all 84 completed jobs from their saved manifests, verifies
identical geometry across mappings and identical saved mappings across repeats,
and measures each of the 28 unique mappings once. It reuses the existing
`parallax_v1` observation caches and the regular sphere references recorded in
`texture_sampling_v1/manifest.json`; it does not require pixel_error folders.
It does not train, refit, change PLY files, or modify existing results.

Outputs:

- `metrics.csv`: absolute sampling density and anisotropy distributions, coverage,
  seam/pole proximity, texture allocation, and three-repeat quality mean/SD.
- `upright_minus_legacy.csv`: paired diagnostic and quality differences.
- `sampling_by_region.csv`: minimum-density summaries by frame and image cell.
- Per-interval/mapping `samples.npz`: shared image coordinates, frame IDs,
  balancing weights, UVs and sampling measurements for subsequent analysis.
- `manifest.json`: source hashes, observation metadata, sphere references and
  measurement policy.

Every mapping uses all the same cached in-image background-mask-selected LiDAR
source pixels. These do not cover every image pixel and are not guaranteed
semantically static. Means equalize occupied 12x8 cells within each frame;
percentiles are unweighted. Shared-valid density means and invalid/coverage
fractions prevent interpreting excluded rays as better sampling.

The canonical sampling floor is evaluated only in regular mapping: minimum
texels/pixel below half the sphere reference, with a Huber penalty on the log
shortfall (delta 0.2). The reference uses the same texture dimensions and pole
policy, but its own physical sphere geometry. Absolute reference densities are
also reported. Visible-UV uses the actual saved crop and allocation; it does not
invent an upright sphere crop or claim to evaluate a Visible-UV fitting floor.

Near-seam (5 degrees) and near-pole (10 degrees) fractions are descriptive, not
quality thresholds. UV rectangle area is not physical surface area. This measures
initial cameras and texture mapping, not learned flow. Association with render
quality is exploratory and cannot establish that geometry refinement will help.
