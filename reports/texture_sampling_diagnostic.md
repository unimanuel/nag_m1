# Texture sampling and correspondence diagnostic

Run from the NAG repository in the `nag1` environment:

```bash
DRY_RUN=1 bash diagnose_ego_texture_sampling.sh
bash diagnose_ego_texture_sampling.sh
```

This evaluates all seven W191/W254 ego intervals using completed sphere, AR3,
raw-v4, Moderate-Y, original reprojection, and available scale-limited runs in
both regular and Visible-UV modes. The latter is available for three intervals.
Only benchmark runs are used; seam-ablation runs are excluded. Geometry and
texture layouts come from each run's saved initial diagnostics, including its
effective radii and Visible-UV crop/resolution. Camera poses are the common
original calibrated poses, not the learned final poses.

It requires the existing full-interval caches under
`D:\CGProjectData\waymo\diagnostics\parallax_v1`. These are checked for stale
inputs/code. No pixel-error folders, GPU, training, or geometry fitting are
needed. If validation requests cache preparation, run
`bash prepare_ego_parallax_diagnostics.sh` and then retry. Do not bypass cache
validation.

Results are written to
`D:\CGProjectData\waymo\diagnostics\texture_sampling_v1`:

- `metrics.csv`: quality, correspondence, texture sampling, coverage and valid
  sample counts for each run. Both ordinary and frame/image-cell balanced means.
  `common_*` means additionally restrict all geometries within a mapping to the
  same valid samples; common coverage fractions accompany these comparisons.
- `sampling_by_region.csv`: per-frame and 12-by-8 image-cell sampling summaries.
- `deltas_vs_sphere.csv`: differences relative to the same interval/mapping sphere.
- `exploratory_correlations.csv`: Spearman correlations of those differences
  with PSNR, SSIM and LPIPS differences, separately for each mapping.
- `manifest.json`: saved run inventory, observation provenance and measurement policy.
- Per-run `samples.npz` and `summary.json`: individual samples for further plots.

For source image coordinate p, the diagnostic measures the Jacobian of actual
texture texels with respect to image pixels using central differences at
p +/- 0.5 pixels. It reports the smaller and larger singular values (texels per
pixel), their product (texel area per image pixel), and their ratio (directional
imbalance). Large condition numbers can expose directional compression; small
values of both singular values expose uniform compression that condition number
alone misses. Regular texture seams are wrapped. Visible-UV tile crossings and
failed intersections are excluded from finite summaries and counted explicitly.
Fractions below 0.25, 0.5 and 1 texel/pixel are descriptive thresholds, not proven
quality boundaries. Larger values are not automatically better.

Cross-frame pixel, camera-angular, surface-angular and texture-texel correspondence
errors reuse the established evaluator on target-LiDAR-supported observations.
These measure geometric/texture disagreement before learned flow. Source sampling
uses background-mask-selected LiDAR pixels; this is not exhaustive image coverage
or guaranteed semantic static-point selection. Missing sky LiDAR and changing
tile coverage must be considered when interpreting results.

Balanced means give equal mass to each sampled frame and each occupied image
cell within that frame, before invalid samples are removed. Correlations are
exploratory, with dependent runs and no statistical significance claims. They
test whether these proposed terms distinguish harmful geometries before adding
them to the optimizer; no combined loss or automatic candidate choice is applied.

Optional overrides:

```bash
SEQUENCES="W191_0_40 W254_0_37 W254_139_195" bash diagnose_ego_texture_sampling.sh
OUTPUT_ROOT=/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_halfstep \
  bash diagnose_ego_texture_sampling.sh --difference-step 0.25
```

Use a separate output folder for subsets or finite-difference sensitivity runs.
`PYTHON_BIN`, `PREPARED_ROOT`, and `OUTPUT_ROOT` can override their defaults.
