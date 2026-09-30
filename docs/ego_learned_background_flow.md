# Final-checkpoint learned-background-flow investigation

This is the next diagnostic after `diagnose_ego_texture_flow.sh`. It reads
existing final checkpoints and saved images; it does **not** train, fit geometry,
change mappings, regenerate observation caches, or modify existing run folders.
No `pixel_error` folder or LiDAR observation cache is required. Activate the
usual CUDA-capable `nag1` environment.

## Run

```bash
conda activate nag1
DRY_RUN=1 bash diagnose_ego_learned_background_flow.sh
bash diagnose_ego_learned_background_flow.sh
```

The default is **24 checkpoint diagnostics**: W191 0–40, W254 0–37,
W254 139–195 (the three problematic intervals), and W254 86–137 (successful
control); sphere, original-reprojection v1, and legacy sampling-floor v2;
regular and Visible-UV for each. It uses repeat 01 of the floor benchmark,
chosen in advance rather than the best repeat or a three-repeat mean.
Each checkpoint uses every frame, a 16-pixel-spaced grid and one-output-pixel
central differences. A new process handles each checkpoint to release GPU
memory between jobs. Jobs run sequentially. Logs are in each job's `console.log`.
The successful control may retain identical original/floor geometry; that is
a useful control for stochastic learned-flow differences, not another width change.

Output: `D:\CGProjectData\waymo\diagnostics\learned_background_flow_v1`.
Open `index.html` there. Send back that folder when finished.

An interrupted batch can resume with identical inputs and settings:

```bash
bash diagnose_ego_learned_background_flow.sh --resume
```

Optional smaller first check (separate output so it cannot be confused with the
full experiment):

```bash
bash diagnose_ego_learned_background_flow.sh \
  --sequences W254_86_137 --families reprojection_v1 --modes regular \
  --max-frames 3 --verify-frames 1 \
  --output /mnt/d/CGProjectData/waymo/diagnostics/learned_background_flow_smoke
```

To include all seven ego intervals, supply all seven labels with `--sequences`.
`--floor-repeats 1 2 3` adds the other two floor repeats. Both are expansions,
not necessary for the initial wide-versus-narrow investigation. `--families`
also supports AR3, raw v4, Moderate-Y and radius-scale v1 if present in the
completed prior diagnostic's inventory.

For the existing upright-pole experiment, use the separate source catalog and
an `autowork_` output folder. This compares repeat 01 of the upright mapping
on the fixed legacy-floor geometry with the completed upright-aware floor fit;
it does not create new geometry or train:

```bash
python nag/scripts/ego_learned_background_flow.py \
  --upright-sources reports/2026-09-28_upright_comparison.sources.json \
  --sequences W191_0_40 W254_0_37 W254_139_195 \
  --output /mnt/d/CGProjectData/waymo/diagnostics/autowork_upright_learned_flow_v1
```

The upright comparison uses the same grid, saved-image parity and learned-flow
definitions as the original diagnostic. It compares completed checkpoints, not
the untrained `autowork_` optimizer candidates; their actual learned flow can
only be measured after training.

## Reproduction gate

Before reading flow, the loader renders three evenly spaced frames at the
**original saved resolution**, with the original checkpoint weights and mapping.
It compares both the full composite and the separately saved background layer.
The background is re-rendered through the callback's joint per-object
composition, not as a standalone background-only query: those paths can differ
even with the same weights. The saved layer must match its actual export path.
The quantization matches the final-render callback: multiply by 255, clamp,
truncate to uint8. Defaults require mean absolute channel difference at most
0.5 byte and 99th-percentile difference at most 2 bytes. These are tolerances
for numerical/PNG quantization effects, **not** reconstruction-quality targets.
Maximum discrepancy and fraction differing by more than one byte are also saved.

`verification/` contains saved render, checkpoint re-render, and 16× amplified
absolute difference side by side. `verification.csv` and `verification.json`
record the values. A failure stops that job before flow analysis; the batch
continues other jobs, excludes failed jobs from comparisons, and exits nonzero.
Do not loosen tolerances to make a loading problem disappear. Inspect the
failure before interpreting the flow.

Initialization diagnostics during checkpoint loading are redirected to the
new job's `runtime/` directory. No trainer, optimizer, callbacks, or source-run
output logger is created. The script only accepts the one `final-80.ckpt` for
each run and records config hashes, checkpoint metadata and image/mask metadata.
It loads trusted local project checkpoints with the project's existing loader.
The loader's empty proxy masks cannot support the visible-foreground sphere
radius heuristic. For proxy construction only, this diagnostic supplies the
already recorded sphere radius; checkpoint loading then restores the exact
registered radius buffer. It does not refit a sphere or change the saved config.

## What is measured

The background modality call exposes its actual native learned offset
`f(p,t) = (delta_u, delta_v)` before converting it into image-space motion.
The same call uses the checkpoint's learned camera transforms, actual surface
intersections, texture rotation, flow weights and time interpolation. Required
flow-reference timestamps are included. These are **backward texture-lookup
offsets**, not tracked object motion, ground-truth optical flow, or the sparse
geometric correction field from the previous diagnostic.

- Raw native/global-UV flow magnitude, image-pixel-space magnitude, and effective
  UV flow after the renderer's longitude wrap and latitude clamp.
- Central-difference spatial gradients in UV/output-pixel, fixed-texture
  texels/output-pixel and output-pixel/output-pixel, plus native UV curvature.
- Local transport `J_after @ inverse(J_before)`: texture stretch singular values,
  condition, determinant and departure from identity. Ill-conditioned source
  charts, invalid hits and flowed points behind the camera are not zeros: they
  are excluded. Longitude differences are locally periodic.
- Fixed initial-RGB texel offsets/gradients using the checkpoint's actual tile
  bounds, texture resolution and `align_corners`. Visible-UV densification affects
  **this lookup only**, not the global-UV input to the flow or learned RGB network.
  Texel comparisons require both endpoints and the stencil to remain in a
  continuous supported tile branch.
- Saved full-composite normalized RGB MAE and MSE; saved background-only RGB MAE.
  RGB values are divided by 255. There is no 0.10 numerical cap on these metrics.
  Background-only RGB error is meaningful mainly outside foreground masks.

The sampled output-pixel grid is converted using the renderer's actual
`linspace(0, camera_size, output_size)` convention, rather than assuming a
different pixel-center calibration. Pixel-flow units are converted back to the
saved-image resolution. All geometries use the same output grid and frame scope.

`fields.npz` keeps signed native UV and image-pixel flow, global UV coordinates,
all scalar fields, region masks, validity flags and frame/grid indices. Regions
are foreground-mask interior, a three-pixel boundary band, background exterior,
and all pixels. These masks **do not identify static background semantically**.
Flow is measured even behind foreground, but interpretations should focus on
background interiors and use boundary/foreground as separate controls.

## Comparisons and interpretation

Each job writes `metrics.csv`, `by_frame.csv`, `fields.npz`, verification outputs
and three field figures. The root writes combined `metrics.csv`,
`comparisons.csv`, an inventory `manifest.json`, and `index.html`.
`comparison_images/` contains paired reference/candidate panels with shared
color scales and signed difference maps for RGB MAE and learned pixel-flow
gradient. Blue means candidate minus reference is negative; this is an advantage
for reconstruction error, **not automatically for flow gradient**. White means
zero difference, with no dead-zone threshold; missing support is grey. Color
limits use a common 95th percentile only for display, not a numerical metric cap.
Comparisons use exactly matched frames/pixels and the intersection of finite
support for each quantity. Aggregates assign equal mass per contributing frame.
`n` and `frames` report support. The script pairs floor against original
reprojection and sphere; original reprojection is also paired against sphere.
Differences are **candidate minus reference**. Negative image-error differences
favor the candidate, but smaller flow alone is not a quality improvement.
PSNR/SSIM/LPIPS are actual final metrics of the selected checkpoint's own run.

Within-frame Spearman associations relate spatial flow variation to local image
MAE, and changes in that variation to changes in MAE on common supports. These
are descriptive associations: pixels are correlated, and changing geometry
also changes sampling, network training and compositing. They do not prove that
flow caused a reconstruction error, and do not measure network approximation
capacity or optimization difficulty directly. Field figures use per-run color
scales, explicitly labelled; compare numerical CSVs for cross-run differences.

The decision this enables: do the narrower, better-rendering geometries need
larger offsets but less localized flow distortion at error-heavy background
regions, or do the wide shapes simply provide too little useful initial texture?
It is diagnostic evidence before designing another optimizer loss, not a new
geometry-selection rule or a reason to force flow magnitude to zero.
