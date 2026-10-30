# Dense final-checkpoint background-flow maps

This is inference-only postprocessing of a trusted local final checkpoint.
It does not construct a trainer, change learned weights or geometry, overwrite
the source run, or require `pixel_error` or LiDAR observations. Do not run the
GPU extraction alongside a controlled training batch.

The first case is W191 128–198, legacy-pole regular ellipsoid, seed 42 repeat 1
from the September pole-repeatability benchmark. No best-repeat selection is
performed. Run from the repository with the existing CUDA-capable interpreter:

```bash
bash export_W191_128_198_dense_flow.sh
```

The launcher resumes only with identical inputs, chunk size and source-code
hashes. Its default extraction budget is 1,800 seconds; it stops between frames
and exits with code 2 if exhausted. Completed frame files are published
atomically and retained. Repeat the same command to continue. CPU reporting
runs after extraction and is additional to that budget. Source inputs are
checked before and after processing. New source-run or geometry files are
never written.

Default output:

```text
D:\CGProjectData\waymo\diagnostics\autowork_W191_128_198_dense_flow_2026-10-30\
  index.html                 # overview, links and embedded MP4
  background_flow.mp4        # reconstruction / magnitude / MAE / gradient
  magnitude/                 # full-resolution per-frame magnitude heatmaps
  gradient/                  # full-resolution per-frame gradient heatmaps
  panels/                    # four-panel per-frame images
  frames/                    # signed pixel/native-UV fields, metrics and masks
  temporal_average/          # mean maps and valid-background frame counts
  temporal_average.npz       # numerical means and per-measure support counts
  by_frame.csv               # every frame, measure and region
  summary.csv / summary.json # equal-frame aggregate summaries
  verification/              # saved / rerendered / amplified differences
  verification.csv / .json  # three-frame composite/background parity gate
  manifest.json              # config/checkpoint hashes and input provenance
  progress.json / result.json
```

The generic entrypoint `reports/dense_background_flow.py` accepts `--run`, `--output`, `--chunk-pixels`,
`--max-seconds` and `--resume`. Output must use a separate `autowork_` directory.
This first exporter supports curved backgrounds; it does not claim a planar
checkpoint adapter. The renderer itself already provides per-object planar
and curved flow through its modality API.

## Definitions and checks

The actual background modality exposes its backward texture-lookup offset.
It is applied to the actual curved surface, projected through the learned
camera, and converted to pixels at the saved-image resolution. This is not
frame-to-frame optical flow or a physical object-motion estimate.

The scalar magnitude is `sqrt(fx**2 + fy**2)`. The temporal map averages those
magnitudes on valid background support, not the norm of the mean vector.
Opposite directions therefore do not cancel. Image coordinates stay fixed:
these maps do not track the same 3D point through a moving camera sequence.

Native UV magnitudes retain the network's raw coordinates. Effective UV
magnitudes instead wrap longitude and clamp latitude as the renderer does.
Neither should be compared directly with another geometry's raw UV units as
if it were an image-pixel distance. Visible-UV densification is handled by the
runtime query; this exporter does not add fixed-initial-texture texel metrics.

The image-pixel gradient is the Frobenius norm of the central-difference
Jacobian of the two-component displacement, in pixels/pixel. It excludes the
outer image border and stencils with any invalid hit/projection. Zero flow
and invalid flow are distinguished. Native quantities and pixel magnitudes
are extracted once per pixel rather than using five GPU stencil queries.

The background exterior excludes a three-pixel union-mask boundary band.
Foreground interiors, boundaries and all valid rays are separate CSV controls;
these masks are not semantic proof of static background. Background-only MAE
and full-composite MAE are both retained on normalized RGB [0,1].

Before extraction, three evenly spaced frames must reproduce both the saved
composite and the callback-equivalent saved background layer. The existing
gate permits mean channel difference <=0.5 byte and P99 <=2 bytes. Failures
stop extraction without widening tolerances. All source frames are measured
only after this gate passes; it is a representative loading check, not a
claim that all 71 frames were separately rerendered for parity.

Heatmaps share a scale across all frames of this run. Display ceilings are
95th percentiles of deterministic subsamples of valid background values; no
numeric metric is capped. Grey means excluded, not zero. The MP4 uses OpenCV's
MPEG-4 encoder; use a local video player if the browser cannot play that codec.
Across-run comparisons would need new common scales and matched support.

Summary statistics first measure each frame, then give contributing frames
equal weight. The reported median/P90 summaries average frame medians/P90s;
they are not pooled percentiles. Within-frame Spearman associations use every
fourth row and column, are descriptive, and do not establish causality or
independent-pixel significance. Less flow is not automatically better quality.

Tests (no training):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests" \
  /home/mang/anaconda3/envs/nag1/bin/python -m unittest \
  test_dense_background_flow test_learned_flow_diagnostics
```

## First measured case, 30 October 2026

The pilot completed all 71 frames at 960 x 640 (43,622,400 pixel queries).
Loading, three-frame render parity and extraction took 447.4 seconds;
the complete export took 624.3 seconds. Each frame's numerical arrays, scalar
heatmaps and four-panel image are retained. The 10-fps MP4 has 71 frames at
1920 x 1376; beginning, middle and end frames decoded successfully.
The complete output occupies approximately 2.00 GiB on D:.

On valid background exterior support, equal-frame averages were:

| Quantity | Mean of frame means | Mean of frame medians | Mean of frame P90s |
| --- | ---: | ---: | ---: |
| Pixel-flow magnitude (pixels) | 5.33124 | 3.75553 | 11.27260 |
| Pixel-flow gradient (pixels/pixel) | 0.84643 | 0.74635 | 1.45128 |
| Native UV magnitude | 0.00171329 | 0.00137795 | 0.00337176 |
| Composite RGB MAE [0,1] | 0.0203541 | 0.0102734 | 0.0457700 |

The mean within-frame Spearman association with composite MAE was -0.00963
for pixel magnitude and +0.00909 for pixel gradient. These weak descriptive
associations do not show that flow causes errors or that this geometry is
better than a sphere. This pilot analyzes one ellipsoid repeat only.

All six parity comparisons passed: maximum mean difference was 0.000591 byte
and every P99 was zero. Support counts in the temporal magnitude map range
from 52 to 71 frames; this explains why its unweighted image average is not
identical to the equal-frame summary above.

The 22 dense/existing-flow unit tests pass, including agreement with the old
five-point diagnostic and non-cancellation of opposite vectors. The exporter
lives under `reports/`, outside frozen training/preparation code discovery.
The existing seven-interval release gate still passes and the historical
final-baseline training code snapshot is unchanged. The original pilot folder
is retained with an `_pilot` suffix and an exact executed-source archive; the
canonical folder is regenerated by the reporting-only entrypoint.
