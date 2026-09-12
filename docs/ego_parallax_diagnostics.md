# Existing-geometry parallax and spatial-error diagnostics

This workflow evaluates completed W191/W254 sphere, clipped centered AR3,
raw depth-proxy v4, Moderate-Y, and fixed-geometry texture-orientation runs.
It does not fit geometry or launch training. Existing `final/pixel_error`
folders are not required. Run it from WSL in the `nag1` conda environment.

## Run the two controlled intervals

```bash
cd /mnt/c/Coding/CGProject/nag1
conda activate nag1
export SEQUENCES="W254_37_81 W254_86_137"

DRY_RUN=1 bash prepare_ego_parallax_diagnostics.sh
bash prepare_ego_parallax_diagnostics.sh
bash evaluate_ego_parallax_diagnostics.sh
bash export_ego_spatial_error_comparison.sh
bash summarize_ego_parallax_diagnostics.sh
```

Open `D:\CGProjectData\waymo\diagnostics\parallax_v1\report.html`.
The full metrics, observation counts, signed error maps, and alignment
previews are linked from this report. A signed map is blue where the candidate
has lower RGB-MAE, white for equality, and red where it has higher error.
The default signed-MAE display cap is 0.10; numeric metrics are not clipped.

To process all seven intervals, reuse completed work:

```bash
unset SEQUENCES
bash prepare_ego_parallax_diagnostics.sh
bash evaluate_ego_parallax_diagnostics.sh
bash export_ego_spatial_error_comparison.sh
bash summarize_ego_parallax_diagnostics.sh
```

The initial inventory contains 59 saved runs: seven variants per W191 interval,
eight per W254 interval, and six additional seam-test runs. The historical set
has no regular-sphere baseline for W191. It is reported as missing; Sphere UV
is never substituted for a regular-sphere comparison. All available runs get
absolute spatial metrics, even where a matching sphere comparison is missing.

When a new benchmark is written beneath a later `runs/YYYY-MM-DD` directory,
include that directory without changing source code. For the missing W191
regular-sphere runs, first execute:

```bash
bash run_W191_regular_sphere_benchmarks.sh
```

Then include its dated output directory when rerunning the diagnostic stages:

```bash
export PARALLAX_EXTRA_RUN_ROOTS="2026-09-12"
bash prepare_ego_parallax_diagnostics.sh
bash evaluate_ego_parallax_diagnostics.sh
bash export_ego_spatial_error_comparison.sh
bash summarize_ego_parallax_diagnostics.sh
```

Replace `2026-09-12` with the actual date folder printed by the runner if the
runs are launched on another date. The inventory rejects duplicate matching
run names, preventing an old and new baseline from being silently mixed.

## Controls and resumability

All wrappers accept extra Python options; use `--help` for the complete list.
Common environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON_BIN` | `python3` | Interpreter from the active environment |
| `SEQUENCES` | all seven | Space-separated interval labels |
| `OUTPUT_ROOT` | `/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1` | Output and cache root |
| `PARALLAX_EXTRA_RUN_ROOTS` | empty | Space-separated additional `runs/` folders to inventory |
| `RAYS_PER_FRAME` | 256 | Spatially balanced source samples |
| `TARGETS_PER_FRAME` | 8 | Nearby/distant/turning target-frame candidates |
| `FRAME_LIMIT` | 0 (all) | Evenly spaced smoke-test frames only |
| `BOUNDARY_WIDTH` | 3 | Foreground dilation/erosion radius in rendered pixels |
| `DISPLAY_CAP` | 0.10 | Signed RGB-MAE saturation |
| `DRY_RUN` | 0 | Validate inventory and source file existence without writes |
| `FORCE` | 0 | Recompute the matching cache entry |

Use a separate output root for smoke tests. For example:

```bash
export OUTPUT_ROOT=/tmp/nag_parallax_smoke
export SEQUENCES="W254_37_81 W254_86_137"
export FRAME_LIMIT=3 RAYS_PER_FRAME=32 TARGETS_PER_FRAME=2
bash prepare_ego_parallax_diagnostics.sh
bash evaluate_ego_parallax_diagnostics.sh
bash export_ego_spatial_error_comparison.sh
bash summarize_ego_parallax_diagnostics.sh
unset OUTPUT_ROOT SEQUENCES FRAME_LIMIT RAYS_PER_FRAME TARGETS_PER_FRAME
```

Sample-limited outputs are explicitly labelled `smoke_subset`. Their saved
training metrics still describe the complete interval. Do not interpret a
smoke report as the full experiment.

Each stage writes to a content/settings-keyed directory. Completed results
are reused; incomplete entries are recomputed. JSON pointers are written only
after completion. File paths, sizes, nanosecond modification times, diagnostic
code hashes, settings, and saved run diagnostics establish provenance. File
contents restored with exactly identical size and mtime are not detectable by
the file-stat fingerprint. Code or source changes require preparation again,
and stale evaluation/spatial results are rejected when summarizing. `FORCE=1`
rewrites generated diagnostics only; it does not modify training results.

## Geometry and camera contract

Geometry comes from each run's `initial/background_diagnostics` record.
Ellipsoid effective radii already include margin, clipping, or overrides;
source JSON radii are not substituted. Historical sphere records contain
radius/center rounded to four decimal places. Their fixed texture rotation is
identity. Ellipsoid texture rotations, seam offsets, visible-tile bounds, and
texture resolutions come from the saved diagnostics.

All candidates use the same original calibrated NAG camera poses and
intrinsics. Learned camera changes are not applied, so these are diagnostics
of the fixed initial proxy and calibration, not measurements of final neural
flow. The inventory checks saved camera configurations against the dataset
and rejects unsupported camera normalization, distortion, or differing image
sizes. Intrinsics follow `PinholeCameraConfig.get_intrinsics`: both focal
entries scale with image width; the principal point scales in each dimension.

The loader reuses `nag-waymo-processing`'s per-frame coordinate alignment with
pose correction enabled. FRONT-projected LiDAR points are projected into the
undistorted training-image coordinates. Their coordinates are checked against
undistorted Waymo pixel labels; a P90 discrepancy above 15 rendered pixels
stops preparation and writes the alignment failure details. Small discrepancies
can remain from Waymo projection timing/calibration. The threshold is an
alignment check, not a claim of subpixel accuracy. Inspect the exported
alignment preview images as well as the numeric residuals.

Masks are unions of the saved NAG mask layers, sampled in the same coordinate
system used for training and resized with nearest-neighbor interpolation.
This assumes those masks correspond to the undistorted images; previews expose
misalignment for inspection. No semantic classifier or static-object tracker
is introduced. Unmasked moving objects can remain.

## Shared observation selection

All FRONT points, including masked points, contribute to the target sparse
depth buffer so foreground objects can occlude background. Keep the nearest
depth in each pixel, then query the nearest measured pixel within a 2-pixel
radius. The z-depth tolerance is `0.5 m + 0.03 * query_depth`:

- Within tolerance: supported.
- Point behind measured surface: occluded.
- Point in front of measured surface beyond tolerance: inconsistent.
- No nearby measurement: unknown.

Source rays require supported self-visibility, a valid image projection, and
absence from the foreground mask. Deterministic round-robin sampling covers
12x8 image cells. Targets include temporal neighbors, distant frames, and
heading changes; no roads or turn counts are specified. Target projections
must be inside the FRONT image, in front of its camera, and outside its masks.
Occluded/inconsistent pairs are rejected. Supported and unknown pairs are
cached separately. Primary statistics use supported pairs; the sensitivity
rows include both supported and unknown pairs. Pixel radius and depth
tolerances are command-line options, recorded in cache metadata.

Different depths across nearby pixels can occur at discontinuities. This is
conservative sparse evidence, not proof that the same surface was observed.
There is no dense hole filling or baseline multiplication. Per-baseline,
heading-bin and source-frame tables expose the sampling distribution.

## Measurements

For point P seen from cameras i,j, intersect both camera-to-P rays with the
same proxy to obtain Qi,Qj. Report:

- Mean of forward `|project_j(Qi)-project_j(P)|` and reverse pixel error.
- Mean forward/reverse camera-direction angular error, in degrees.
- Angular separation of `D^-1 R^T(Qi-c)` and `D^-1 R^T(Qj-c)`, normalized
  to unit vectors. This is a unit-sphere angle, not a physical geodesic.
- Shortest periodic-longitude UV distance, seam-crossing frequency, and
  effective texture-texel displacement. Visible-UV coordinates use their saved
  tile scale; out-of-tile pairs have undefined texel distance and a coverage
  count. Texel distance does not describe hash-grid resolution or learned flow.
- Relative/log ray-depth errors over unique sampled source observations,
  maximum camera normalized radius, and conservative clearance
  `min(radii)*(1-max_camera_normalized_radius)` in metres.

Nonpositive-depth/invalid proxy projections invalidate the pair across the
headline metrics and are counted. A proxy outside the target image but still
in front of the camera retains its pixel error. Finite-only percentiles are
accompanied by counts; an additional bounded pixel score caps every error at
the image diagonal and assigns that penalty to invalid pairs. This score is a
diagnostic, not a newly selected optimization objective.

The spatial exporter divides each image into disjoint foreground interiors,
background away from boundaries, and a band spanning both sides of mask
boundaries. It computes normalized RGB-MAE, RMSE, and PSNR directly from saved
8-bit renders and bilinearly resized ground truth. Sequence region metrics
are pixel weighted. PSNR for a perfect match is recorded as null plus
`perfect_match=true`, to keep JSON/CSV finite; empty regions are also marked
by their zero pixel count. Saved whole-image SSIM/LPIPS are included in the
report but are not recomputed as masked metrics.

## Reading the report

`comparison.csv` contains both visibility populations and geometry/spatial
statistics. `deltas_vs_matching_sphere.csv` compares within each sequence and
texture mode. `within_sequence_correlations.csv` reports descriptive Spearman
correlations between geometry metrics and background MAE, excluding seam
reruns and geometries with invalid projections. There are few independent
geometries: these coefficients are exploratory, not significance tests.

The plots show changes relative to the matching sphere; lower-left means the
candidate improves both the correspondence metric and background MAE. Missing
baselines and sequences whose stages are unfinished are explicit. A new
fitting objective should be chosen only after examining visibility sensitivity,
spatial differences, and the consistency of the relationship across intervals.

## Implementation verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_parallax_diagnostics.py -v
```

The invariant tests cover analytic intersections, oriented ellipsoids, zero
parallax for points on the surface and pure rotation, nonzero parallax under
translation with wrong depth, projection scaling, sparse visibility, UV
wrapping/tile coverage, disjoint regions, provenance invalidation and run-name
carriage returns. NumPy, SciPy, Pillow, PyYAML, OpenCV and Matplotlib are used;
the diagnostic entry point does not import the NAG training stack or require
CUDA.
