# Side-by-side presentation videos

This CPU-only tool composes two completed runs' `final/complete_scene.mp4`
videos into one labeled H.264 MP4. It never loads a checkpoint, imports a NAG
model, starts training, uses CUDA, or edits source runs. Pillow and PyYAML are
required; FFmpeg/FFprobe with the `libx264` encoder are also required. They are
already available in the local `nag1` environment.

## First sequence: W254 86–137, regular Plane versus Ellipsoid

From the repository root in WSL:

```bash
conda activate nag1
DRY_RUN=1 bash make_W254_86_137_plane_ellipsoid_video.sh
bash make_W254_86_137_plane_ellipsoid_video.sh
```

The launcher selects repeat **01** of the matched paper Plane baseline and
legacy regular-mapping full-recipe Ellipsoid, not a best-of-three result. It
writes to:

```text
D:\CGProjectData\waymo\diagnostics\autowork_presentation_comparisons\
  W254_86_137_plane_vs_ellipsoid_repeat01.mp4
  W254_86_137_plane_vs_ellipsoid_repeat01.preview.png
  W254_86_137_plane_vs_ellipsoid_repeat01.json
```

The header displays each **individual run's final mean PSNR**, read from the
epoch matching `max_epochs` in `tracker/train_epoch_MeanMPSNR.csv`. It does not
show the paper's three-repeat average or a per-frame changing score. The JSON
records exact scores, segment/frame identity, run paths, seed, background class,
UV mode, configuration/tracker/video hashes, playback rate and output details.

The default preserves the source videos' playback rate (5 FPS for this pair).
The 52 selected frames therefore play for 10.4 seconds. To play those same
frames at 10 FPS instead (5.2 seconds), choose a fresh output name:

```bash
OUTPUT=/mnt/d/CGProjectData/waymo/diagnostics/autowork_presentation_comparisons/W254_86_137_10fps.mp4 \
  bash make_W254_86_137_plane_ellipsoid_video.sh --fps 10
```

`--fps` changes presentation speed; it does not interpolate new scene frames.
The tool keeps aspect ratio, adds a black header above rather than over the
image, and defaults to native panel resolution. Encoding is lossy presentation
output, not a new metric evaluation. The scores refer to saved run metrics.

## Other sequences / treatments

Pass the exact completed attempt folders (containing `training_config.yaml`):

```bash
python nag/scripts/make_result_comparison_video.py \
  --left-run '/path/to/plane/attempt_001' \
  --right-run '/path/to/ellipsoid/attempt_001' \
  --left-title 'PLANE' --right-title 'ELLIPSOID' \
  --sequence-label 'W191 128–198' \
  --output '/mnt/d/CGProjectData/waymo/diagnostics/autowork_presentation_comparisons/W191_128_198.mp4'
```

Use `--right-title 'ELLIPSOID UV'` when comparing a Visible-UV run. Labels are
literal Pillow text, not shell expressions or FFmpeg filter fragments. Default
labels abbreviate the segment ID for display only; validation uses its full ID.

Inputs must have matching segment IDs, explicit selected frame indices, source
image paths, rendering-resolution settings, epochs and video dimensions. Both
videos must contain exactly one frame per selected index. Different source FPS
requires an explicit `--fps` choice. Mismatches fail instead of silently using
the shortest video. This validates saved provenance/counts, not the semantic
content of each decoded frame. Source video quality cannot be recovered by
encoding it again.

Output files are never overwritten. Choose another `--output` name to rerun.
`--dry-run` probes the videos and reads metadata but writes no files or folders.
The launcher also accepts `LEFT_RUN`, `RIGHT_RUN`, `OUTPUT`, and `PYTHON_BIN`.
Binary selection can be overridden with `--ffmpeg`, `--ffprobe`, or environment
variables `FFMPEG_BIN` and `FFPROBE_BIN`. Other options include `--panel-width`
(even, ≥160), `--header-height` (even, ≥24), `--font`, `--threads` (default 2),
and `--crf` (default 18). More CPU workers can affect other concurrent jobs;
the default is deliberately small.

## Tests

```bash
python -m unittest discover -s tests -p test_result_comparison_video.py -v
bash -n make_W254_86_137_plane_ellipsoid_video.sh
```
