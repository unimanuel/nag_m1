#!/usr/bin/env python3
"""Export interpretable per-pixel reconstruction-error maps for a NAG run.

The diagnostic uses mean absolute RGB error at every pixel.  It writes an
exact 16-bit grayscale representation, fixed-scale Turbo heatmaps, heatmap
overlays, 2x2 inspection panels, frame statistics, and a contact sheet of the
worst frames.  A single robust color scale is shared by every frame so colors
remain comparable over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TURBO_COEFFICIENTS = np.asarray(
    [
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ground-truth-dir", required=True, type=Path)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-step", default=1, type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: RUN_DIR/final/pixel_error",
    )
    parser.add_argument(
        "--display-percentile",
        default=99.0,
        type=float,
        help="Sequence-wide MAE percentile mapped to the hottest color (default: 99).",
    )
    parser.add_argument(
        "--display-cap",
        type=float,
        help="Optional fixed normalized MAE cap in [0,1], overriding --display-percentile.",
    )
    parser.add_argument("--overlay-opacity", default=0.8, type=float)
    parser.add_argument("--worst-frame-count", default=12, type=int)
    return parser.parse_args()


def render_paths(complete_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^(?P<index>\d+)_t\.png$")
    result: list[tuple[int, Path]] = []
    for path in complete_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            result.append((int(match.group("index")), path))
    result.sort(key=lambda item: item[0])
    if not result:
        raise FileNotFoundError(f"No NAG complete renders found in {complete_dir}")
    return result


def ground_truth_path(directory: Path, frame_index: int) -> Path:
    candidates: list[Path] = []
    for stem in (f"{frame_index:03d}", str(frame_index)):
        candidates.extend(path for path in directory.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected one ground-truth image for frame {frame_index} in {directory}; found {unique}"
        )
    return unique[0]


def rgb_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def pixel_mae(render: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(render.astype(np.float32) - target.astype(np.float32)), axis=-1) / 255.0


def turbo(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 0.0, 1.0)
    powers = np.stack([values ** exponent for exponent in range(6)], axis=-1)
    rgb = powers @ TURBO_COEFFICIENTS.T
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def label_image(image: Image.Image, label: str, bar_height: int = 28) -> Image.Image:
    labelled = Image.new("RGB", (image.width, image.height + bar_height), "black")
    labelled.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(labelled)
    draw.text((8, 7), label, fill="white")
    return labelled


def inspection_panel(
    target: np.ndarray,
    render: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    frame_index: int,
    mean_mae: float,
    p95_mae: float,
    display_cap: float,
) -> Image.Image:
    cells = [
        label_image(Image.fromarray(target), f"Ground truth - frame {frame_index}"),
        label_image(Image.fromarray(render), "NAG reconstruction"),
        label_image(Image.fromarray(heatmap), f"Pixel MAE heatmap - fixed scale 0 to {display_cap:.3f}"),
        label_image(Image.fromarray(overlay), f"Error overlay - mean {mean_mae:.4f}, p95 {p95_mae:.4f}"),
    ]
    panel = Image.new("RGB", (2 * cells[0].width, 2 * cells[0].height), "black")
    for index, cell in enumerate(cells):
        panel.paste(cell, ((index % 2) * cell.width, (index // 2) * cell.height))
    return panel


def percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    target = histogram.sum() * percentile / 100.0
    bin_index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    return max((bin_index + 1) / len(histogram), 1.0 / len(histogram))


def contact_sheet(paths: Iterable[Path], labels: Iterable[str], output_path: Path) -> None:
    entries = list(zip(paths, labels))
    if not entries:
        return
    columns = 3
    thumb_width = 480
    thumbs: list[Image.Image] = []
    for path, label in entries:
        with Image.open(path) as source:
            image = source.convert("RGB")
            thumb_height = round(image.height * thumb_width / image.width)
            image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            thumbs.append(label_image(image, label, bar_height=30))
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumbs[0].height), "black")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb.height))
    sheet.save(output_path, quality=94)


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive.")
    if not 0.0 < args.display_percentile <= 100.0:
        raise ValueError("--display-percentile must be in (0,100].")
    if args.display_cap is not None and not 0.0 < args.display_cap <= 1.0:
        raise ValueError("--display-cap must be in (0,1].")
    if not 0.0 <= args.overlay_opacity <= 1.0:
        raise ValueError("--overlay-opacity must be in [0,1].")

    run_dir = args.run_dir.resolve()
    complete_dir = run_dir / "final" / "complete"
    ground_truth_dir = args.ground_truth_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "final" / "pixel_error")).resolve()
    if not complete_dir.is_dir():
        raise FileNotFoundError(f"Missing complete-render directory: {complete_dir}")
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Missing ground-truth directory: {ground_truth_dir}")

    pairs: list[tuple[int, int, Path, Path]] = []
    histogram = np.zeros(4096, dtype=np.int64)
    statistics: list[dict[str, float | int | str]] = []
    for render_index, render_path in render_paths(complete_dir):
        frame_index = args.frame_start + render_index * args.frame_step
        target_path = ground_truth_path(ground_truth_dir, frame_index)
        render = rgb_array(render_path)
        target = rgb_array(target_path, size=(render.shape[1], render.shape[0]))
        mae = pixel_mae(render, target)
        histogram += np.histogram(mae, bins=len(histogram), range=(0.0, 1.0))[0]
        mse = float(np.mean(((render.astype(np.float32) - target.astype(np.float32)) / 255.0) ** 2))
        statistics.append(
            {
                "render_index": render_index,
                "frame_index": frame_index,
                "render_path": str(render_path),
                "ground_truth_path": str(target_path),
                "mean_mae": float(np.mean(mae)),
                "median_mae": float(np.median(mae)),
                "p90_mae": float(np.quantile(mae, 0.90)),
                "p95_mae": float(np.quantile(mae, 0.95)),
                "p99_mae": float(np.quantile(mae, 0.99)),
                "max_mae": float(np.max(mae)),
                "rmse": math.sqrt(mse),
                "psnr": float("inf") if mse == 0.0 else -10.0 * math.log10(mse),
            }
        )
        pairs.append((render_index, frame_index, render_path, target_path))

    display_cap = args.display_cap or percentile_from_histogram(histogram, args.display_percentile)
    heatmap_dir = output_dir / "heatmap"
    overlay_dir = output_dir / "overlay"
    panel_dir = output_dir / "panel"
    mae16_dir = output_dir / "mae16"
    for directory in (heatmap_dir, overlay_dir, panel_dir, mae16_dir):
        directory.mkdir(parents=True, exist_ok=True)

    panel_paths: dict[int, Path] = {}
    for pair, frame_statistics in zip(pairs, statistics):
        render_index, frame_index, render_path, target_path = pair
        render = rgb_array(render_path)
        target = rgb_array(target_path, size=(render.shape[1], render.shape[0]))
        mae = pixel_mae(render, target)
        scaled = np.clip(mae / display_cap, 0.0, 1.0)
        heatmap = turbo(scaled)
        alpha = args.overlay_opacity * np.sqrt(scaled)[..., None]
        overlay = np.rint(target * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
        stem = f"{render_index:02d}_frame_{frame_index:03d}"
        Image.fromarray(heatmap).save(heatmap_dir / f"{stem}_mae_heatmap.png")
        Image.fromarray(overlay).save(overlay_dir / f"{stem}_mae_overlay.png")
        Image.fromarray(np.rint(mae * 65535.0).astype(np.uint16), mode="I;16").save(
            mae16_dir / f"{stem}_mae16.png"
        )
        panel = inspection_panel(
            target,
            render,
            heatmap,
            overlay,
            frame_index,
            float(frame_statistics["mean_mae"]),
            float(frame_statistics["p95_mae"]),
            display_cap,
        )
        panel_path = panel_dir / f"{stem}_panel.jpg"
        panel.save(panel_path, quality=94)
        panel_paths[frame_index] = panel_path

    with (output_dir / "frame_error_statistics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "run_dir": str(run_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "frame_start": args.frame_start,
        "frame_step": args.frame_step,
        "frame_count": len(pairs),
        "error_definition": "Per-pixel mean absolute error over RGB channels, normalized to [0,1].",
        "mae16_encoding": "round(normalized_pixel_mae * 65535); divide stored value by 65535 for MAE.",
        "heatmap_scale": {
            "colormap": "Turbo",
            "minimum": 0.0,
            "maximum": display_cap,
            "maximum_source": "explicit" if args.display_cap is not None else f"sequence percentile {args.display_percentile}",
        },
        "sequence_mean_mae": float(np.mean([float(item["mean_mae"]) for item in statistics])),
        "sequence_mean_psnr": float(np.mean([float(item["psnr"]) for item in statistics])),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    worst = sorted(statistics, key=lambda item: float(item["mean_mae"]), reverse=True)[: args.worst_frame_count]
    contact_sheet(
        [panel_paths[int(item["frame_index"])] for item in worst],
        [f"Frame {int(item['frame_index'])} - mean MAE {float(item['mean_mae']):.4f}" for item in worst],
        output_dir / "worst_frames_contact_sheet.jpg",
    )
    print(f"Created per-pixel error diagnostics for {len(pairs)} frames: {output_dir}")
    print(f"Fixed heatmap scale: 0 to {display_cap:.6f} normalized RGB MAE")


if __name__ == "__main__":
    main()
