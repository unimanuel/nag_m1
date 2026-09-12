#!/usr/bin/env python3
"""Fit a constrained reprojection/depth ellipsoid from cached diagnostics.

The generated JSON intentionally uses the runtime-compatible ray-proxy format
version.  It can be passed directly to ``run_nag.py --background-ellipsoid-path``.
No training is launched here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nag"))
sys.path.insert(0, str(REPO / "libs" / "nag-waymo-processing"))

from diagnostics.parallax_io import SEQUENCES, jsonable
from diagnostics.parallax_observations import check_prepared
from reprojection_ellipsoid import ReprojectionFitConfig, fit_reprojection_depth_ellipsoid


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_geometry(document: dict[str, Any]) -> dict[str, np.ndarray]:
    fit = document.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("Source ellipsoid JSON has no fit mapping.")
    return {
        "center": np.asarray(fit["center"], dtype=np.float64),
        "radii": np.asarray(fit["radii"], dtype=np.float64),
        "rotation": np.asarray(fit["rotation_columns"], dtype=np.float64),
    }


def _validate_contract(label: str, prepared: dict[str, Any], source: dict[str, Any]) -> tuple[str, int, int]:
    if prepared.get("scope") != "full_interval":
        raise ValueError(
            f"{label}: prepared observations have scope {prepared.get('scope')!r}; "
            "run the full prepare stage before fitting."
        )
    segment, start, stop = SEQUENCES[label]
    if source.get("sequence") != f"segment-{segment}":
        raise ValueError(f"{label}: source geometry segment does not match the requested sequence.")
    frame_slice = source.get("frame_slice", {})
    if (frame_slice.get("start"), frame_slice.get("stop"), frame_slice.get("step")) != (start, stop, 1):
        raise ValueError(f"{label}: source geometry frame slice does not match the requested interval.")
    return segment, start, stop


def _copy_inspection_assets(source_path: Path, output_dir: Path) -> list[str]:
    copied = []
    for name in ("front_points.ply", "background_points.ply", "camera_centers.ply"):
        source = source_path.parent / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing source inspection asset: {source}")
        shutil.copy2(source, output_dir / name)
        copied.append(name)
    return copied


def _write_mesh(path: Path, geometry: dict[str, np.ndarray]) -> None:
    # Keep the candidate mesh byte-compatible with the established ellipsoid
    # inspection files rather than introducing another PLY convention.
    from nag_waymo_processing.ellipsoid import _ellipsoid_mesh, _write_mesh_ply

    vertices, faces = _ellipsoid_mesh(geometry["center"], geometry["radii"], geometry["rotation"])
    _write_mesh_ply(str(path), vertices, faces)


def _document(
    source: dict[str, Any],
    fit: dict[str, Any],
    source_path: Path,
    prepared_path: Path,
    label: str,
) -> dict[str, Any]:
    document = copy.deepcopy(source)
    geometry = fit["geometry"]
    metrics = fit["diagnostics"]["final"]
    initial = fit["diagnostics"]["initial_safety_projected"]
    document["format_version"] = 4
    document["point_selection"] = (
        "cached static FRONT-background LiDAR observations with target-frame "
        "sparse-depth support; foreground-mask pixels and unsupported targets are excluded"
    )
    document["fit"] = {
        "mode": "reprojection_depth_proxy",
        "method": (
            "fixed-road-orientation, symmetric cross-frame reprojection ellipsoid "
            "with weak log-depth anchor and safety-constrained radii"
        ),
        "uses_all_selected_points": False,
        "is_minimum_volume": False,
        "center": geometry["center"],
        "radii": geometry["radii"],
        "diameters": geometry["radii"] * 2.0,
        "rotation_columns": geometry["rotation"],
        "base_half_extents": geometry["radii"],
        "uniform_scale": 1.0,
        "max_normalized_radius": None,
        "contains_all_points": False,
        "point_coverage_fraction": 0.0,
        "target_max_camera_normalized_radius": fit["config"]["max_camera_normalized_radius"],
        "max_camera_normalized_radius": metrics["max_camera_normalized_radius"],
        "target_minimum_camera_surface_distance": fit["config"]["minimum_camera_surface_distance"],
        "conservative_minimum_camera_surface_distance": metrics["conservative_camera_clearance_m"],
        "minimum_axis_radius": fit["config"]["minimum_axis_radius"],
        "max_axis_ratio": fit["config"]["max_axis_ratio"],
        "actual_axis_ratio": metrics["maximum_axis_ratio"],
        "reprojection_weight": fit["config"]["reprojection_weight"],
        "depth_weight": fit["config"]["depth_weight"],
        "reprojection_huber_px": fit["config"]["reprojection_huber_px"],
        "depth_huber_log": fit["config"]["depth_huber_log"],
    }
    document["reprojection_fit"] = {
        "schema": 1,
        "label": label,
        "prepared_observations": str(prepared_path.resolve()),
        "source_geometry": str(source_path.resolve()),
        "optimization_pair_count": fit["optimization_pair_count"],
        "optimization_depth_ray_count": fit["optimization_depth_ray_count"],
        "config": fit["config"],
        "initial_safety_projected_metrics": initial,
        "final_metrics": metrics,
        "candidates": fit["candidates"],
        "texture_orientation_policy": "not fitted; NAG training uses legacy orientation explicitly",
        "sphere_clipping": "not applied",
    }
    # Do not retain the source raw-v4 depth report: it would describe a
    # different ellipsoid and could be mistaken for the combined candidate.
    document["ray_depth"] = {
        "selection": "distinct cached static-background source rays used by the weak depth anchor",
        "metrics": {
            "valid_fraction": metrics["valid_depth_fraction"],
            "median_absolute_log_depth_error": metrics["absolute_log_depth_error"]["median"],
            "p90_absolute_log_depth_error": metrics["absolute_log_depth_error"]["p90"],
        },
    }
    document["files"] = {
        "front_points_ply": "front_points.ply",
        "background_points_ply": "background_points.ply",
        "camera_centers_ply": "camera_centers.ply",
        "ellipsoid_mesh_ply": "ellipsoid.ply",
        "reprojection_fit_diagnostics_json": "reprojection_fit_diagnostics.json",
        "reprojection_fit_comparison_csv": "reprojection_fit_comparison.csv",
    }
    return document


def _write_diagnostics(output_dir: Path, fit: dict[str, Any], label: str, source_path: Path, prepared_path: Path) -> None:
    diagnostics = {
        "schema": 1,
        "label": label,
        "source_geometry": str(source_path.resolve()),
        "prepared_observations": str(prepared_path.resolve()),
        "optimization_pair_count": fit["optimization_pair_count"],
        "optimization_depth_ray_count": fit["optimization_depth_ray_count"],
        "config": fit["config"],
        "initial_safety_projected": fit["diagnostics"]["initial_safety_projected"],
        "final": fit["diagnostics"]["final"],
        "candidates": fit["candidates"],
        "interpretation": (
            "This pre-training result measures fixed-geometry LiDAR correspondence only. "
            "It does not predict learned flow, appearance, or final NAG quality."
        ),
    }
    _write_json(output_dir / "reprojection_fit_diagnostics.json", diagnostics)
    fields = (
        "candidate", "pixel_error_median_px", "pixel_error_p90_px", "camera_angle_median_deg",
        "absolute_log_depth_median", "reprojection_huber_loss", "depth_huber_loss",
        "weighted_data_loss", "valid_pair_fraction", "max_camera_normalized_radius",
        "conservative_camera_clearance_m", "minimum_axis_radius_m", "maximum_axis_ratio",
    )
    rows = []
    for name, metrics in (("initial_safety_projected", fit["diagnostics"]["initial_safety_projected"]), ("combined_fit", fit["diagnostics"]["final"])):
        rows.append({
            "candidate": name,
            "pixel_error_median_px": metrics["pixel_error"]["median"],
            "pixel_error_p90_px": metrics["pixel_error"]["p90"],
            "camera_angle_median_deg": metrics["camera_angle_deg"]["median"],
            "absolute_log_depth_median": metrics["absolute_log_depth_error"]["median"],
            "reprojection_huber_loss": metrics["reprojection_huber_loss"],
            "depth_huber_loss": metrics["depth_huber_loss"],
            "weighted_data_loss": metrics["weighted_data_loss"],
            "valid_pair_fraction": metrics["valid_pair_fraction"],
            "max_camera_normalized_radius": metrics["max_camera_normalized_radius"],
            "conservative_camera_clearance_m": metrics["conservative_camera_clearance_m"],
            "minimum_axis_radius_m": metrics["minimum_axis_radius_m"],
            "maximum_axis_ratio": metrics["maximum_axis_ratio"],
        })
    with (output_dir / "reprojection_fit_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=sorted(SEQUENCES))
    parser.add_argument("--prepared", type=Path, required=True, help="Sequence prepared.json from parallax_v1.")
    parser.add_argument("--source-ellipsoid", type=Path, required=True, help="Existing raw-v4 ellipsoid.json used only as initialization.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=24000)
    parser.add_argument("--max-depth-rays", type=int, default=14000)
    parser.add_argument("--max-camera-normalized-radius", type=float, default=0.65)
    parser.add_argument("--minimum-camera-surface-distance", type=float, default=6.0)
    parser.add_argument("--minimum-axis-radius", type=float, default=8.0)
    parser.add_argument("--max-axis-ratio", type=float, default=10.0)
    parser.add_argument("--reprojection-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.15)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory already contains files: {args.output_dir}; pass --overwrite to replace this candidate only.")
    # A wrapper dry run intentionally performs no cache rewrite.  Its pointer
    # may be stale merely because a new run-name classifier was added; the real
    # preparation command directly before fitting refreshes that provenance.
    prepared = _load_json(args.prepared) if args.dry_run else check_prepared(args.prepared, REPO)
    source = _load_json(args.source_ellipsoid)
    segment, start, stop = _validate_contract(args.label, prepared, source)
    pairs_path = Path(prepared["directory"]) / "pairs.npz"
    if not pairs_path.is_file():
        parser.error(f"Prepared pair cache is missing: {pairs_path}")
    config = ReprojectionFitConfig(
        max_pairs=args.max_pairs,
        max_depth_rays=args.max_depth_rays,
        seed=args.seed,
        reprojection_weight=args.reprojection_weight,
        depth_weight=args.depth_weight,
        max_camera_normalized_radius=args.max_camera_normalized_radius,
        minimum_camera_surface_distance=args.minimum_camera_surface_distance,
        minimum_axis_radius=args.minimum_axis_radius,
        max_axis_ratio=args.max_axis_ratio,
        max_iterations=args.max_iterations,
    )
    if args.dry_run:
        print(f"{args.label}: source={args.source_ellipsoid}")
        print(f"{args.label}: pairs={pairs_path}; output={args.output_dir}")
        print(f"{args.label}: interval=segment-{segment} frames {start}-{stop - 1}; config={config}")
        return
    with np.load(pairs_path, allow_pickle=False) as archive:
        pairs = {name: archive[name] for name in archive.files}
    fit = fit_reprojection_depth_ellipsoid(pairs, _source_geometry(source), config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_inspection_assets(args.source_ellipsoid, args.output_dir)
    _write_mesh(args.output_dir / "ellipsoid.ply", fit["geometry"])
    _write_diagnostics(args.output_dir, fit, args.label, args.source_ellipsoid, args.prepared)
    document = _document(source, fit, args.source_ellipsoid, args.prepared, args.label)
    _write_json(args.output_dir / "ellipsoid.json", document)
    final = fit["diagnostics"]["final"]
    print(f"{args.label}: wrote {args.output_dir / 'ellipsoid.json'}")
    print(f"{args.label}: copied {', '.join(copied)}; regenerated ellipsoid.ply")
    print(
        f"{args.label}: median reprojection={final['pixel_error']['median']:.3f}px, "
        f"P90={final['pixel_error']['p90']:.3f}px, "
        f"clearance={final['conservative_camera_clearance_m']:.3f}m, "
        f"axis ratio={final['maximum_axis_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
