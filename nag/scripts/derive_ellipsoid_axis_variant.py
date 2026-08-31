#!/usr/bin/env python3
"""Create a self-contained axis-scaled variant of an existing ellipsoid fit.

The tool is deliberately a geometry derivation, not a re-fit: the source
centre and orientation stay unchanged and the supplied local radii replace the
source radii exactly.  It writes a new JSON, a matching triangle-mesh PLY, and
copies the source point/camera PLYs so the folder can be opened independently
in a point-cloud viewer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ELLIPSOID_ORANGE = np.asarray([255, 145, 30], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--radii",
        required=True,
        type=float,
        nargs=3,
        metavar=("FORWARD", "LATERAL", "UP"),
        help="Replacement radii in the source fit's local road-aligned frame.",
    )
    parser.add_argument("--label", required=True, help="Human-readable variant label.")
    return parser.parse_args()


def read_binary_point_ply(path: Path) -> np.ndarray:
    """Read XYZ vertices from the project binary point PLY convention."""

    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    vertex_count = next(
        int(line.split()[2]) for line in header.splitlines() if line.startswith("element vertex ")
    )
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertices = np.frombuffer(raw, dtype=vertex_dtype, count=vertex_count, offset=header_end)
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def mesh_surface(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    latitude_segments: int = 24,
    longitude_segments: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    for latitude_index in range(latitude_segments + 1):
        latitude = -np.pi / 2.0 + np.pi * latitude_index / latitude_segments
        for longitude_index in range(longitude_segments):
            longitude = 2.0 * np.pi * longitude_index / longitude_segments
            unit = np.asarray(
                [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]
            )
            vertices.append(center + rotation @ (radii * unit))

    faces: list[tuple[int, int, int]] = []
    for latitude_index in range(latitude_segments):
        row = latitude_index * longitude_segments
        next_row = (latitude_index + 1) * longitude_segments
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            a, b = row + longitude_index, row + next_longitude
            c, d = next_row + longitude_index, next_row + next_longitude
            faces.extend(((a, c, b), (b, c, d)))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    colors = np.broadcast_to(ELLIPSOID_ORANGE, vertices.shape)
    vertex_records = np.empty(
        len(vertices),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex_records["x"], vertex_records["y"], vertex_records["z"] = vertices.T
    vertex_records["red"], vertex_records["green"], vertex_records["blue"] = colors.T
    face_records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_records["count"] = 3
    face_records["indices"] = faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        file.write(header)
        vertex_records.tofile(file)
        face_records.tofile(file)
    os.replace(temporary_path, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    source_json = args.source_json.resolve()
    output_dir = args.output_dir.resolve()
    radii = np.asarray(args.radii, dtype=np.float64)
    if not source_json.is_file():
        raise FileNotFoundError(f"Missing source ellipsoid JSON: {source_json}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("All replacement radii must be finite and positive.")

    with source_json.open("r", encoding="utf-8") as file:
        source_document = json.load(file)
    document = copy.deepcopy(source_document)
    fit = document.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("Source JSON does not contain a fit object.")
    center = np.asarray(fit.get("center"), dtype=np.float64)
    rotation = np.asarray(fit.get("rotation_columns"), dtype=np.float64)
    if center.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Source fit must contain a 3D center and 3x3 rotation_columns.")

    source_dir = source_json.parent
    output_dir.mkdir(parents=True)
    copied_files: dict[str, str] = {}
    for source_name in ("front_points.ply", "background_points.ply", "camera_centers.ply"):
        source_path = source_dir / source_name
        if source_path.is_file():
            shutil.copy2(source_path, output_dir / source_name)
            copied_files[source_name] = source_name

    source_points_name = "background_points.ply" if (output_dir / "background_points.ply").is_file() else "front_points.ply"
    points = read_binary_point_ply(output_dir / source_points_name)
    local_points = (points - center) @ rotation
    normalized_radii = np.linalg.norm(local_points / radii, axis=1)
    fit["radii"] = radii.tolist()
    fit["diameters"] = (2.0 * radii).tolist()
    fit["base_half_extents"] = radii.tolist()
    fit["max_normalized_radius"] = float(normalized_radii.max())
    fit["contains_all_points"] = bool(normalized_radii.max() <= 1.0 + 1e-12)
    fit["point_coverage_fraction"] = float(np.mean(normalized_radii <= 1.0))
    fit["method"] = f"{fit.get('method', 'ellipsoid')} (axis-scaled variant: {args.label})"

    # Ray-depth metrics in the source document apply only to the unmodified
    # surface.  Do not carry them forward as if they described this variant.
    document.pop("ray_depth", None)
    document["derived_variant"] = {
        "label": args.label,
        "source_ellipsoid_json": str(source_json),
        "operation": "Replaced local [forward, lateral, up] radii; center and rotation unchanged.",
        "source_radii": source_document["fit"]["radii"],
        "replacement_radii": radii.tolist(),
        "point_coverage_source": source_points_name,
        "ray_depth_metrics_recomputed": False,
    }
    document["files"] = {
        **copied_files,
        "ellipsoid_mesh_ply": "ellipsoid.ply",
    }

    vertices, faces = mesh_surface(center, radii, rotation)
    write_mesh_ply(output_dir / "ellipsoid.ply", vertices, faces)
    with (output_dir / "ellipsoid.json").open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, default=json_default)
        file.write("\n")
    print(f"Created {output_dir}")
    print(f"  label: {args.label}")
    print(f"  radii [forward, lateral, up]: {radii.tolist()}")
    print(f"  point coverage: {fit['point_coverage_fraction']:.6f}")


if __name__ == "__main__":
    main()
