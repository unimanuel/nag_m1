#!/usr/bin/env python3
"""Export the effective NAG sphere and ellipsoid background surfaces as PLY meshes.

This is deliberately independent of a training run: it reads the source
ellipsoid JSON written by ``nagwp ellipsoid`` and applies the same runtime
scale margin and per-axis cap that ``load_ellipsoid_geometry`` applies in NAG.
The sphere center is the robust ellipsoid's camera-mean center, matching the
corresponding NAG sphere setup for these experiments.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np


SPHERE_RED = np.asarray([230, 45, 45], dtype=np.uint8)
ELLIPSOID_BLUE = np.asarray([45, 105, 230], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ellipsoid-json", required=True, type=Path,
                        help="Robust AR3 ellipsoid.json used by the NAG run.")
    parser.add_argument("--sphere-radius", required=True, type=float,
                        help="Final radius (m) of the matching NAG sphere background.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory that will receive the mesh PLY files.")
    parser.add_argument("--scale-margin", type=float, default=0.02,
                        help="Runtime ellipsoid scale margin used by NAG (default: 0.02).")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Per-axis ellipsoid cap in metres; default is --sphere-radius.")
    parser.add_argument("--latitude-segments", type=int, default=48,
                        help="Latitude subdivisions for each mesh (default: 48).")
    parser.add_argument("--longitude-segments", type=int, default=96,
                        help="Longitude subdivisions for each mesh (default: 96).")
    return parser.parse_args()


def mesh_surface(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    latitude_segments: int,
    longitude_segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an oriented ellipsoid mesh using NAG's column-vector rotation."""

    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, latitude_segments + 1)
    longitudes = np.linspace(0.0, 2.0 * np.pi, longitude_segments, endpoint=False)
    latitude, longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    unit = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    ).reshape(-1, 3)
    # Existing NAG geometry is ``center + rotation @ (radii * unit)`` for a
    # column vector.  Rows therefore use ``local @ rotation.T``.
    vertices = center + (unit * radii) @ rotation.T

    faces = []
    for latitude_index in range(latitude_segments):
        row = latitude_index * longitude_segments
        next_row = (latitude_index + 1) * longitude_segments
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            a, b = row + longitude_index, row + next_longitude
            c, d = next_row + longitude_index, next_row + next_longitude
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian triangle PLY with per-vertex RGB colour."""

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape == (3,):
        colors = np.broadcast_to(colors, vertices.shape)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or colors.shape != vertices.shape:
        raise ValueError("vertices and colors must both have shape N x 3.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape M x 3.")

    vertex_records = np.empty(
        len(vertices),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
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


def main() -> None:
    args = parse_args()
    if not args.ellipsoid_json.is_file():
        raise FileNotFoundError(f"Missing ellipsoid JSON: {args.ellipsoid_json}")
    if not np.isfinite(args.sphere_radius) or args.sphere_radius <= 0.0:
        raise ValueError("--sphere-radius must be finite and positive.")
    if not 0.0 <= args.scale_margin <= 1.0:
        raise ValueError("--scale-margin must be in [0, 1].")
    if args.latitude_segments < 3 or args.longitude_segments < 3:
        raise ValueError("Mesh segment counts must both be at least 3.")

    with args.ellipsoid_json.open("r", encoding="utf-8") as file:
        document = json.load(file)
    fit = document.get("fit", {})
    center = np.asarray(fit.get("center"), dtype=np.float64)
    source_radii = np.asarray(fit.get("radii"), dtype=np.float64)
    rotation = np.asarray(fit.get("rotation_columns"), dtype=np.float64)
    if center.shape != (3,) or source_radii.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("ellipsoid JSON must contain fit.center, fit.radii, and fit.rotation_columns.")
    if not (np.isfinite(center).all() and np.isfinite(source_radii).all() and np.isfinite(rotation).all()):
        raise ValueError("Ellipsoid geometry contains non-finite values.")
    if np.any(source_radii <= 0.0):
        raise ValueError("Ellipsoid source radii must be positive.")

    max_radius = args.sphere_radius if args.max_radius is None else args.max_radius
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("--max-radius must be finite and positive.")
    effective_ellipsoid_radii = np.minimum(source_radii * (1.0 + args.scale_margin), max_radius)
    sphere_vertices, sphere_faces = mesh_surface(
        center, np.full(3, args.sphere_radius), np.eye(3),
        args.latitude_segments, args.longitude_segments,
    )
    ellipsoid_vertices, ellipsoid_faces = mesh_surface(
        center, effective_ellipsoid_radii, rotation,
        args.latitude_segments, args.longitude_segments,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_mesh_ply(args.output_dir / "sphere_red.ply", sphere_vertices, sphere_faces, SPHERE_RED)
    write_mesh_ply(
        args.output_dir / "ellipsoid_ar3_clipped_blue.ply",
        ellipsoid_vertices,
        ellipsoid_faces,
        ELLIPSOID_BLUE,
    )
    combined_vertices = np.concatenate((sphere_vertices, ellipsoid_vertices), axis=0)
    combined_faces = np.concatenate((sphere_faces, ellipsoid_faces + len(sphere_vertices)), axis=0)
    combined_colors = np.concatenate((
        np.broadcast_to(SPHERE_RED, sphere_vertices.shape),
        np.broadcast_to(ELLIPSOID_BLUE, ellipsoid_vertices.shape),
    ))
    write_mesh_ply(
        args.output_dir / "sphere_red_and_ellipsoid_ar3_clipped_blue.ply",
        combined_vertices,
        combined_faces,
        combined_colors,
    )
    metadata = {
        "coordinate_frame": document.get("coordinate_frame"),
        "source_ellipsoid_json": str(args.ellipsoid_json.resolve()),
        "sequence": document.get("sequence"),
        "frame_slice": document.get("frame_slice"),
        "center": center.tolist(),
        "sphere": {"color_rgb": SPHERE_RED.tolist(), "radius": args.sphere_radius},
        "ellipsoid": {
            "color_rgb": ELLIPSOID_BLUE.tolist(),
            "source_radii": source_radii.tolist(),
            "scale_margin": args.scale_margin,
            "max_radius": max_radius,
            "effective_radii": effective_ellipsoid_radii.tolist(),
            "rotation_columns": rotation.tolist(),
        },
        "mesh_resolution": {
            "latitude_segments": args.latitude_segments,
            "longitude_segments": args.longitude_segments,
            "vertices_per_surface": int(len(sphere_vertices)),
            "triangles_per_surface": int(len(sphere_faces)),
        },
    }
    metadata_path = args.output_dir / "background_geometry_meshes.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote red sphere: {args.output_dir / 'sphere_red.ply'}")
    print(f"Wrote blue clipped AR3 ellipsoid: {args.output_dir / 'ellipsoid_ar3_clipped_blue.ply'}")
    print(f"Wrote combined comparison mesh: {args.output_dir / 'sphere_red_and_ellipsoid_ar3_clipped_blue.ply'}")
    print(f"Sphere radius: {args.sphere_radius:.4f} m")
    print(f"Effective ellipsoid radii: {effective_ellipsoid_radii.tolist()} m")


if __name__ == "__main__":
    main()
