"""Constrained cross-frame reprojection fitting for an oriented ellipsoid.

This module deliberately consumes the cached, visibility-supported observation
pairs produced by :mod:`nag.diagnostics.parallax_observations`.  It is separate
from the Waymo point-cloud extraction package because the cache supplies the
critical target-view visibility, undistorted intrinsics, and NAG-coordinate
camera poses needed for a multi-frame reprojection objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import numpy as np
from scipy.optimize import minimize

from diagnostics.parallax_core import angle_degrees, intersect, unit


@dataclass(frozen=True)
class ReprojectionFitConfig:
    """Settings for original or source-scale-regularized combined-fit experiments."""

    max_pairs: int = 24000
    max_depth_rays: int = 14000
    seed: int = 20260912
    reprojection_weight: float = 1.0
    depth_weight: float = 0.15
    reprojection_huber_px: float = 8.0
    depth_huber_log: float = 0.20
    radius_excess_weight: float = 0.0
    radius_excess_huber_log: float = 0.20
    max_camera_normalized_radius: float = 0.65
    minimum_camera_surface_distance: float = 6.0
    minimum_axis_radius: float = 8.0
    max_axis_ratio: float = 10.0
    max_iterations: int = 200


def _array_geometry(geometry: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Validate and normalize a runtime-compatible ellipsoid dictionary."""

    result = {
        "center": np.asarray(geometry["center"], dtype=np.float64),
        "radii": np.asarray(geometry["radii"], dtype=np.float64),
        "rotation": np.asarray(geometry["rotation"], dtype=np.float64),
    }
    if result["center"].shape != (3,) or result["radii"].shape != (3,) or result["rotation"].shape != (3, 3):
        raise ValueError("Ellipsoid geometry must contain center (3), radii (3), and rotation (3,3).")
    if not all(np.isfinite(value).all() for value in result.values()) or np.any(result["radii"] <= 0.0):
        raise ValueError("Ellipsoid geometry must be finite with positive radii.")
    if not np.allclose(result["rotation"].T @ result["rotation"], np.eye(3), atol=1e-6):
        raise ValueError("Ellipsoid rotation must be orthonormal.")
    if np.linalg.det(result["rotation"]) <= 0.0:
        raise ValueError("Ellipsoid rotation must be right-handed.")
    return result


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * values * values / delta, absolute - 0.5 * delta)


def _project_batched(points: np.ndarray, poses: np.ndarray, ids: np.ndarray, intrinsics: np.ndarray):
    """Project row-wise world points through NAG camera-to-world poses."""

    local = np.einsum("ni,nij->nj", points - poses[ids, :3, 3], poses[ids, :3, :3])
    homogeneous = local @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, local[:, 2]


def _sample_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, size=maximum, replace=False))


def _camera_safety(geometry: Mapping[str, Any], poses: np.ndarray) -> Dict[str, float]:
    geometry = _array_geometry(geometry)
    local = ((poses[:, :3, 3] - geometry["center"]) @ geometry["rotation"]) / geometry["radii"]
    normalized = np.linalg.norm(local, axis=1)
    maximum = float(normalized.max())
    return {
        "max_camera_normalized_radius": maximum,
        "conservative_camera_clearance_m": float(geometry["radii"].min() * (1.0 - maximum)),
        "minimum_axis_radius_m": float(geometry["radii"].min()),
        "maximum_axis_ratio": float(geometry["radii"].max() / geometry["radii"].min()),
    }


def enforce_safety(
    geometry: Mapping[str, Any],
    poses: np.ndarray,
    config: ReprojectionFitConfig,
) -> Dict[str, np.ndarray]:
    """Expand radii only, making the reported geometric safeguards hard limits.

    The fit itself has smooth penalties for these limits.  This final projection
    makes them exact even when L-BFGS-B terminates close to a hinge boundary.
    It never applies a sphere-radius cap or changes orientation.
    """

    result = _array_geometry(geometry)
    radii = np.maximum(result["radii"], config.minimum_axis_radius)
    radii = np.maximum(radii, float(radii.max()) / config.max_axis_ratio)
    provisional = {**result, "radii": radii}
    safety = _camera_safety(provisional, poses)
    normalized = safety["max_camera_normalized_radius"]
    minimum_radius = float(radii.min())
    scale = max(
        1.0,
        normalized / config.max_camera_normalized_radius,
        normalized + config.minimum_camera_surface_distance / minimum_radius,
    )
    return {**result, "radii": radii * scale}


def _pair_residuals(pairs: Mapping[str, np.ndarray], geometry: Mapping[str, Any], pair_indices: np.ndarray):
    """Symmetric source-to-target and target-to-source pixel residuals."""

    points = np.asarray(pairs["points"], dtype=np.float64)[pair_indices]
    source = np.asarray(pairs["source"], dtype=np.int64)[pair_indices]
    target = np.asarray(pairs["target"], dtype=np.int64)[pair_indices]
    poses = np.asarray(pairs["poses"], dtype=np.float64)
    intrinsics = np.asarray(pairs["intrinsics"], dtype=np.float64)
    centers = poses[:, :3, 3]

    source_direction = unit(points - centers[source])
    target_direction = unit(points - centers[target])
    source_proxy, _, source_hit = intersect(centers[source], source_direction, geometry)
    target_proxy, _, target_hit = intersect(centers[target], target_direction, geometry)
    real_in_target, _ = _project_batched(points, poses, target, intrinsics)
    real_in_source, _ = _project_batched(points, poses, source, intrinsics)
    source_in_target, source_target_depth = _project_batched(source_proxy, poses, target, intrinsics)
    target_in_source, target_source_depth = _project_batched(target_proxy, poses, source, intrinsics)
    valid = (
        source_hit
        & target_hit
        & (source_target_depth > 1e-6)
        & (target_source_depth > 1e-6)
        & np.isfinite(source_in_target).all(axis=1)
        & np.isfinite(target_in_source).all(axis=1)
    )
    forward = np.linalg.norm(source_in_target - real_in_target, axis=1)
    reverse = np.linalg.norm(target_in_source - real_in_source, axis=1)
    errors = (forward + reverse) / 2.0
    errors[~valid] = np.nan
    camera_angle = (
        angle_degrees(source_proxy - centers[target], points - centers[target])
        + angle_degrees(target_proxy - centers[source], points - centers[source])
    ) / 2.0
    camera_angle[~valid] = np.nan
    return {"pixel_error": errors, "camera_angle_deg": camera_angle, "valid": valid}


def _depth_residuals(pairs: Mapping[str, np.ndarray], geometry: Mapping[str, Any], source_indices: np.ndarray):
    """Absolute log-depth residuals for distinct sampled source observations."""

    points = np.asarray(pairs["source_points"], dtype=np.float64)[source_indices]
    ids = np.asarray(pairs["source_ids"], dtype=np.int64)[source_indices]
    poses = np.asarray(pairs["poses"], dtype=np.float64)
    origins = poses[ids, :3, 3]
    directions = unit(points - origins)
    _, predicted, hit = intersect(origins, directions, geometry)
    observed = np.linalg.norm(points - origins, axis=1)
    residual = np.full(len(points), np.nan, dtype=np.float64)
    valid = hit & np.isfinite(predicted) & (observed > 1e-6)
    residual[valid] = np.log(predicted[valid] / observed[valid])
    return {"log_depth_error": residual, "valid": valid}


def _radius_excess_log(geometry: Mapping[str, Any], reference_max_axis_radius: float) -> np.ndarray:
    """Non-negative log excess above a source-derived global radius scale.

    The reference is the largest radius of the safety-projected raw-v4
    initializer.  It is deliberately *not* an axis-wise prior: an ellipsoid
    may freely redistribute its size between forward, vertical, and lateral
    axes.  The term only rejects an axis becoming globally much larger than
    the conservative source geometry, a failure mode that symmetric
    reprojection leaves weakly determined on near-straight intervals.
    """

    if not np.isfinite(reference_max_axis_radius) or reference_max_axis_radius <= 0.0:
        raise ValueError("reference_max_axis_radius must be finite and positive.")
    radii = _array_geometry(geometry)["radii"]
    return np.maximum(0.0, np.log(radii / reference_max_axis_radius))


def regularizer_version(config: ReprojectionFitConfig) -> str:
    """Stable user-facing name for the fitted objective variant."""

    return "new_radius_scale" if config.radius_excess_weight > 0.0 else "original"


def _finite_distribution(values: np.ndarray) -> Dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "invalid_count": int(len(values) - len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.90)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
    }


def evaluate_geometry(
    pairs: Mapping[str, np.ndarray],
    geometry: Mapping[str, Any],
    pair_indices: np.ndarray,
    source_indices: np.ndarray,
    config: ReprojectionFitConfig,
    reference_max_axis_radius: float | None = None,
) -> Dict[str, Any]:
    """Measure the actual terms and safety state of a candidate geometry."""

    geometry = _array_geometry(geometry)
    pair = _pair_residuals(pairs, geometry, pair_indices)
    depth = _depth_residuals(pairs, geometry, source_indices)
    pair_valid = pair["valid"]
    depth_valid = depth["valid"]
    reprojection_loss = (
        float(_huber(pair["pixel_error"][pair_valid], config.reprojection_huber_px).mean())
        if pair_valid.any()
        else float("inf")
    )
    depth_loss = (
        float(_huber(depth["log_depth_error"][depth_valid], config.depth_huber_log).mean())
        if depth_valid.any()
        else float("inf")
    )
    reference = float(geometry["radii"].max()) if reference_max_axis_radius is None else reference_max_axis_radius
    radius_excess = _radius_excess_log(geometry, reference)
    radius_excess_loss = float(_huber(radius_excess, config.radius_excess_huber_log).mean())
    safety = _camera_safety(geometry, np.asarray(pairs["poses"], dtype=np.float64))
    return {
        "geometry": {key: value.tolist() for key, value in geometry.items()},
        "pixel_error": _finite_distribution(pair["pixel_error"]),
        "camera_angle_deg": _finite_distribution(pair["camera_angle_deg"]),
        "absolute_log_depth_error": _finite_distribution(np.abs(depth["log_depth_error"])),
        "valid_pair_fraction": float(pair_valid.mean()),
        "valid_depth_fraction": float(depth_valid.mean()),
        "reprojection_huber_loss": reprojection_loss,
        "depth_huber_loss": depth_loss,
        "radius_reference_max_axis_radius_m": reference,
        "radius_excess_log": _finite_distribution(radius_excess),
        "radius_excess_huber_loss": radius_excess_loss,
        "weighted_data_loss": float(
            config.reprojection_weight * reprojection_loss
            + config.depth_weight * depth_loss
            + config.radius_excess_weight * radius_excess_loss
        ),
        **safety,
    }


def fit_reprojection_depth_ellipsoid(
    pairs: Mapping[str, np.ndarray],
    initial_geometry: Mapping[str, Any],
    config: ReprojectionFitConfig = ReprojectionFitConfig(),
) -> Dict[str, Any]:
    """Fit a fixed-orientation ellipsoid with reprojection as the dominant term.

    The cached pairs are already static-background selected and have source and
    target LiDAR support.  The optimizer therefore does not tailor itself to a
    number of streets or turns: every selected pair contributes symmetrically,
    irrespective of its heading change.  The weak log-depth term anchors areas
    with little useful camera baseline.  The final safety projection forbids
    the very thin/near-camera raw-v4 solution, without sphere clipping.
    """

    if config.max_pairs < 128 or config.max_depth_rays < 128:
        raise ValueError("max_pairs and max_depth_rays must each be at least 128.")
    if not (0.0 < config.max_camera_normalized_radius < 1.0):
        raise ValueError("max_camera_normalized_radius must lie in (0, 1).")
    if min(config.minimum_camera_surface_distance, config.minimum_axis_radius) <= 0.0:
        raise ValueError("minimum camera clearance and axis radius must be positive.")
    if config.max_axis_ratio < 1.0:
        raise ValueError("max_axis_ratio must be at least one.")
    if min(config.reprojection_weight, config.depth_weight, config.reprojection_huber_px,
           config.depth_huber_log, config.radius_excess_huber_log) <= 0.0 or config.radius_excess_weight < 0.0:
        raise ValueError("Weights and Huber thresholds must be non-negative with positive active thresholds.")

    required = {"points", "source", "target", "visibility", "poses", "intrinsics", "source_points", "source_ids"}
    missing = sorted(required.difference(pairs))
    if missing:
        raise ValueError(f"Prepared pair cache is missing fields: {missing}")
    supported = np.flatnonzero(np.asarray(pairs["visibility"]) == 1)
    if len(supported) < 128:
        raise ValueError("At least 128 visibility-supported cross-frame pairs are required.")
    pair_indices = supported[_sample_indices(len(supported), config.max_pairs, config.seed)]
    source_indices = _sample_indices(len(pairs["source_points"]), config.max_depth_rays, config.seed + 1)
    poses = np.asarray(pairs["poses"], dtype=np.float64)
    initial = enforce_safety(initial_geometry, poses, config)
    reference_max_axis_radius = float(initial["radii"].max())
    rotation = initial["rotation"]
    camera_mean = poses[:, :3, 3].mean(axis=0)

    def decode(parameters: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "center": camera_mean + rotation @ parameters[:3],
            "radii": np.exp(parameters[3:]),
            "rotation": rotation,
        }

    def objective(parameters: np.ndarray) -> float:
        geometry = decode(parameters)
        pair = _pair_residuals(pairs, geometry, pair_indices)
        depth = _depth_residuals(pairs, geometry, source_indices)
        pair_valid = pair["valid"]
        depth_valid = depth["valid"]
        invalid_pair_fraction = 1.0 - float(pair_valid.mean())
        invalid_depth_fraction = 1.0 - float(depth_valid.mean())
        if not pair_valid.any() or not depth_valid.any():
            return 1e9
        reprojection = float(_huber(pair["pixel_error"][pair_valid], config.reprojection_huber_px).mean())
        ray_depth = float(_huber(depth["log_depth_error"][depth_valid], config.depth_huber_log).mean())
        radius_excess_loss = float(_huber(
            _radius_excess_log(geometry, reference_max_axis_radius), config.radius_excess_huber_log
        ).mean())
        safety = _camera_safety(geometry, poses)
        ratio_penalty = max(0.0, np.log(safety["maximum_axis_ratio"] / config.max_axis_ratio))
        penalty = (
            2e4 * max(0.0, safety["max_camera_normalized_radius"] - config.max_camera_normalized_radius) ** 2
            + 5e2 * max(0.0, config.minimum_camera_surface_distance - safety["conservative_camera_clearance_m"]) ** 2
            + 5e2 * max(0.0, config.minimum_axis_radius - safety["minimum_axis_radius_m"]) ** 2
            + 5e2 * ratio_penalty**2
            + 1e5 * (invalid_pair_fraction + invalid_depth_fraction)
        )
        return (
            config.reprojection_weight * reprojection
            + config.depth_weight * ray_depth
            + config.radius_excess_weight * radius_excess_loss
            + penalty
        )

    initial_offset = (initial["center"] - camera_mean) @ rotation
    starts = []
    for scale in (1.0, 1.25, 1.75):
        starts.append(np.concatenate((initial_offset, np.log(initial["radii"] * scale))))
    # A camera-mean candidate prevents all starts from assuming that the old
    # depth-only center was correct, while keeping the same fixed orientation.
    starts.append(np.concatenate((np.zeros(3), np.log(initial["radii"] * 1.25))))
    bounds = [(-80.0, 160.0), (-30.0, 30.0), (-20.0, 20.0)] + [(np.log(1.0), np.log(350.0))] * 3
    candidates = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.max_iterations, "maxls": 100, "ftol": 1e-10, "gtol": 1e-7},
        )
        raw = decode(result.x)
        repaired = enforce_safety(raw, poses, config)
        repaired_parameters = np.concatenate(
            ((repaired["center"] - camera_mean) @ rotation, np.log(repaired["radii"]))
        )
        candidates.append({
            "result": result,
            "raw_geometry": raw,
            "geometry": repaired,
            "post_projection_objective": float(objective(repaired_parameters)),
        })
    best = min(candidates, key=lambda item: item["post_projection_objective"])
    final = best["geometry"]
    diagnostics = {
        "initial_safety_projected": evaluate_geometry(
            pairs, initial, pair_indices, source_indices, config, reference_max_axis_radius,
        ),
        "final": evaluate_geometry(
            pairs, final, pair_indices, source_indices, config, reference_max_axis_radius,
        ),
    }
    candidate_summary = []
    for index, candidate in enumerate(candidates):
        result = candidate["result"]
        candidate_summary.append({
            "index": index,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit),
            "optimizer_objective": float(result.fun),
            "post_projection_objective": candidate["post_projection_objective"],
            "post_projection_geometry": {key: value.tolist() for key, value in candidate["geometry"].items()},
        })
    return {
        "geometry": final,
        "config": asdict(config),
        "optimization_pair_count": int(len(pair_indices)),
        "optimization_depth_ray_count": int(len(source_indices)),
        "regularizer_version": regularizer_version(config),
        "reference_max_axis_radius_m": reference_max_axis_radius,
        "initial_geometry": initial,
        "diagnostics": diagnostics,
        "candidates": candidate_summary,
    }
