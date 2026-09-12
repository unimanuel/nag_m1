"""CPU-only invariants for the constrained reprojection/depth fit."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nag"))
from diagnostics.parallax_core import intersect, unit
from reprojection_ellipsoid import (
    ReprojectionFitConfig,
    enforce_safety,
    evaluate_geometry,
    fit_reprojection_depth_ellipsoid,
    regularizer_version,
)


def synthetic_pairs():
    """Points exactly on one ellipsoid, seen under a small camera baseline."""

    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [-1.0, 0.0, 1.0]
    geometry = {"center": np.array([0.0, 0.0, 10.0]), "radii": np.array([20.0, 10.0, 16.0]), "rotation": np.eye(3)}
    rng = np.random.default_rng(12)
    directions = rng.normal(scale=[0.22, 0.16, 0.0], size=(180, 3))
    directions[:, 2] = 1.0
    directions = unit(directions)
    points, _, valid = intersect(np.repeat(poses[0, :3, 3][None], len(directions), axis=0), directions, geometry)
    assert valid.all()
    point_rows = np.concatenate((points, points), axis=0)
    source = np.concatenate((np.zeros(len(points), dtype=np.int32), np.ones(len(points), dtype=np.int32)))
    target = np.concatenate((np.ones(len(points), dtype=np.int32), np.zeros(len(points), dtype=np.int32)))
    return {
        "points": point_rows,
        "source": source,
        "target": target,
        "visibility": np.ones(len(point_rows), dtype=np.uint8),
        "poses": poses,
        "intrinsics": np.array([[900.0, 0.0, 480.0], [0.0, 900.0, 320.0], [0.0, 0.0, 1.0]]),
        "source_points": points,
        "source_ids": np.zeros(len(points), dtype=np.int32),
    }, geometry


class ReprojectionEllipsoidTests(unittest.TestCase):
    def test_safety_projection_enforces_all_hard_limits(self):
        pairs, _ = synthetic_pairs()
        config = ReprojectionFitConfig(
            max_camera_normalized_radius=0.65,
            minimum_camera_surface_distance=6.0,
            minimum_axis_radius=8.0,
            max_axis_ratio=4.0,
        )
        unsafe = {"center": np.array([0.0, 0.0, 10.0]), "radii": np.array([30.0, 3.0, 2.0]), "rotation": np.eye(3)}
        repaired = enforce_safety(unsafe, pairs["poses"], config)
        normalized = np.linalg.norm(
            ((pairs["poses"][:, :3, 3] - repaired["center"]) @ repaired["rotation"]) / repaired["radii"],
            axis=1,
        ).max()
        clearance = repaired["radii"].min() * (1.0 - normalized)
        self.assertLessEqual(normalized, config.max_camera_normalized_radius + 1e-12)
        self.assertGreaterEqual(clearance, config.minimum_camera_surface_distance - 1e-12)
        self.assertGreaterEqual(repaired["radii"].min(), config.minimum_axis_radius)
        self.assertLessEqual(repaired["radii"].max() / repaired["radii"].min(), config.max_axis_ratio + 1e-12)

    def test_combined_fit_reduces_reprojection_without_violating_safety(self):
        pairs, truth = synthetic_pairs()
        config = ReprojectionFitConfig(
            max_pairs=360,
            max_depth_rays=180,
            max_camera_normalized_radius=0.70,
            minimum_camera_surface_distance=3.0,
            minimum_axis_radius=5.0,
            max_axis_ratio=6.0,
            max_iterations=80,
            seed=9,
        )
        initial = {**truth, "radii": np.array([28.0, 7.0, 7.0])}
        fit = fit_reprojection_depth_ellipsoid(pairs, initial, config)
        before = fit["diagnostics"]["initial_safety_projected"]
        after = fit["diagnostics"]["final"]
        self.assertLess(after["pixel_error"]["median"], before["pixel_error"]["median"])
        self.assertEqual(after["pixel_error"]["invalid_count"], 0)
        self.assertLessEqual(after["max_camera_normalized_radius"], config.max_camera_normalized_radius + 1e-12)
        self.assertGreaterEqual(after["conservative_camera_clearance_m"], config.minimum_camera_surface_distance - 1e-12)
        self.assertGreaterEqual(after["minimum_axis_radius_m"], config.minimum_axis_radius)
        self.assertLessEqual(after["maximum_axis_ratio"], config.max_axis_ratio + 1e-12)

    def test_radius_excess_regularizer_penalizes_only_above_reference_scale(self):
        pairs, truth = synthetic_pairs()
        config = ReprojectionFitConfig(max_pairs=360, max_depth_rays=180, radius_excess_weight=5.0)
        within = evaluate_geometry(pairs, truth, np.arange(360), np.arange(180), config, 20.0)
        expanded = evaluate_geometry(
            pairs, {**truth, "radii": np.array([40.0, 10.0, 16.0])},
            np.arange(360), np.arange(180), config, 20.0,
        )
        self.assertEqual(regularizer_version(config), "new_radius_scale")
        self.assertEqual(within["radius_excess_log"]["p95"], 0.0)
        self.assertGreater(expanded["radius_excess_log"]["p95"], 0.0)
        self.assertGreater(expanded["radius_excess_huber_loss"], 0.0)

    def test_zero_radius_excess_weight_preserves_original_label(self):
        self.assertEqual(regularizer_version(ReprojectionFitConfig()), "original")


if __name__ == "__main__":
    unittest.main()
