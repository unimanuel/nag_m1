import json
from types import SimpleNamespace

import pytest
import torch

from nag.model.background_image_ellipsoid_scene_node_3d import (
    BackgroundImageEllipsoidSceneNode3D,
    load_ellipsoid_geometry,
    validate_ellipsoid_matches_config,
)
from nag.model.sphere_utils import ray_ellipsoid_intersection
from nag.model.view_dependent_background_image_ellipsoid_scene_node_3d import (
    ViewDependentBackgroundImageEllipsoidSceneNode3D,
)


def _rotation_z(angle_degrees):
    angle = torch.deg2rad(torch.tensor(float(angle_degrees)))
    return torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _global_position(center, rotation):
    position = torch.eye(4).repeat(1, 1, 1)
    position[0, :3, :3] = rotation
    position[0, :3, 3] = torch.as_tensor(center)
    return position


def _write_geometry(path, contains_all_points=True, format_version=1, point_coverage=None):
    fit = {
        "center": [1.0, 2.0, 3.0],
        "radii": [2.0, 3.0, 4.0],
        "rotation_columns": torch.eye(3).tolist(),
        "contains_all_points": contains_all_points,
    }
    if point_coverage is not None:
        fit["point_coverage_fraction"] = point_coverage
    path.write_text(
        json.dumps(
            {
                "format_version": format_version,
                "sequence": "segment-123",
                "frame_slice": {
                    "start": 2,
                    "stop": 7,
                    "step": 1,
                    "first": 2,
                    "last": 6,
                    "count": 5,
                },
                "point_count": 42,
                "fit": fit,
            }
        ),
        encoding="utf-8",
    )


def test_oriented_ellipsoid_intersection_uses_local_axes():
    rotation = _rotation_z(90.0)
    local_x_in_world = rotation[:, 0]
    origins = torch.tensor([[[10.0, -4.0, 2.0]]])
    directions = local_x_in_world.reshape(1, 1, 3)

    points, hit, distance = ray_ellipsoid_intersection(
        ray_origins=origins,
        ray_directions=directions,
        center=torch.tensor([[10.0, -4.0, 2.0]]),
        radii=torch.tensor([2.0, 3.0, 4.0]),
        rotation=rotation.unsqueeze(0),
    )

    assert hit.item()
    assert distance.item() == pytest.approx(2.0, abs=1e-6)
    assert torch.allclose(points[0, 0], origins[0, 0] + 2.0 * local_x_in_world)


def test_ellipsoid_intersection_chooses_near_root_for_outside_ray_and_rejects_miss():
    origins = torch.tensor([[[5.0, 0.0, 0.0]], [[5.0, 5.0, 0.0]]])
    directions = torch.tensor([[[-1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    points, hit, distance = ray_ellipsoid_intersection(
        origins,
        directions,
        center=torch.zeros(1, 3),
        radii=torch.tensor([2.0, 3.0, 4.0]),
        rotation=torch.eye(3).unsqueeze(0),
    )

    assert hit[:, 0].tolist() == [True, False]
    assert distance[0, 0].item() == pytest.approx(3.0, abs=1e-6)
    assert torch.allclose(points[0, 0], torch.tensor([2.0, 0.0, 0.0]))


def test_geometry_loader_applies_margin_and_validates_yaml_identity(tmp_path):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(path)
    geometry = load_ellipsoid_geometry(path, scale_margin=0.05)

    assert torch.allclose(geometry.radii, torch.tensor([2.1, 3.15, 4.2]))
    assert geometry.source_radii.tolist() == [2.0, 3.0, 4.0]
    assert geometry.point_count == 42
    config = SimpleNamespace(
        frame_indices_filter=slice(2, 7, 1),
        data_path="/datasets/waymo/segment-123",
    )
    validate_ellipsoid_matches_config(geometry, config)

    config.frame_indices_filter = slice(3, 7, 1)
    with pytest.raises(ValueError, match="does not match"):
        validate_ellipsoid_matches_config(geometry, config)


def test_geometry_loader_final_radii_override_bypasses_margin(tmp_path):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(path)

    geometry = load_ellipsoid_geometry(
        path,
        scale_margin=0.25,
        radii_override=[8.0, 6.0, 5.0],
    )

    assert geometry.source_radii.tolist() == [2.0, 3.0, 4.0]
    assert geometry.radii_override.tolist() == [8.0, 6.0, 5.0]
    assert geometry.radii.tolist() == [8.0, 6.0, 5.0]


@pytest.mark.parametrize(
    "override, message",
    [
        ([1.0, 2.0], "exactly three"),
        ([1.0, 0.0, 3.0], "finite positive"),
        ([1.0, float("inf"), 3.0], "finite positive"),
    ],
)
def test_geometry_loader_rejects_invalid_final_radii_override(tmp_path, override, message):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(path)

    with pytest.raises(ValueError, match=message):
        load_ellipsoid_geometry(path, scale_margin=0.02, radii_override=override)


def test_geometry_loader_rejects_non_enclosing_source_fit(tmp_path):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(path, contains_all_points=False)
    with pytest.raises(ValueError, match="does not contain all"):
        load_ellipsoid_geometry(path, scale_margin=0.02)


def test_geometry_loader_accepts_explicit_robust_coverage(tmp_path):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(
        path,
        contains_all_points=False,
        format_version=2,
        point_coverage=0.995,
    )

    geometry = load_ellipsoid_geometry(path, scale_margin=0.02)

    assert not geometry.source_contains_all_points
    assert geometry.source_point_coverage_fraction == pytest.approx(0.995)


def test_geometry_loader_rejects_insufficient_robust_coverage(tmp_path):
    path = tmp_path / "ellipsoid.json"
    _write_geometry(
        path,
        contains_all_points=False,
        format_version=2,
        point_coverage=0.98,
    )
    with pytest.raises(ValueError, match="at least 0.99"):
        load_ellipsoid_geometry(path, scale_margin=0.02)


@pytest.mark.parametrize(
    "node_class",
    [
        BackgroundImageEllipsoidSceneNode3D,
        ViewDependentBackgroundImageEllipsoidSceneNode3D,
    ],
)
def test_ellipsoid_uv_round_trip_and_surface_normals(node_class):
    node = node_class.__new__(node_class)
    torch.nn.Module.__init__(node)
    node.register_buffer("ellipsoid_radii", torch.tensor([2.0, 3.0, 4.0]))
    node.sphere_uv_y_flip = False
    rotation = _rotation_z(31.0)
    position = _global_position([5.0, -2.0, 7.0], rotation)
    uv = torch.tensor([[[0.17, 0.24]], [[0.71, 0.68]]])

    points = node._sphere_uv_to_global_points(uv, position)
    recovered_uv = node._global_points_to_sphere_uv(points, position)
    normals = node.get_global_surface_normals(points, position)

    assert torch.allclose(recovered_uv, uv, atol=1e-6)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(2, 1), atol=1e-6)
    local_points = torch.matmul(
        (points - position[0, :3, 3]).unsqueeze(-2), rotation
    ).squeeze(-2)
    normalized_radius = (local_points / node.ellipsoid_radii).norm(dim=-1)
    assert torch.allclose(normalized_radius, torch.ones_like(normalized_radius), atol=1e-6)


def test_ellipsoid_background_declares_curved_not_spherical_protocol():
    assert BackgroundImageEllipsoidSceneNode3D.is_curved_background
    assert BackgroundImageEllipsoidSceneNode3D.is_ellipsoidal_background
    assert not BackgroundImageEllipsoidSceneNode3D.is_spherical_background
    assert ViewDependentBackgroundImageEllipsoidSceneNode3D.is_curved_background


@pytest.mark.parametrize(
    "base_class",
    [
        BackgroundImageEllipsoidSceneNode3D,
        ViewDependentBackgroundImageEllipsoidSceneNode3D,
    ],
)
@pytest.mark.parametrize(
    "visible_uv_argument",
    ["ellipsoid_visible_uv_texture", "sphere_visible_uv_texture"],
)
def test_for_camera_loads_geometry_checks_cameras_and_constructs_oriented_node(
    tmp_path, base_class, visible_uv_argument
):
    geometry_path = tmp_path / "ellipsoid.json"
    _write_geometry(geometry_path)
    times = torch.linspace(0.0, 1.0, 5)

    class CapturingNode(base_class):
        def __init__(self, **kwargs):
            self.constructor_args = kwargs

    class Camera:
        _image_resolution = torch.tensor([4, 6])
        _times = times

        def get_global_position(self, t):
            positions = torch.eye(4).repeat(len(t), 1, 1)
            positions[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
            return positions

        def global_to_image_coordinates(
            self, points, t, v_includes_time, clamp, return_ok
        ):
            assert not v_includes_time
            image_points = torch.empty((*points.shape[:-1], 1, 2))
            image_points[..., 0, 0] = 2.5
            image_points[..., 0, 1] = 1.5
            valid = torch.ones((*points.shape[:-1], 1), dtype=torch.bool)
            return image_points, valid

        def get_global_rays(self, uv, t, uv_includes_time):
            assert not uv_includes_time
            origins = torch.tensor([1.0, 2.0, 3.0]).expand(len(uv), len(t), 3)
            directions = torch.tensor([0.0, 0.0, 1.0]).expand(len(uv), len(t), 3)
            return origins, directions

    class World:
        child = None

        def add_scene_children(self, child):
            self.child = child

    config = SimpleNamespace(
        frame_indices_filter=slice(2, 7, 1),
        data_path="/datasets/waymo/segment-123",
        plane_flow_control_points_ratio=1.0,
        plane_align_corners=True,
        tinycudann_network_dtype=torch.float16,
        output_path=None,
    )
    world = World()

    visible_uv_kwargs = {visible_uv_argument: {"enabled": True}}
    node = base_class.for_camera.__func__(
        CapturingNode,
        camera=Camera(),
        images=torch.zeros(5, 3, 4, 6),
        masks=torch.zeros(5, 1, 4, 6),
        depths=torch.zeros(5, 1, 4, 6),
        times=times,
        dataset=object(),
        world=world,
        config=config,
        proxy_init=False,
        color_mask_resolution=[4, 8],
        ellipsoid_path=geometry_path,
        ellipsoid_scale_margin=0.02,
        **visible_uv_kwargs,
    )

    assert node is world.child
    assert torch.allclose(
        node.constructor_args["ellipsoid_radii"], torch.tensor([2.04, 3.06, 4.08])
    )
    assert torch.allclose(
        node.constructor_args["translation"], torch.tensor([[1.0, 2.0, 3.0]]).expand(5, -1)
    )
    assert torch.allclose(
        node.constructor_args["orientation"], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(5, -1)
    )
    assert node.constructor_args["initial_rgb"].shape == (3, 4, 8)
    assert node.constructor_args["visible_uv_texture_metadata"]["enabled"]
