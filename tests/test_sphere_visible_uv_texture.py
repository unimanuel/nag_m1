import torch
import pytest

from nag.model.background_image_sphere_scene_node_3d import (
    BackgroundImageSphereSceneNode3D,
)


def test_circular_uv_interval_prefers_seam_crossing_interval():
    start, span = BackgroundImageSphereSceneNode3D._compute_circular_uv_interval(
        torch.tensor([0.98, 0.02, 0.01])
    )

    assert start == pytest.approx(0.98, abs=1e-6)
    assert span == pytest.approx(0.04, abs=1e-6)


def test_local_texture_lookup_uses_fallback_outside_tile():
    node = BackgroundImageSphereSceneNode3D.__new__(
        BackgroundImageSphereSceneNode3D
    )
    torch.nn.Module.__init__(node)
    node.register_buffer("initial_rgb", torch.ones(3, 4, 4))
    node.register_buffer(
        "visible_uv_texture_bounds",
        torch.tensor([0.0, 0.5, 0.0, 1.0]),
    )
    node.register_buffer(
        "visible_uv_texture_fallback_rgb",
        torch.tensor([0.0, 0.0, 1.0]),
    )
    node.visible_uv_texture_enabled = True
    node.align_corners = True

    values = node.get_initial_rgb(
        torch.tensor([[-0.25, 0.0], [0.25, 0.0]])
    )

    assert torch.allclose(values[0], torch.ones(3))
    assert torch.allclose(values[1], torch.tensor([0.0, 0.0, 1.0]))
