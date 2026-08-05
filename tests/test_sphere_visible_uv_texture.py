import torch
import pytest

from nag.model.background_image_sphere_scene_node_3d import (
    BackgroundImageSphereSceneNode3D,
)
from nag.model.view_dependent_background_image_sphere_scene_node_3d import (
    ViewDependentBackgroundImageSphereSceneNode3D,
)


def test_circular_uv_interval_prefers_seam_crossing_interval():
    start, span = BackgroundImageSphereSceneNode3D._compute_circular_uv_interval(
        torch.tensor([0.98, 0.02, 0.01])
    )

    assert start == pytest.approx(0.98, abs=1e-6)
    assert span == pytest.approx(0.04, abs=1e-6)


@pytest.mark.parametrize(
    "node_class",
    [
        BackgroundImageSphereSceneNode3D,
        ViewDependentBackgroundImageSphereSceneNode3D,
    ],
)
def test_local_texture_lookup_uses_fallback_outside_tile(node_class):
    node = node_class.__new__(node_class)
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


@pytest.mark.parametrize(
    "node_class",
    [
        BackgroundImageSphereSceneNode3D,
        ViewDependentBackgroundImageSphereSceneNode3D,
    ],
)
def test_runtime_metric_counts_out_of_tile_lookups(node_class):
    node = node_class.__new__(node_class)
    torch.nn.Module.__init__(node)
    node.register_buffer(
        "visible_uv_texture_bounds",
        torch.tensor([0.0, 0.5, 0.0, 1.0]),
    )
    node.visible_uv_texture_enabled = True

    context = {}
    node._record_visible_uv_texture_lookups(
        torch.tensor([[[-0.25, 0.0], [0.25, 0.0]]]),
        context,
    )

    assert context["visible_uv_texture_fallback_count"].item() == 1
    assert context["visible_uv_texture_lookup_count"].item() == 2


def test_view_dependent_sphere_records_lookups_during_rgb_computation():
    """The view-dependent override must not bypass the visible-UV metric."""
    node = ViewDependentBackgroundImageSphereSceneNode3D.__new__(
        ViewDependentBackgroundImageSphereSceneNode3D
    )
    torch.nn.Module.__init__(node)
    node.flow_weight = 0.0
    node.wrap_u = True
    node.clamp_v = True
    node.visible_uv_texture_enabled = True
    node.register_buffer(
        "visible_uv_texture_bounds",
        torch.tensor([0.0, 0.5, 0.0, 1.0]),
    )
    node.dtype = torch.float32
    node._index = 0
    node.rgb_weight = 0.0
    node.view_dependence_weight = 0.0
    node.render_texture_map = False
    node.get_rgb = lambda coords, sin_epoch: torch.zeros(
        (*coords.shape[:-1], 3), dtype=coords.dtype, device=coords.device
    )
    node.get_initial_rgb = lambda coords: torch.zeros(
        (*coords.shape[:-1], 3), dtype=coords.dtype, device=coords.device
    )
    node.get_view_dependence = lambda coords, **kwargs: torch.zeros(
        (*coords.shape[:-1], 3), dtype=coords.dtype, device=coords.device
    )

    context = {}
    node.compute_rgb_alpha(
        uv=torch.tensor([[[-0.25, 0.0], [0.25, 0.0]]]),
        t=torch.tensor([0.0, 1.0]),
        sin_epoch=torch.tensor(1.0),
        ray_directions=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
        context=context,
    )

    assert context["visible_uv_texture_fallback_count"].item() == 1
    assert context["visible_uv_texture_lookup_count"].item() == 2
