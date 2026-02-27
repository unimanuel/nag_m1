from __future__ import annotations

"""View-dependent spherical (360°) background.

This class mirrors :class:`ViewDependentBackgroundImagePlaneSceneNode3D` but
replaces the plane UV mapping with an equirectangular sphere mapping.

The view-dependence network in NAG expects two incline-angle components. For a
plane this is computed w.r.t. the constant normal (0,0,1). For a sphere, the
normal varies per query point, so we:

1) Convert rays into the sphere's local coordinate system.
2) Build a local tangent frame at the hit point (t1, t2, n).
3) Express the ray direction in that tangent frame, where the normal is the +z
   axis.
4) Re-use the original incline-angle code with normal=(0,0,1).
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.util.typing import VEC_TYPE, DEFAULT

from nag.model.view_dependent_background_image_plane_scene_node_3d import (
    ViewDependentBackgroundImagePlaneSceneNode3D,
)
from nag.model.background_image_sphere_scene_node_3d import BackgroundImageSphereSceneNode3D

from nag.model.sphere_utils import direction_to_equirect_uv, to_tangent_frame, normalize


class ViewDependentBackgroundImageSphereSceneNode3D(
    ViewDependentBackgroundImagePlaneSceneNode3D
):
    """View-dependent spherical background."""

    is_spherical_background: bool = True

    def __init__(
        self,
        num_flow_control_points: int,
        sphere_radius: Union[float, torch.Tensor] = 1.0,
        sphere_uv_y_flip: bool = False,
        **kwargs,
    ):
        super().__init__(num_flow_control_points=num_flow_control_points, **kwargs)
        self.register_buffer(
            "sphere_radius",
            tensorify(sphere_radius, dtype=self.dtype, device=self._translation.device),
        )
        self.sphere_uv_y_flip = bool(sphere_uv_y_flip)

        # Enable periodicity in longitude and clamping in latitude.
        self.wrap_u = True
        self.clamp_v = True

    def get_sphere_radius(self) -> torch.Tensor:
        return self.sphere_radius

    def _global_points_to_sphere_uv(self, points_global: torch.Tensor, global_position: torch.Tensor) -> torch.Tensor:
        pts_local = global_to_local(global_position, points_global, v_include_time=True)[..., :3]
        return direction_to_equirect_uv(pts_local, y_flip=self.sphere_uv_y_flip)

    def _to_local_tangent_rays(
        self,
        ray_origins_global: torch.Tensor,
        ray_directions_global: torch.Tensor,
        hit_points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        """Return ray directions expressed in the hit-point tangent frame."""
        # Convert ray directions to sphere-local coordinates (about the sphere center).
        ray_targets_global = ray_origins_global + ray_directions_global
        ray_o_local = global_to_local(global_position, ray_origins_global, v_include_time=True)[..., :3]
        ray_t_local = global_to_local(global_position, ray_targets_global, v_include_time=True)[..., :3]
        ray_d_local = ray_t_local - ray_o_local

        # Normal at hit point in local coords.
        hit_local = global_to_local(global_position, hit_points_global, v_include_time=True)[..., :3]
        n = normalize(hit_local)

        # Express direction in tangent coordinates where n is +z.
        ray_d_tan = to_tangent_frame(ray_d_local, n)
        return ray_d_tan

    def forward(
        self,
        uv: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
        global_position: Optional[torch.Tensor] = None,
        uv_in_plane_space: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if uv_in_plane_space:
            uv01 = uv[..., :2]
            hit_points = None
        else:
            if global_position is None:
                global_position = self.get_global_position(t)
            uv01 = self._global_points_to_sphere_uv(uv, global_position)
            hit_points = uv

        uv_net = uv01 - 0.5

        if self.ray_dependent and (ray_origins is not None) and (ray_directions is not None) and (hit_points is not None):
            ray_d_tan = self._to_local_tangent_rays(
                ray_origins_global=ray_origins,
                ray_directions_global=ray_directions,
                hit_points_global=hit_points,
                global_position=global_position,
            )
        else:
            ray_d_tan = ray_directions

        return self.compute_rgb_alpha(
            uv=uv_net,
            t=t,
            sin_epoch=sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
            ray_origins=None,
            ray_directions=ray_d_tan,
            context=context,
            **kwargs,
        )

    def forward_modality(
        self,
        uv_plane: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        uv_net = uv_plane[..., :2] - 0.5
        return self.compute_rgb_alpha(
            uv=uv_net,
            t=t,
            sin_epoch=sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            context=context,
            **kwargs,
        )

    # Reuse the robust spherical initialization.
    @classmethod
    def for_camera(
        cls,
        camera: TimedCameraSceneNode3D,
        images: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        times: torch.Tensor,
        dataset: Any,
        scene_cutoff_distance: float = 1.5,
        dtype: torch.dtype = torch.float32,
        relative_scale_margin: float = 0.1,
        world: Optional[AbstractSceneNode] = None,
        name: Optional[str] = None,
        index: Optional[int] = None,
        nag_model: Optional[Any] = None,
        color_mask_resolution: Optional[VEC_TYPE] = None,
        color_mask_smoothing: bool = True,
        num_flow_control_points: Optional[int] = None,
        config: Any = None,
        correct_lens_distortion: bool = False,
        proxy_init: bool = False,
        sphere_uv_y_flip: bool = False,
        **kwargs,
    ):
        # Delegate most init to BackgroundImageSphereSceneNode3D.for_camera but instantiate cls.
        return BackgroundImageSphereSceneNode3D.for_camera.__func__(
            cls,
            camera=camera,
            images=images,
            masks=masks,
            depths=depths,
            times=times,
            dataset=dataset,
            scene_cutoff_distance=scene_cutoff_distance,
            dtype=dtype,
            relative_scale_margin=relative_scale_margin,
            world=world,
            name=name,
            index=index,
            nag_model=nag_model,
            color_mask_resolution=color_mask_resolution,
            color_mask_smoothing=color_mask_smoothing,
            num_flow_control_points=num_flow_control_points,
            config=config,
            correct_lens_distortion=correct_lens_distortion,
            proxy_init=proxy_init,
            sphere_uv_y_flip=sphere_uv_y_flip,
            **kwargs,
        )
