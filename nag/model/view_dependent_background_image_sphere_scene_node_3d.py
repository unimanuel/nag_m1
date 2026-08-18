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
import torch.nn.functional as F

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
    is_curved_background: bool = True

    def __init__(
        self,
        num_flow_control_points: int,
        sphere_radius: Union[float, torch.Tensor] = 1.0,
        sphere_uv_y_flip: bool = False,
        visible_uv_texture_bounds: Optional[VEC_TYPE] = None,
        visible_uv_texture_fallback_rgb: Optional[VEC_TYPE] = None,
        visible_uv_texture_metadata: Optional[Dict[str, Any]] = None,
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

        # The view-dependent sphere inherits its RGB/network implementation
        # from the planar view-dependent node, so it cannot inherit the
        # visible-UV state initialization from BackgroundImageSphereSceneNode3D.
        # Keep the same state and validation here so the shared for_camera()
        # constructor works for both spherical node variants.
        if visible_uv_texture_bounds is not None:
            visible_bounds = tensorify(
                visible_uv_texture_bounds,
                dtype=self.dtype,
                device=self._translation.device,
            ).reshape(-1)
            if visible_bounds.numel() != 4:
                raise ValueError(
                    "visible_uv_texture_bounds must contain "
                    "[u_start, u_span, v_start, v_span]."
                )
            if not torch.isfinite(visible_bounds).all():
                raise ValueError("visible_uv_texture_bounds must be finite.")
            if visible_bounds[1] <= 0 or visible_bounds[1] > 1:
                raise ValueError("visible UV u_span must be in (0, 1].")
            if visible_bounds[3] <= 0 or visible_bounds[3] > 1:
                raise ValueError("visible UV v_span must be in (0, 1].")
        else:
            visible_bounds = torch.zeros(
                4, dtype=self.dtype, device=self._translation.device
            )

        self.register_buffer("visible_uv_texture_bounds", visible_bounds)
        self.visible_uv_texture_enabled = visible_uv_texture_bounds is not None

        if visible_uv_texture_fallback_rgb is None:
            fallback_rgb = torch.zeros(
                3, dtype=self.dtype, device=self._translation.device
            )
        else:
            fallback_rgb = tensorify(
                visible_uv_texture_fallback_rgb,
                dtype=self.dtype,
                device=self._translation.device,
            ).reshape(-1)
            if fallback_rgb.numel() != 3:
                raise ValueError("visible_uv_texture_fallback_rgb must contain RGB values.")
        self.register_buffer("visible_uv_texture_fallback_rgb", fallback_rgb.clamp(0, 1))
        self.visible_uv_texture_metadata = dict(visible_uv_texture_metadata or {})

    def get_sphere_radius(self) -> torch.Tensor:
        return self.sphere_radius

    def intersect_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        global_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return BackgroundImageSphereSceneNode3D.intersect_rays(
            self,
            ray_origins,
            ray_directions,
            global_position,
        )

    def get_global_surface_normals(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageSphereSceneNode3D.get_global_surface_normals(
            self,
            points_global,
            global_position,
        )

    def get_visible_uv_texture_metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable diagnostics for the optional local texture."""
        metadata = dict(self.visible_uv_texture_metadata)
        metadata["enabled"] = bool(self.visible_uv_texture_enabled)
        metadata["fallback_rgb"] = [
            float(value)
            for value in self.visible_uv_texture_fallback_rgb.detach().cpu().tolist()
        ]
        if self.visible_uv_texture_enabled:
            bounds = self.visible_uv_texture_bounds.detach().cpu().tolist()
            metadata["u_start"] = float(bounds[0])
            metadata["u_span"] = float(bounds[1])
            metadata["v_start"] = float(bounds[2])
            metadata["v_span"] = float(bounds[3])
        return metadata

    def get_initial_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        """Sample the visible local tile, with a prior outside its bounds."""
        if not self.visible_uv_texture_enabled:
            return super().get_initial_rgb(uv)

        inside, local_u, local_v = self._visible_uv_texture_coordinates(uv)

        local_grid = torch.stack([local_u, local_v], dim=-1)
        grid = (local_grid * 2.0 - 1.0)[None, None, ...]
        local_rgb = F.grid_sample(
            self.initial_rgb.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )[0].reshape(3, -1).permute(1, 0).reshape(*uv.shape[:-1], 3)
        fallback_rgb = self.visible_uv_texture_fallback_rgb.to(
            dtype=local_rgb.dtype, device=local_rgb.device
        )
        return torch.where(inside[..., None], local_rgb, fallback_rgb)

    def _visible_uv_texture_coordinates(
        self, uv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return BackgroundImageSphereSceneNode3D._visible_uv_texture_coordinates(
            self, uv
        )

    def _record_visible_uv_texture_lookups(
        self, uv: torch.Tensor, context: Optional[Dict[str, Any]]
    ) -> None:
        return BackgroundImageSphereSceneNode3D._record_visible_uv_texture_lookups(
            self, uv, context
        )

    @staticmethod
    def _compute_background_fallback_rgb(
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageSphereSceneNode3D._compute_background_fallback_rgb(
            images, masks
        )

    @staticmethod
    def _compute_circular_uv_interval(u_values: torch.Tensor) -> Tuple[float, float]:
        return BackgroundImageSphereSceneNode3D._compute_circular_uv_interval(u_values)

    @classmethod
    def _compute_visible_uv_texture_layout(cls, *args, **kwargs):
        return BackgroundImageSphereSceneNode3D._compute_visible_uv_texture_layout.__func__(
            cls, *args, **kwargs
        )

    def _get_native_flow(
        self,
        uv_net: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Share the same sphere texture-flow definition as the non-view-dependent
        # background even though this class inherits from the view-dependent
        # planar base for its RGB path.
        return BackgroundImageSphereSceneNode3D._get_native_flow(
            self,
            uv_net,
            t,
            sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
        )

    def _apply_uv_bounds(self, uv_net: torch.Tensor) -> torch.Tensor:
        return BackgroundImageSphereSceneNode3D._apply_uv_bounds(self, uv_net)

    def _sphere_uv_to_global_points(
        self,
        uv01: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageSphereSceneNode3D._sphere_uv_to_global_points(
            self,
            uv01,
            global_position,
        )

    def get_flowed_global_points(
        self,
        points_global: torch.Tensor,
        flow: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageSphereSceneNode3D.get_flowed_global_points(
            self,
            points_global,
            flow,
            global_position,
        )

    def _global_points_to_sphere_uv(self, points_global: torch.Tensor, global_position: torch.Tensor) -> torch.Tensor:
        # `global_to_local` uses the singular `v_include_time` keyword; the
        # sphere path needs the timed transform variant here.
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
        # Use the timed transform path because the sphere center can vary over time.
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
        global_position: Optional[torch.Tensor] = None,
        uv_in_plane_space: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if uv_in_plane_space:
            uv01 = uv_plane[..., :2]
            ray_d_tan = ray_directions
        else:
            if global_position is None:
                global_position = self.get_global_position(t)
            uv01 = self._global_points_to_sphere_uv(uv_plane, global_position)
            # Object-modality rendering hands us global rays for spherical
            # backgrounds so we can reuse the same tangent-frame conversion as
            # the main forward path.
            if self.ray_dependent and ray_origins is not None and ray_directions is not None:
                ray_d_tan = self._to_local_tangent_rays(
                    ray_origins_global=ray_origins,
                    ray_directions_global=ray_directions,
                    hit_points_global=uv_plane,
                    global_position=global_position,
                )
            else:
                ray_d_tan = ray_directions

        uv_net = uv01 - 0.5
        # Keep modality flow in sphere texture coordinates so the caller can
        # convert it into true image-space motion with the camera model.
        flow = self._get_native_flow(
            uv_net,
            t,
            sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
        )
        rgb, alpha = self.compute_rgb_alpha(
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
        return rgb, alpha, flow

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
        sphere_radius_override: Optional[float] = None,
        sphere_radius_override_source: Optional[str] = None,
        **kwargs,
    ):
        # Reuse the shared spherical init logic so view-dependent and plain
        # spherical backgrounds stay argument-compatible with the runner.
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
            sphere_radius_override=sphere_radius_override,
            sphere_radius_override_source=sphere_radius_override_source,
            **kwargs,
        )
