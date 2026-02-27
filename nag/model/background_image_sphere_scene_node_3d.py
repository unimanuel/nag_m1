from __future__ import annotations

"""Spherical (360°) background node.

This is a drop-in replacement for :class:`BackgroundImagePlaneSceneNode3D` that
parameterizes the background on a sphere (equirectangular UV) instead of an
infinite plane.

Key idea
--------
The renderer still provides *intersection points* per object. For this node we
expect those points to lie on a sphere surface. We convert them to local sphere
coordinates, compute equirectangular UV, and then reuse the existing image/flow
networks.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.util.typing import DEFAULT, VEC_TYPE
from tools.logger.logging import logger

from nag.model.sphere_utils import direction_to_equirect_uv


class BackgroundImageSphereSceneNode3D(BackgroundImagePlaneSceneNode3D):
    """Background represented by an equirectangular texture on a sphere."""

    # Used by the models to detect that this background should be intersected as a sphere.
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

        # Tell the inherited compute_rgb_alpha() to apply periodic wrap in U
        # (longitude) and clamping in V (latitude).
        self.wrap_u = True
        self.clamp_v = True

    def get_sphere_radius(self) -> torch.Tensor:
        return self.sphere_radius

    # ------------------------------------------------------------------
    # Mapping: global intersection point -> spherical UV
    # ------------------------------------------------------------------
    def _global_points_to_sphere_uv(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        """Convert global points on the sphere into equirectangular UV in [0, 1].

        Parameters
        ----------
        points_global : torch.Tensor
            (B, T, 3) global intersection points (assumed on the sphere).
        global_position : torch.Tensor
            (T, 4, 4) transformation matrix of this node.
        """
        # Local points in sphere coordinates (origin at sphere center)
        pts_local = global_to_local(global_position, points_global, v_include_time=True)[..., :3]
        uv01 = direction_to_equirect_uv(pts_local, y_flip=self.sphere_uv_y_flip)
        return uv01

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        uv: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
        global_position: Optional[torch.Tensor] = None,
        uv_in_plane_space: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render background for the given intersection points.

        For compatibility with the plane implementation, the parameter name
        ``uv`` still refers to the *3D query points* when ``uv_in_plane_space``
        is False.
        """
        if uv_in_plane_space:
            # Here, uv is expected to be (B, T, 2) already in [0, 1].
            uv01 = uv[..., :2]
        else:
            if global_position is None:
                global_position = self.get_global_position(t)
            uv01 = self._global_points_to_sphere_uv(uv, global_position)

        # Convert to the [-0.5, 0.5] convention used throughout NAG.
        uv_net = uv01 - 0.5
        return self.compute_rgb_alpha(
            uv=uv_net,
            t=t,
            sin_epoch=sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
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
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # uv_plane expected in [0, 1]
        uv_net = uv_plane[..., :2] - 0.5
        return self.compute_rgb_alpha(
            uv=uv_net,
            t=t,
            sin_epoch=sin_epoch,
            right_idx_flow=right_idx_flow,
            rel_frac_flow=rel_frac_flow,
            context=context,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------
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
        """Initialize a spherical background.

        We keep initialization deliberately simple and robust: we compute the mean
        observed background color (mask==0) and fill the spherical texture with it.

        The sphere center is set to the mean camera position; its radius is chosen
        large enough to cover the camera trajectory plus the requested
        ``scene_cutoff_distance``.
        """
        if world is None:
            raise ValueError("world must be provided")

        if color_mask_resolution is None or color_mask_resolution == DEFAULT:
            # Equirectangular: width ~ 2*height
            color_mask_resolution = [512, 1024]

        if num_flow_control_points is None:
            if config is not None and hasattr(config, "plane_flow_control_points_ratio"):
                num_flow_control_points = int(round(len(times) * config.plane_flow_control_points_ratio))
            else:
                num_flow_control_points = 1

        # Compute a constant initial RGB from the visible background pixels.
        if proxy_init:
            init_rgb = torch.zeros((3, int(color_mask_resolution[0]), int(color_mask_resolution[1])), dtype=dtype, device=times.device)
        else:
            with torch.no_grad():
                # background pixels where no object mask is present
                bg_mask = (masks.sum(dim=1) == 0).to(images.dtype)  # (T, H, W)
                # Avoid shape mismatch if masks resolution differs from images
                if bg_mask.shape[-2:] != images.shape[-2:]:
                    # Nearest down/up sample to images resolution
                    bg_mask = torch.nn.functional.interpolate(
                        bg_mask.unsqueeze(1), size=images.shape[-2:], mode="nearest"
                    ).squeeze(1)
                denom = bg_mask.sum() * 1.0
                if denom.item() < 1:
                    mean_rgb = images.mean(dim=(0, 2, 3))
                    logger.warning("No background pixels found in masks; falling back to global mean color.")
                else:
                    mean_rgb = (images * bg_mask[:, None, :, :]).sum(dim=(0, 2, 3)) / denom
                mean_rgb = mean_rgb.clamp(0, 1)
                init_rgb = mean_rgb[:, None, None].expand(
                    3, int(color_mask_resolution[0]), int(color_mask_resolution[1])
                ).contiguous()

        # Sphere center: mean camera position
        cam_positions = camera.get_global_position(t=times)[:, :3, 3].detach()  # (T, 3)
        center = cam_positions.mean(dim=0)
        max_cam_dist = (cam_positions - center[None, :]).norm(dim=-1).max()
        # Conservative radius: trajectory radius + cutoff distance (+ margin)
        radius = (max_cam_dist + float(scene_cutoff_distance)) * (1.0 + float(relative_scale_margin))

        # Provide dummy plane scale/offset (unused for spherical UV), but required by base classes.
        plane_scale = torch.tensor([1.0, 1.0], dtype=dtype, device=times.device)
        plane_scale_offset = torch.tensor([-0.5, -0.5], dtype=dtype, device=times.device)

        translation_t = center[None, :].expand(len(camera._times), -1).to(dtype)
        orientation_t = torch.zeros((len(camera._times), 4), dtype=dtype, device=times.device)
        orientation_t[..., 3] = 1.0

        # Keep compatibility with the planar background `for_camera` API.
        # These reprojection-specific options are meaningful for plane init,
        # but the spherical init path does not use them.
        del color_mask_smoothing
        del correct_lens_distortion

        args = dict(kwargs)
        args.update(
            is_camera_attached=False,
            plane_scale=plane_scale,
            plane_scale_offset=plane_scale_offset,
            translation=translation_t,
            times=camera._times,
            orientation=orientation_t,
            initial_rgb=init_rgb,
            flow_reference_time=0,
            num_flow_control_points=num_flow_control_points,
            dtype=dtype,
            name=name,
            index=index,
            sphere_radius=radius,
            sphere_uv_y_flip=sphere_uv_y_flip,
            network_dtype=getattr(config, "tinycudann_network_dtype", torch.float16) if config is not None else torch.float16,
        )

        if nag_model is not None:
            from nag.model.nag_functional_model import NAGFunctionalModel
            if isinstance(nag_model, NAGFunctionalModel):
                args = nag_model.patch_background_args(args)

        sphere = cls(**args)
        world.add_scene_children(sphere)
        return sphere
