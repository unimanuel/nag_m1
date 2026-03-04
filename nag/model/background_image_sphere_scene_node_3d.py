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
import torch.nn.functional as F

from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.util.typing import DEFAULT, VEC_TYPE
from tools.logger.logging import logger

from nag.model.sphere_utils import direction_to_equirect_uv, equirect_uv_to_direction, normalize


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

    def _get_native_flow(
        self,
        uv_net: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the learned texture-space flow before camera reprojection."""
        if self.flow_weight != 0:
            if self.deprecated_flow:
                return self._compute_flow(
                    uv_net,
                    t,
                    sin_epoch,
                    right_idx_flow=right_idx_flow,
                    rel_frac_flow=rel_frac_flow,
                )
            return self.get_flow(
                uv_net,
                t,
                sin_epoch,
                right_idx_flow=right_idx_flow,
                rel_frac_flow=rel_frac_flow,
            )
        return torch.zeros_like(uv_net)

    def _apply_uv_bounds(self, uv_net: torch.Tensor) -> torch.Tensor:
        """Mirror the spherical wrap/clamp rules used during RGB sampling."""
        qx = uv_net[..., 0]
        qy = uv_net[..., 1]
        if getattr(self, "wrap_u", False):
            qx = torch.remainder(qx + 0.5, 1.0) - 0.5
        if getattr(self, "clamp_v", False):
            qy = qy.clamp(-0.5, 0.5)
        return torch.stack([qx, qy], dim=-1)

    def _sphere_uv_to_global_points(
        self,
        uv01: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        """Convert equirectangular UVs back to global points on the sphere."""
        local_dirs = equirect_uv_to_direction(uv01, y_flip=self.sphere_uv_y_flip)
        local_points = local_dirs * self.get_sphere_radius().to(dtype=uv01.dtype, device=uv01.device)
        return local_to_global(global_position, local_points, v_include_time=True)[..., :3]

    def get_flowed_global_points(
        self,
        points_global: torch.Tensor,
        flow: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        """Move global sphere hit points by texture-space flow on the sphere."""
        uv01 = self._global_points_to_sphere_uv(points_global, global_position)
        flowed_uv = self._apply_uv_bounds(uv01 - 0.5 + flow) + 0.5
        return self._sphere_uv_to_global_points(flowed_uv, global_position)

    @staticmethod
    def _project_points_to_grid(
        image_points: torch.Tensor,
        image_resolution: torch.Tensor,
        align_corners: bool,
    ) -> torch.Tensor:
        """Convert pixel coordinates to the normalized grid_sample domain."""
        height = int(image_resolution[0].item())
        width = int(image_resolution[1].item())
        if align_corners:
            grid_x = (image_points[..., 0] / max(width - 1, 1)) * 2.0 - 1.0
            grid_y = (image_points[..., 1] / max(height - 1, 1)) * 2.0 - 1.0
        else:
            grid_x = ((image_points[..., 0] + 0.5) / width) * 2.0 - 1.0
            grid_y = ((image_points[..., 1] + 0.5) / height) * 2.0 - 1.0
        return torch.stack([grid_x, grid_y], dim=-1)

    @classmethod
    def _reproject_images_to_sphere_texture(
        cls,
        camera: TimedCameraSceneNode3D,
        images: torch.Tensor,
        masks: torch.Tensor,
        times: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
        resolution: VEC_TYPE,
        dtype: torch.dtype,
        y_flip: bool = False,
        align_corners: bool = True,
    ) -> torch.Tensor:
        """Build an equirectangular initialization by sampling valid background pixels."""
        tex_h = int(resolution[0])
        tex_w = int(resolution[1])

        # Project each texel center back into the cameras so the sphere starts
        # from a stitched background texture instead of a single mean color.
        v = (torch.arange(tex_h, dtype=dtype, device=times.device) + 0.5) / tex_h
        u = (torch.arange(tex_w, dtype=dtype, device=times.device) + 0.5) / tex_w
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        sphere_uv = torch.stack([uu, vv], dim=-1)
        texel_dirs = equirect_uv_to_direction(sphere_uv, y_flip=y_flip)
        sphere_points = center.view(1, 1, 3) + texel_dirs * radius

        # Only unmasked pixels are eligible to initialize the sphere texture.
        # This excludes object regions so texels are selected from background only.
        background_mask = (masks.sum(dim=1, keepdim=True) == 0).to(images.dtype)
        if background_mask.shape[-2:] != images.shape[-2:]:
            background_mask = F.interpolate(
                background_mask,
                size=images.shape[-2:],
                mode="nearest",
            )

        # If no frame ever contributes to a texel, fall back to a dataset-level
        # background color prior instead of leaving the texel black.
        denom = background_mask.sum()
        if denom.item() < 1:
            fallback_rgb = images.mean(dim=(0, 2, 3))
            logger.warning("No background pixels found in masks; falling back to global mean color.")
        else:
            fallback_rgb = (images * background_mask).sum(dim=(0, 2, 3)) / denom
        fallback_rgb = fallback_rgb.clamp(0, 1)

        best_rgb = torch.zeros((3, tex_h, tex_w), dtype=dtype, device=times.device)
        best_weight = torch.zeros((1, tex_h, tex_w), dtype=dtype, device=times.device)
        sphere_normals = texel_dirs

        for frame_idx, frame_time in enumerate(times):
            image_points, valid = camera.global_to_image_coordinates(
                sphere_points,
                t=frame_time.unsqueeze(0),
                v_includes_time=False,
                clamp=False,
                return_ok=True,
            )
            image_points = image_points[..., 0, :]
            # Ignore projections that are outside the camera frustum or otherwise invalid.
            valid = valid[..., 0].to(dtype).unsqueeze(0)

            sample_grid = cls._project_points_to_grid(
                image_points=image_points,
                image_resolution=camera._image_resolution,
                align_corners=align_corners,
            ).unsqueeze(0)
            # Stop uniformly averaging every valid frame: pick the best view
            # per texel so the sphere init keeps sharper details instead of
            # blending across frames with different perspective/parallax.
            dist2 = sample_grid[0, ..., 0] ** 2 + sample_grid[0, ..., 1] ** 2
            # Prefer samples nearer the image center, where distortion and edge artifacts
            # are typically lower than near the border.
            center_weight = torch.exp(-dist2).unsqueeze(0)
            camera_position = camera.get_global_position(t=frame_time.unsqueeze(0))[0, :3, 3].to(dtype)
            view_dir = normalize(sphere_points - camera_position.view(1, 1, 3))
            # Prefer views that look more head-on at the sphere texel. The texel normal
            # is the sphere's radial direction at that texel center.
            incidence_weight = (view_dir * sphere_normals).sum(dim=-1).clamp_min(0.0).unsqueeze(0)

            sampled_rgb = F.grid_sample(
                images[frame_idx:frame_idx + 1].to(dtype=dtype),
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=align_corners,
            )[0]
            sampled_bg = F.grid_sample(
                background_mask[frame_idx:frame_idx + 1],
                sample_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=align_corners,
            )[0]
            # A frame can win a texel only if it lands on a background pixel, projects
            # validly, stays near the image center, and views the texel front-facing.
            weight = sampled_bg * valid * center_weight * incidence_weight
            # Keep the single highest-scoring frame for each texel instead of averaging.
            use_sample = weight > best_weight
            best_rgb = torch.where(use_sample.expand_as(best_rgb), sampled_rgb, best_rgb)
            best_weight = torch.where(use_sample, weight, best_weight)

        # Any texels unseen by all cameras keep a reasonable color prior
        # instead of starting from black or a flat zero-filled patch.
        return torch.where(
            best_weight > 0,
            best_rgb,
            fallback_rgb[:, None, None].expand(3, tex_h, tex_w),
        ).contiguous()

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
        # `global_to_local` uses the singular `v_include_time` keyword; the
        # sphere path needs the timed transform variant here.
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
        global_position: Optional[torch.Tensor] = None,
        uv_in_plane_space: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The functional modality path is shared with planar objects, so this
        # method accepts either plane-space UVs or the global 3D sphere hits.
        if uv_in_plane_space:
            uv01 = uv_plane[..., :2]
        else:
            if global_position is None:
                global_position = self.get_global_position(t)
            uv01 = self._global_points_to_sphere_uv(uv_plane, global_position)

        uv_net = uv01 - 0.5
        # Export the same learned sphere-surface flow that drives texture
        # lookup, then let the functional model reproject it into image space.
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
            context=context,
            **kwargs,
        )
        return rgb, alpha, flow

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

        The sphere center is set to the mean camera position; its radius is chosen
        large enough to cover the camera trajectory plus the requested
        ``scene_cutoff_distance``.
        """
        if world is None:
            raise ValueError("world must be provided")

        if color_mask_resolution is None or color_mask_resolution == DEFAULT:
            # Scale the sphere texture from the available image resolution so
            # the equirectangular init does not lose detail to a hard-coded
            # low-res map on larger scenes. Keep the old 512x1024 floor so the
            # smaller debug configs do not regress.
            res_x = min(camera._image_resolution[1], images.shape[3])
            res_y = min(camera._image_resolution[0], images.shape[2])
            base_height = int(round(float(res_y) * (1.0 + float(relative_scale_margin))))
            base_width = int(round(float(res_x) * (1.0 + float(relative_scale_margin))))
            sphere_height = max(512, base_height)
            sphere_width = max(1024, base_width, 2 * sphere_height)
            color_mask_resolution = [sphere_height, sphere_width]
            logger.info(
                "Color resolution on spherical background set to default. "
                f"Using resolution {sphere_height} x {sphere_width} (H x W)."
            )

        if num_flow_control_points is None:
            if config is not None and hasattr(config, "plane_flow_control_points_ratio"):
                num_flow_control_points = int(round(len(times) * config.plane_flow_control_points_ratio))
            else:
                num_flow_control_points = 1

        # Sphere center: mean camera position
        cam_positions = camera.get_global_position(t=times)[:, :3, 3].detach()  # (T, 3)
        center = cam_positions.mean(dim=0)
        max_cam_dist = (cam_positions - center[None, :]).norm(dim=-1).max()
        # Conservative radius: trajectory radius + cutoff distance (+ margin)
        radius = (max_cam_dist + float(scene_cutoff_distance)) * (1.0 + float(relative_scale_margin))

        if proxy_init:
            init_rgb = torch.zeros((3, int(color_mask_resolution[0]), int(color_mask_resolution[1])), dtype=dtype, device=times.device)
        else:
            with torch.no_grad():
                # Unlike the plane background, the sphere has no natural
                # reference view, so seed it by averaging reprojected camera
                # colors into an equirectangular texture instead of one flat mean.
                init_rgb = BackgroundImageSphereSceneNode3D._reproject_images_to_sphere_texture(
                    camera=camera,
                    images=images,
                    masks=masks,
                    times=times,
                    center=center.to(dtype),
                    radius=radius.to(dtype),
                    resolution=color_mask_resolution,
                    dtype=dtype,
                    y_flip=sphere_uv_y_flip,
                    align_corners=getattr(config, "plane_align_corners", True) if config is not None else True,
                )

        # Provide dummy plane scale/offset (unused for spherical UV), but required by base classes.
        plane_scale = torch.tensor([1.0, 1.0], dtype=dtype, device=times.device)
        plane_scale_offset = torch.tensor([-0.5, -0.5], dtype=dtype, device=times.device)

        translation_t = center[None, :].expand(len(camera._times), -1).to(dtype)
        orientation_t = torch.zeros((len(camera._times), 4), dtype=dtype, device=times.device)
        orientation_t[..., 3] = 1.0

        # Keep compatibility with the planar background `for_camera` API so the
        # runner can pass the same kwargs to plane and sphere backgrounds.
        # The spherical init stitches directly through the camera model, so the
        # plane-specific smoothing/distortion knobs do not map one-to-one here.
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
