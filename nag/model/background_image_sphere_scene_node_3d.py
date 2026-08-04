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

import os
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

from nag.model.sphere_utils import (
    direction_to_equirect_uv,
    equirect_uv_to_direction,
    normalize,
    ray_sphere_intersection,
)


def _write_sphere_geometry_diagnostic(config: Any, geometry: Dict[str, Any]) -> None:
    output_path = getattr(config, "output_path", None)
    if output_path is None:
        return

    diagnostic_dir = os.path.join(
        str(output_path), "initial", "background_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    diagnostic_path = os.path.join(diagnostic_dir, "sphere_geometry.txt")

    lines = [
        "Sphere Background Geometry Diagnostic",
        "",
    ]
    lines.extend([f"{key}: {value}" for key, value in geometry.items()])

    with open(diagnostic_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


class BackgroundImageSphereSceneNode3D(BackgroundImagePlaneSceneNode3D):
    """Background represented by an equirectangular texture on a sphere."""

    # Used by the models to detect that this background should be intersected as a sphere.
    is_spherical_background: bool = True

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

        # Tell the inherited compute_rgb_alpha() to apply periodic wrap in U
        # (longitude) and clamping in V (latitude).
        self.wrap_u = True
        self.clamp_v = True

        visible_bounds = None
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
            fallback_rgb = torch.zeros(3, dtype=self.dtype, device=self._translation.device)
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
        """Sample the dense local tile using global spherical UV coordinates.

        Flow and the learned RGB network continue to use global UVs. This
        remapping is applied only to the fixed initial RGB lookup, after the
        global flow offset has been computed.
        """
        if not self.visible_uv_texture_enabled:
            return super().get_initial_rgb(uv)

        u_start, u_span, v_start, v_span = self.visible_uv_texture_bounds.unbind()
        uv01 = uv + 0.5

        # U is periodic. The interval is represented by its start and positive
        # span, so intervals crossing u=0 remain contiguous in local space.
        local_u = torch.remainder(uv01[..., 0] - u_start, 1.0)
        local_v = (uv01[..., 1] - v_start) / v_span.clamp_min(1e-8)
        inside_u = (u_span >= 1.0 - 1e-6) | (local_u <= u_span)
        inside_v = (local_v >= 0.0) & (local_v <= 1.0)
        inside = inside_u & inside_v

        local_u = local_u / u_span.clamp_min(1e-8)
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

    @staticmethod
    def _compute_background_fallback_rgb(
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the RGB prior used for unseen sphere UVs."""
        background_mask = (masks.sum(dim=1, keepdim=True) == 0).to(images.dtype)
        if background_mask.shape[-2:] != images.shape[-2:]:
            background_mask = F.interpolate(
                background_mask,
                size=images.shape[-2:],
                mode="nearest",
            )

        denom = background_mask.sum()
        if denom.item() < 1:
            logger.warning(
                "No background pixels found in masks; falling back to global mean color."
            )
            fallback_rgb = images.mean(dim=(0, 2, 3))
        else:
            fallback_rgb = (images * background_mask).sum(dim=(0, 2, 3)) / denom
        return fallback_rgb.clamp(0, 1)

    @staticmethod
    def _compute_circular_uv_interval(u_values: torch.Tensor) -> Tuple[float, float]:
        """Return the shortest circular interval containing normalized U values."""
        values = torch.remainder(u_values.reshape(-1), 1.0)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            raise ValueError("Cannot compute a circular UV interval from no values.")

        values = torch.sort(values).values
        extended = torch.cat([values, values[:1] + 1.0])
        gaps = extended[1:] - extended[:-1]
        largest_gap_idx = int(torch.argmax(gaps).item())
        largest_gap = float(gaps[largest_gap_idx].item())
        start_idx = (largest_gap_idx + 1) % values.numel()
        start = float(values[start_idx].item())
        span = max(0.0, min(1.0, 1.0 - largest_gap))
        return start, span

    @classmethod
    def _compute_visible_uv_texture_layout(
        cls,
        camera: TimedCameraSceneNode3D,
        times: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
        resolution: VEC_TYPE,
        margin_uv: VEC_TYPE = (0.05, 0.05),
        interior_samples_per_axis: int = 9,
        y_flip: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any], int]:
        """Find a global UV tile covering the camera's sphere intersections.

        All image-border pixel centres are sampled so perspective/spherical
        curvature and lens distortion cannot make the tile too small. A small
        interior grid catches views approaching an equirectangular pole, where
        coordinate extrema need not lie on the image boundary.
        """
        if len(margin_uv) != 2:
            raise ValueError("margin_uv must contain [u_margin, v_margin].")
        margin_u = float(margin_uv[0])
        margin_v = float(margin_uv[1])
        if margin_u < 0 or margin_v < 0:
            raise ValueError("margin_uv values must be non-negative.")
        if interior_samples_per_axis < 1:
            raise ValueError("interior_samples_per_axis must be positive.")

        device = times.device
        dtype = times.dtype
        image_height = int(camera._image_resolution[0].item())
        image_width = int(camera._image_resolution[1].item())
        if image_height < 2 or image_width < 2:
            return None, {"enabled": False, "reason": "invalid_image_resolution"}, 0

        # Use every border pixel centre. This is inexpensive compared with
        # reprojecting the full color texture and avoids corner-only bounds.
        x_border = torch.arange(image_width, dtype=dtype, device=device)
        y_border = torch.arange(image_height, dtype=dtype, device=device)
        top = torch.stack((x_border, torch.zeros_like(x_border)), dim=-1)
        bottom = torch.stack(
            (x_border, torch.full_like(x_border, image_height - 1)), dim=-1
        )
        left = torch.stack((torch.zeros_like(y_border), y_border), dim=-1)
        right = torch.stack(
            (torch.full_like(y_border, image_width - 1), y_border), dim=-1
        )

        interior_count = max(1, int(interior_samples_per_axis))
        interior_axis_x = torch.linspace(
            0, image_width - 1, interior_count, dtype=dtype, device=device
        )
        interior_axis_y = torch.linspace(
            0, image_height - 1, interior_count, dtype=dtype, device=device
        )
        interior_y, interior_x = torch.meshgrid(
            interior_axis_y, interior_axis_x, indexing="ij"
        )
        interior = torch.stack((interior_x.reshape(-1), interior_y.reshape(-1)), dim=-1)
        pixel_uv = torch.cat((top, bottom, left, right, interior), dim=0)

        ray_origins, ray_directions = camera.get_global_rays(
            uv=pixel_uv,
            t=times,
            uv_includes_time=False,
        )
        sphere_points, hit, _ = ray_sphere_intersection(
            ray_origins,
            ray_directions,
            center,
            radius,
        )
        local_directions = sphere_points - center
        uv01 = direction_to_equirect_uv(local_directions, y_flip=y_flip)
        valid = hit & torch.isfinite(uv01).all(dim=-1)
        uv01 = uv01[valid]
        if uv01.numel() == 0:
            return None, {"enabled": False, "reason": "no_valid_sphere_intersections"}, 0

        raw_u_start, raw_u_span = cls._compute_circular_uv_interval(uv01[:, 0])
        raw_v_min = float(uv01[:, 1].min().item())
        raw_v_max = float(uv01[:, 1].max().item())

        expanded_u_span = min(1.0, raw_u_span + 2.0 * margin_u)
        if expanded_u_span >= 1.0 - 1e-7:
            u_start = 0.0
            expanded_u_span = 1.0
        else:
            u_start = (raw_u_start - margin_u) % 1.0

        v_start = max(0.0, raw_v_min - margin_v)
        v_end = min(1.0, raw_v_max + margin_v)
        expanded_v_span = max(1e-7, v_end - v_start)
        if expanded_v_span >= 1.0 - 1e-7:
            v_start = 0.0
            expanded_v_span = 1.0

        total_texels = int(resolution[0]) * int(resolution[1])
        angular_aspect = max(1e-7, 2.0 * expanded_u_span / expanded_v_span)
        tile_height = max(1, int(round((total_texels / angular_aspect) ** 0.5)))
        tile_width = max(1, total_texels // tile_height)
        tile_width = min(tile_width, total_texels)

        bounds = torch.tensor(
            [u_start, expanded_u_span, v_start, expanded_v_span],
            dtype=dtype,
            device=device,
        )
        metadata = {
            "enabled": True,
            "reason": "visible_uv_bounds",
            "raw_u_start": raw_u_start,
            "raw_u_span": raw_u_span,
            "raw_v_min": raw_v_min,
            "raw_v_max": raw_v_max,
            "u_start": u_start,
            "u_span": expanded_u_span,
            "v_start": v_start,
            "v_span": expanded_v_span,
            "raw_uv_area": raw_u_span * max(0.0, raw_v_max - raw_v_min),
            "expanded_uv_area": expanded_u_span * expanded_v_span,
            "density_gain": 1.0 / (expanded_u_span * expanded_v_span),
            "border_sample_count": int(pixel_uv.shape[0] - interior.shape[0]),
            "interior_sample_count": int(interior.shape[0]),
            "resolution": [tile_height, tile_width],
            "total_texels": total_texels,
            "allocated_texels": tile_height * tile_width,
        }
        return bounds, metadata, tile_height * tile_width

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
        uv_bounds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build an equirectangular initialization by sampling valid background pixels."""
        tex_h = int(resolution[0])
        tex_w = int(resolution[1])

        # Project each texel center back into the cameras so the sphere starts
        # from a stitched background texture instead of a single mean color.
        if uv_bounds is None:
            u_start, u_span, v_start, v_span = 0.0, 1.0, 0.0, 1.0
        else:
            u_start, u_span, v_start, v_span = [
                float(value) for value in uv_bounds.detach().cpu().tolist()
            ]
        v = v_start + v_span * (
            torch.arange(tex_h, dtype=dtype, device=times.device) + 0.5
        ) / tex_h
        u = u_start + u_span * (
            torch.arange(tex_w, dtype=dtype, device=times.device) + 0.5
        ) / tex_w
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
        fallback_rgb = cls._compute_background_fallback_rgb(images, masks)

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
        sphere_visible_uv_texture: Optional[Dict[str, Any]] = None,
        sphere_radius_override: Optional[float] = None,
        sphere_radius_override_source: Optional[str] = None,
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
        computed_radius = (max_cam_dist + float(scene_cutoff_distance)) * (1.0 + float(relative_scale_margin))
        radius = computed_radius
        radius_source = "computed"
        override_value = None
        if sphere_radius_override is not None:
            override_radius = tensorify(sphere_radius_override, dtype=dtype, device=times.device)
            if override_radius.numel() != 1:
                raise ValueError("sphere_radius_override must be a scalar value.")
            override_radius = override_radius.reshape(())
            if not torch.isfinite(override_radius):
                raise ValueError(f"sphere_radius_override must be finite, got {sphere_radius_override}.")
            if override_radius <= 0:
                raise ValueError(f"sphere_radius_override must be positive, got {float(override_radius.detach().cpu().item()):.4f}.")
            if override_radius <= max_cam_dist.to(dtype=override_radius.dtype, device=override_radius.device):
                raise ValueError(
                    "sphere_radius_override must be greater than max_cam_dist "
                    f"({float(max_cam_dist.detach().cpu().item()):.4f}), got {float(override_radius.detach().cpu().item()):.4f}."
                )
            radius = override_radius
            radius_source = sphere_radius_override_source or "override"
            override_value = f"{float(override_radius.detach().cpu().item()):.4f}"

        # Optional local texture layout. The whole configured texture budget is
        # assigned to the visible tile; unexpected UVs use a colour prior and
        # the learned residual network rather than a second texture.
        visible_uv_texture_enabled = False
        visible_uv_texture_bounds = None
        visible_uv_texture_metadata: Dict[str, Any] = {
            "enabled": False,
            "reason": "disabled",
        }
        visible_uv_texture_fallback_rgb = cls._compute_background_fallback_rgb(
            images, masks
        )
        visible_cfg: Dict[str, Any] = {}
        if sphere_visible_uv_texture not in (None, DEFAULT):
            if not isinstance(sphere_visible_uv_texture, dict):
                raise ValueError("sphere_visible_uv_texture must be a mapping.")
            visible_cfg = dict(sphere_visible_uv_texture)

        if bool(visible_cfg.get("enabled", False)):
            margin_uv = visible_cfg.get("margin_uv", (0.05, 0.05))
            interior_samples_per_axis = int(
                visible_cfg.get("interior_samples_per_axis", 9)
            )
            visible_uv_texture_bounds, visible_uv_texture_metadata, _ = cls._compute_visible_uv_texture_layout(
                camera=camera,
                times=times,
                center=center.to(dtype),
                radius=radius.to(dtype),
                resolution=color_mask_resolution,
                margin_uv=margin_uv,
                interior_samples_per_axis=interior_samples_per_axis,
                y_flip=sphere_uv_y_flip,
            )
            visible_uv_texture_enabled = visible_uv_texture_bounds is not None
            if not visible_uv_texture_enabled:
                logger.warning(
                    "Visible UV sphere texture disabled: "
                    f"{visible_uv_texture_metadata.get('reason', 'unknown reason')}"
                )
        else:
            visible_uv_texture_metadata["reason"] = "disabled_by_config"

        sphere_geometry = {
            "max_cam_dist": f"{float(max_cam_dist.detach().cpu().item()):.4f}",
            "scene_cutoff_distance": f"{float(scene_cutoff_distance):.4f}",
            "relative_scale_margin": f"{float(relative_scale_margin):.4f}",
            "computed_radius": f"{float(computed_radius.detach().cpu().item()):.4f}",
            "sphere_radius_override": override_value,
            "radius_source": radius_source,
            "radius": f"{float(radius.detach().cpu().item()):.4f}",
            "center": [round(float(v), 4) for v in center.detach().cpu().tolist()],
            "color_mask_resolution": color_mask_resolution,
            "visible_uv_texture": visible_uv_texture_metadata,
        }
        _write_sphere_geometry_diagnostic(config, sphere_geometry)
        logger.info(
            "Sphere background geometry: "
            f"max_cam_dist={sphere_geometry['max_cam_dist']}, "
            f"scene_cutoff_distance={sphere_geometry['scene_cutoff_distance']}, "
            f"relative_scale_margin={sphere_geometry['relative_scale_margin']}, "
            f"computed_radius={sphere_geometry['computed_radius']}, "
            f"sphere_radius_override={sphere_geometry['sphere_radius_override']}, "
            f"radius_source={sphere_geometry['radius_source']}, "
            f"radius={sphere_geometry['radius']}, "
            f"center={sphere_geometry['center']}, "
            f"color_mask_resolution={sphere_geometry['color_mask_resolution']}, "
            f"visible_uv_texture={sphere_geometry['visible_uv_texture']}"
        )

        if proxy_init:
            if visible_uv_texture_enabled:
                init_resolution = visible_uv_texture_metadata["resolution"]
            else:
                init_resolution = color_mask_resolution
            init_rgb = torch.zeros(
                (3, int(init_resolution[0]), int(init_resolution[1])),
                dtype=dtype,
                device=times.device,
            )
        else:
            with torch.no_grad():
                # Unlike the plane background, the sphere has no natural
                # reference view, so seed it by averaging reprojected camera
                # colors into an equirectangular texture instead of one flat mean.
                if visible_uv_texture_enabled:
                    init_resolution = visible_uv_texture_metadata["resolution"]
                else:
                    init_resolution = color_mask_resolution
                init_rgb = BackgroundImageSphereSceneNode3D._reproject_images_to_sphere_texture(
                    camera=camera,
                    images=images,
                    masks=masks,
                    times=times,
                    center=center.to(dtype),
                    radius=radius.to(dtype),
                    resolution=init_resolution,
                    dtype=dtype,
                    y_flip=sphere_uv_y_flip,
                    align_corners=getattr(config, "plane_align_corners", True) if config is not None else True,
                    uv_bounds=visible_uv_texture_bounds if visible_uv_texture_enabled else None,
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
            visible_uv_texture_bounds=(
                visible_uv_texture_bounds if visible_uv_texture_enabled else None
            ),
            visible_uv_texture_fallback_rgb=(
                visible_uv_texture_fallback_rgb if visible_uv_texture_enabled else None
            ),
            visible_uv_texture_metadata=(
                visible_uv_texture_metadata if visible_uv_texture_enabled else None
            ),
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
