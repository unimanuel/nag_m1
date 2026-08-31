from __future__ import annotations

"""PCA-ellipsoid background geometry loaded from Waymo preprocessing output."""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from nag.model.background_image_sphere_scene_node_3d import (
    BackgroundImageSphereSceneNode3D,
)
from nag.model.sphere_utils import (
    direction_to_equirect_uv,
    equirect_uv_to_direction,
    normalize,
    ray_ellipsoid_intersection,
)
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global
from tools.logger.logging import logger
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.geometric.mappings import rotmat_to_unitquat
from tools.util.torch import tensorify
from tools.util.typing import DEFAULT, VEC_TYPE


@dataclass(frozen=True)
class LoadedEllipsoidGeometry:
    source_path: str
    sequence: str
    frame_slice: Dict[str, int]
    center: torch.Tensor
    source_radii: torch.Tensor
    radii: torch.Tensor
    radii_override: Optional[torch.Tensor]
    max_radius: Optional[float]
    rotation: torch.Tensor
    scale_margin: float
    point_count: int
    source_contains_all_points: bool
    source_point_coverage_fraction: float


def load_ellipsoid_geometry(
    path: Union[os.PathLike, str],
    scale_margin: float,
    radii_override: Optional[VEC_TYPE] = None,
    max_radius: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> LoadedEllipsoidGeometry:
    """Load and validate the geometry contract produced by ``nagwp ellipsoid``."""

    source_path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Ellipsoid geometry does not exist: {source_path}")
    if not 0.0 <= float(scale_margin) <= 1.0:
        raise ValueError(
            f"background_ellipsoid_scale_margin must be in [0, 1], got {scale_margin}."
        )
    with open(source_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    format_version = int(document.get("format_version", 0))
    if format_version not in (1, 2, 3):
        raise ValueError(
            f"Unsupported ellipsoid format version in {source_path}: "
            f"{document.get('format_version')!r}."
        )
    fit = document.get("fit")
    if not isinstance(fit, dict):
        raise ValueError(f"Ellipsoid JSON has no fit mapping: {source_path}")

    center = tensorify(fit.get("center"), dtype=dtype, device=device).reshape(-1)
    source_radii = tensorify(fit.get("radii"), dtype=dtype, device=device).reshape(-1)
    rotation = tensorify(
        fit.get("rotation_columns"), dtype=dtype, device=device
    )
    if center.shape != (3,) or source_radii.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError(
            f"Expected center (3), radii (3), and rotation_columns (3,3) in {source_path}; "
            f"received {tuple(center.shape)}, {tuple(source_radii.shape)}, {tuple(rotation.shape)}."
        )
    if not (
        torch.isfinite(center).all()
        and torch.isfinite(source_radii).all()
        and torch.isfinite(rotation).all()
    ):
        raise ValueError(f"Ellipsoid geometry contains non-finite values: {source_path}")
    if torch.any(source_radii <= 0):
        raise ValueError(f"Ellipsoid radii must be positive in {source_path}.")
    identity = torch.eye(3, dtype=dtype, device=rotation.device)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Ellipsoid rotation is not orthonormal in {source_path}.")
    determinant = torch.linalg.det(rotation)
    if not torch.isclose(determinant, torch.ones_like(determinant), atol=1e-4, rtol=1e-4):
        raise ValueError(
            f"Ellipsoid rotation must be right-handed in {source_path}; det={determinant.item():.6g}."
        )

    frame_slice = document.get("frame_slice")
    if not isinstance(frame_slice, dict):
        raise ValueError(f"Ellipsoid JSON has no frame_slice mapping: {source_path}")
    required_slice_fields = ("start", "stop", "step", "first", "last", "count")
    if any(field not in frame_slice for field in required_slice_fields):
        raise ValueError(f"Ellipsoid frame_slice is incomplete in {source_path}.")
    normalized_slice = {field: int(frame_slice[field]) for field in required_slice_fields}
    source_contains_all_points = bool(fit.get("contains_all_points", False))
    source_point_coverage_fraction = float(
        fit.get("point_coverage_fraction", 1.0 if source_contains_all_points else 0.0)
    )
    if not 0.0 <= source_point_coverage_fraction <= 1.0:
        raise ValueError(
            f"Ellipsoid point coverage must be in [0, 1] in {source_path}, got "
            f"{source_point_coverage_fraction}."
        )
    if format_version == 1 and not source_contains_all_points:
        raise ValueError(f"Ellipsoid source fit does not contain all selected points: {source_path}")
    if format_version == 2 and source_point_coverage_fraction < 0.99:
        raise ValueError(
            f"Robust ellipsoid source fit covers only {source_point_coverage_fraction:.6f} of "
            f"selected points in {source_path}; at least 0.99 is required."
        )
    # Version 3 is a ray-depth proxy fit.  It intentionally minimizes
    # camera-ray depth mismatch rather than enclosing a fixed percentage of
    # point samples, so its validity is established by the final camera-inside
    # check in ``for_camera`` below instead of a point-coverage threshold.

    if radii_override is not None and max_radius is not None:
        raise ValueError(
            "background_ellipsoid_radii_override and background_ellipsoid_max_radius "
            "cannot be used together."
        )
    normalized_max_radius = None
    if max_radius is not None:
        normalized_max_radius = float(max_radius)
        if not math.isfinite(normalized_max_radius) or normalized_max_radius <= 0:
            raise ValueError("background_ellipsoid_max_radius must be finite and positive.")

    effective_radii = source_radii * (1.0 + float(scale_margin))
    override_tensor = None
    if radii_override is not None:
        override_tensor = tensorify(radii_override, dtype=dtype, device=device).reshape(-1)
        if override_tensor.shape != (3,):
            raise ValueError(
                "background_ellipsoid_radii_override must contain exactly three values; "
                f"received shape {tuple(override_tensor.shape)}."
            )
        if not torch.isfinite(override_tensor).all() or torch.any(override_tensor <= 0):
            raise ValueError(
                "background_ellipsoid_radii_override must contain three finite positive values."
            )
        effective_radii = override_tensor
    elif normalized_max_radius is not None:
        effective_radii = torch.clamp(effective_radii, max=normalized_max_radius)

    return LoadedEllipsoidGeometry(
        source_path=source_path,
        sequence=str(document.get("sequence", "")),
        frame_slice=normalized_slice,
        center=center,
        source_radii=source_radii,
        radii=effective_radii,
        radii_override=override_tensor,
        max_radius=normalized_max_radius,
        rotation=rotation,
        scale_margin=float(scale_margin),
        point_count=int(document.get("point_count", 0)),
        source_contains_all_points=source_contains_all_points,
        source_point_coverage_fraction=source_point_coverage_fraction,
    )


def validate_ellipsoid_matches_config(
    geometry: LoadedEllipsoidGeometry,
    config: Any,
) -> None:
    """Reject accidental pairing of a YAML interval with another fit."""

    frame_filter = getattr(config, "frame_indices_filter", None)
    if not isinstance(frame_filter, slice):
        raise ValueError("Ellipsoid backgrounds require a slice frame_indices_filter.")
    expected = {
        "start": int(frame_filter.start or 0),
        "stop": int(frame_filter.stop),
        "step": int(frame_filter.step or 1),
    }
    actual = {field: geometry.frame_slice[field] for field in expected}
    if actual != expected:
        raise ValueError(
            f"Ellipsoid interval {actual} from {geometry.source_path} does not match "
            f"the NAG configuration interval {expected}."
        )
    data_path = os.fspath(getattr(config, "data_path", ""))
    configured_sequence = os.path.basename(os.path.normpath(data_path))
    if geometry.sequence != configured_sequence:
        raise ValueError(
            f"Ellipsoid sequence {geometry.sequence!r} from {geometry.source_path} does not "
            f"match NAG data sequence {configured_sequence!r}."
        )


def _write_ellipsoid_geometry_diagnostic(config: Any, geometry: Dict[str, Any]) -> None:
    output_path = getattr(config, "output_path", None)
    if output_path is None:
        return
    diagnostic_dir = os.path.join(str(output_path), "initial", "background_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    diagnostic_path = os.path.join(diagnostic_dir, "ellipsoid_geometry.txt")
    lines = ["Ellipsoid Background Geometry Diagnostic", ""]
    lines.extend([f"{key}: {value}" for key, value in geometry.items()])
    with open(diagnostic_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


class BackgroundImageEllipsoidSceneNode3D(BackgroundImageSphereSceneNode3D):
    """Background texture parameterized on a fixed, oriented ellipsoid."""

    is_spherical_background: bool = False
    is_ellipsoidal_background: bool = True
    is_curved_background: bool = True

    def __init__(
        self,
        num_flow_control_points: int,
        ellipsoid_radii: VEC_TYPE,
        ellipsoid_source_path: Optional[str] = None,
        ellipsoid_geometry_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            num_flow_control_points=num_flow_control_points,
            sphere_radius=1.0,
            **kwargs,
        )
        radii = tensorify(
            ellipsoid_radii,
            dtype=self.dtype,
            device=self._translation.device,
        ).reshape(-1)
        if radii.shape != (3,) or not torch.isfinite(radii).all() or torch.any(radii <= 0):
            raise ValueError("ellipsoid_radii must contain three finite positive values.")
        self.register_buffer("ellipsoid_radii", radii)
        self.ellipsoid_source_path = ellipsoid_source_path
        self.ellipsoid_geometry_metadata = dict(ellipsoid_geometry_metadata or {})

    def get_ellipsoid_radii(self) -> torch.Tensor:
        return self.ellipsoid_radii

    def intersect_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        global_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ray_ellipsoid_intersection(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            center=global_position[..., :3, 3],
            radii=self.get_ellipsoid_radii(),
            rotation=global_position[..., :3, :3],
        )

    def get_global_surface_normals(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        local_points = global_to_local(
            global_position, points_global, v_include_time=True
        )[..., :3]
        radii = self.get_ellipsoid_radii().to(
            dtype=local_points.dtype, device=local_points.device
        )
        local_normals = local_points / (radii * radii)
        rotation = global_position[..., :3, :3]
        world_normals = torch.matmul(
            local_normals.unsqueeze(-2), rotation.transpose(-1, -2)
        ).squeeze(-2)
        return normalize(world_normals)

    def _global_points_to_sphere_uv(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        local_points = global_to_local(
            global_position, points_global, v_include_time=True
        )[..., :3]
        radii = self.get_ellipsoid_radii().to(
            dtype=local_points.dtype, device=local_points.device
        )
        unit_sphere_points = local_points / radii
        return direction_to_equirect_uv(
            unit_sphere_points, y_flip=self.sphere_uv_y_flip
        )

    def _sphere_uv_to_global_points(
        self,
        uv01: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        local_directions = equirect_uv_to_direction(
            uv01, y_flip=self.sphere_uv_y_flip
        )
        radii = self.get_ellipsoid_radii().to(
            dtype=uv01.dtype, device=uv01.device
        )
        local_points = local_directions * radii
        return local_to_global(
            global_position, local_points, v_include_time=True
        )[..., :3]

    @classmethod
    def _compute_visible_uv_texture_layout(
        cls,
        camera: TimedCameraSceneNode3D,
        times: torch.Tensor,
        center: torch.Tensor,
        radii: torch.Tensor,
        rotation: torch.Tensor,
        resolution: VEC_TYPE,
        margin_uv: VEC_TYPE = (0.05, 0.05),
        interior_samples_per_axis: int = 9,
        y_flip: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any], int]:
        if len(margin_uv) != 2:
            raise ValueError("margin_uv must contain [u_margin, v_margin].")
        margin_u, margin_v = float(margin_uv[0]), float(margin_uv[1])
        if margin_u < 0 or margin_v < 0:
            raise ValueError("margin_uv values must be non-negative.")
        if interior_samples_per_axis < 1:
            raise ValueError("interior_samples_per_axis must be positive.")

        device, dtype = times.device, times.dtype
        image_height = int(camera._image_resolution[0].item())
        image_width = int(camera._image_resolution[1].item())
        if image_height < 2 or image_width < 2:
            return None, {"enabled": False, "reason": "invalid_image_resolution"}, 0

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
        interior_axis_x = torch.linspace(
            0, image_width - 1, int(interior_samples_per_axis), dtype=dtype, device=device
        )
        interior_axis_y = torch.linspace(
            0, image_height - 1, int(interior_samples_per_axis), dtype=dtype, device=device
        )
        interior_y, interior_x = torch.meshgrid(
            interior_axis_y, interior_axis_x, indexing="ij"
        )
        interior = torch.stack(
            (interior_x.reshape(-1), interior_y.reshape(-1)), dim=-1
        )
        pixel_uv = torch.cat((top, bottom, left, right, interior), dim=0)

        with torch.no_grad():
            ray_origins, ray_directions = camera.get_global_rays(
                uv=pixel_uv, t=times, uv_includes_time=False
            )
            surface_points, hit, _ = ray_ellipsoid_intersection(
                ray_origins,
                ray_directions,
                center,
                radii,
                rotation,
            )
            local_points = torch.matmul(
                (surface_points - center).unsqueeze(-2), rotation
            ).squeeze(-2)
            uv01 = direction_to_equirect_uv(
                local_points / radii, y_flip=y_flip
            )
            valid = hit & torch.isfinite(uv01).all(dim=-1)
            uv01 = uv01[valid]
        if uv01.numel() == 0:
            return None, {"enabled": False, "reason": "no_valid_ellipsoid_intersections"}, 0

        raw_u_start, raw_u_span = cls._compute_circular_uv_interval(uv01[:, 0])
        raw_v_min, raw_v_max = float(uv01[:, 1].min()), float(uv01[:, 1].max())
        expanded_u_span = min(1.0, raw_u_span + 2.0 * margin_u)
        if expanded_u_span >= 1.0 - 1e-7:
            u_start, expanded_u_span = 0.0, 1.0
        else:
            u_start = (raw_u_start - margin_u) % 1.0
        v_start = max(0.0, raw_v_min - margin_v)
        v_end = min(1.0, raw_v_max + margin_v)
        expanded_v_span = max(1e-7, v_end - v_start)
        if expanded_v_span >= 1.0 - 1e-7:
            v_start, expanded_v_span = 0.0, 1.0

        total_texels = int(resolution[0]) * int(resolution[1])
        angular_aspect = max(1e-7, 2.0 * expanded_u_span / expanded_v_span)
        tile_height = max(1, int(round((total_texels / angular_aspect) ** 0.5)))
        tile_width = min(total_texels, max(1, total_texels // tile_height))
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
    def _reproject_images_to_ellipsoid_texture(
        cls,
        camera: TimedCameraSceneNode3D,
        images: torch.Tensor,
        masks: torch.Tensor,
        times: torch.Tensor,
        center: torch.Tensor,
        radii: torch.Tensor,
        rotation: torch.Tensor,
        resolution: VEC_TYPE,
        dtype: torch.dtype,
        y_flip: bool = False,
        align_corners: bool = True,
        uv_bounds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex_h, tex_w = int(resolution[0]), int(resolution[1])
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
        texture_uv = torch.stack([uu, vv], dim=-1)
        unit_directions = equirect_uv_to_direction(texture_uv, y_flip=y_flip)
        local_points = unit_directions * radii
        surface_points = center.view(1, 1, 3) + torch.matmul(
            local_points, rotation.transpose(-1, -2)
        )
        local_normals = unit_directions / radii
        surface_normals = normalize(
            torch.matmul(local_normals, rotation.transpose(-1, -2))
        )

        background_mask = (masks.sum(dim=1, keepdim=True) == 0).to(images.dtype)
        if background_mask.shape[-2:] != images.shape[-2:]:
            background_mask = F.interpolate(
                background_mask, size=images.shape[-2:], mode="nearest"
            )
        fallback_rgb = cls._compute_background_fallback_rgb(images, masks)
        best_rgb = torch.zeros((3, tex_h, tex_w), dtype=dtype, device=times.device)
        best_weight = torch.zeros((1, tex_h, tex_w), dtype=dtype, device=times.device)

        for frame_index, frame_time in enumerate(times):
            image_points, valid = camera.global_to_image_coordinates(
                surface_points,
                t=frame_time.unsqueeze(0),
                v_includes_time=False,
                clamp=False,
                return_ok=True,
            )
            image_points = image_points[..., 0, :]
            valid = valid[..., 0].to(dtype).unsqueeze(0)
            sample_grid = cls._project_points_to_grid(
                image_points=image_points,
                image_resolution=camera._image_resolution,
                align_corners=align_corners,
            ).unsqueeze(0)
            dist2 = sample_grid[0, ..., 0] ** 2 + sample_grid[0, ..., 1] ** 2
            center_weight = torch.exp(-dist2).unsqueeze(0)
            camera_position = camera.get_global_position(
                t=frame_time.unsqueeze(0)
            )[0, :3, 3].to(dtype)
            view_direction = normalize(
                surface_points - camera_position.view(1, 1, 3)
            )
            incidence_weight = (
                (view_direction * surface_normals)
                .sum(dim=-1)
                .clamp_min(0.0)
                .unsqueeze(0)
            )
            sampled_rgb = F.grid_sample(
                images[frame_index : frame_index + 1].to(dtype=dtype),
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=align_corners,
            )[0]
            sampled_background = F.grid_sample(
                background_mask[frame_index : frame_index + 1],
                sample_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=align_corners,
            )[0]
            weight = sampled_background * valid * center_weight * incidence_weight
            use_sample = weight > best_weight
            best_rgb = torch.where(use_sample.expand_as(best_rgb), sampled_rgb, best_rgb)
            best_weight = torch.where(use_sample, weight, best_weight)

        return torch.where(
            best_weight > 0,
            best_rgb,
            fallback_rgb[:, None, None].expand(3, tex_h, tex_w),
        ).contiguous()

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
        ellipsoid_visible_uv_texture: Optional[Dict[str, Any]] = None,
        ellipsoid_path: Optional[Union[os.PathLike, str]] = None,
        ellipsoid_scale_margin: float = 0.02,
        ellipsoid_radii_override: Optional[VEC_TYPE] = None,
        ellipsoid_max_radius: Optional[float] = None,
        **kwargs,
    ):
        if world is None:
            raise ValueError("world must be provided")
        if ellipsoid_path is None:
            raise ValueError("background_ellipsoid_path must be configured for an ellipsoid background.")
        geometry = load_ellipsoid_geometry(
            ellipsoid_path,
            scale_margin=ellipsoid_scale_margin,
            radii_override=ellipsoid_radii_override,
            max_radius=ellipsoid_max_radius,
            dtype=dtype,
            device=times.device,
        )
        if config is not None:
            validate_ellipsoid_matches_config(geometry, config)
        center, radii, rotation = geometry.center, geometry.radii, geometry.rotation

        camera_positions = camera.get_global_position(t=times)[:, :3, 3].detach().to(dtype)
        camera_local = torch.matmul(camera_positions - center, rotation) / radii
        camera_normalized_radii = camera_local.norm(dim=-1)
        max_camera_normalized_radius = camera_normalized_radii.max()
        if max_camera_normalized_radius >= 1.0:
            raise ValueError(
                f"Camera trajectory is outside ellipsoid {geometry.source_path}: maximum "
                f"normalized radius is {max_camera_normalized_radius.item():.6f}. Increase "
                "background_ellipsoid_scale_margin or regenerate the fit."
            )

        if color_mask_resolution is None or color_mask_resolution == DEFAULT:
            resolution_x = min(camera._image_resolution[1], images.shape[3])
            resolution_y = min(camera._image_resolution[0], images.shape[2])
            base_height = int(round(float(resolution_y) * (1.0 + float(relative_scale_margin))))
            base_width = int(round(float(resolution_x) * (1.0 + float(relative_scale_margin))))
            ellipsoid_height = max(512, base_height)
            ellipsoid_width = max(1024, base_width, 2 * ellipsoid_height)
            color_mask_resolution = [ellipsoid_height, ellipsoid_width]
            logger.info(
                "Color resolution on ellipsoid background set to default. "
                f"Using resolution {ellipsoid_height} x {ellipsoid_width} (H x W)."
            )
        if num_flow_control_points is None:
            if config is not None and hasattr(config, "plane_flow_control_points_ratio"):
                num_flow_control_points = int(
                    round(len(times) * config.plane_flow_control_points_ratio)
                )
            else:
                num_flow_control_points = 1

        visible_config = ellipsoid_visible_uv_texture
        if visible_config in (None, DEFAULT):
            visible_config = sphere_visible_uv_texture
        visible_config = {} if visible_config in (None, DEFAULT) else dict(visible_config)
        visible_bounds = None
        visible_metadata: Dict[str, Any] = {"enabled": False, "reason": "disabled_by_config"}
        fallback_rgb = cls._compute_background_fallback_rgb(images, masks)
        if bool(visible_config.get("enabled", False)):
            visible_bounds, visible_metadata, _ = cls._compute_visible_uv_texture_layout(
                camera=camera,
                times=times,
                center=center,
                radii=radii,
                rotation=rotation,
                resolution=color_mask_resolution,
                margin_uv=visible_config.get("margin_uv", (0.05, 0.05)),
                interior_samples_per_axis=int(
                    visible_config.get("interior_samples_per_axis", 9)
                ),
                y_flip=sphere_uv_y_flip,
            )
            if visible_bounds is None:
                logger.warning(
                    "Visible UV ellipsoid texture disabled: "
                    f"{visible_metadata.get('reason', 'unknown reason')}"
                )

        visible_enabled = visible_bounds is not None
        init_resolution = (
            visible_metadata["resolution"] if visible_enabled else color_mask_resolution
        )
        if proxy_init:
            initial_rgb = torch.zeros(
                (3, int(init_resolution[0]), int(init_resolution[1])),
                dtype=dtype,
                device=times.device,
            )
        else:
            with torch.no_grad():
                initial_rgb = cls._reproject_images_to_ellipsoid_texture(
                    camera=camera,
                    images=images,
                    masks=masks,
                    times=times,
                    center=center,
                    radii=radii,
                    rotation=rotation,
                    resolution=init_resolution,
                    dtype=dtype,
                    y_flip=sphere_uv_y_flip,
                    align_corners=(
                        getattr(config, "plane_align_corners", True)
                        if config is not None
                        else True
                    ),
                    uv_bounds=visible_bounds,
                )

        geometry_diagnostic = {
            "source_path": geometry.source_path,
            "sequence": geometry.sequence,
            "frame_slice": geometry.frame_slice,
            "point_count": geometry.point_count,
            "source_contains_all_points": geometry.source_contains_all_points,
            "source_point_coverage_fraction": geometry.source_point_coverage_fraction,
            "source_radii": geometry.source_radii.detach().cpu().tolist(),
            "scale_margin": geometry.scale_margin,
            "radii_override": (
                geometry.radii_override.detach().cpu().tolist()
                if geometry.radii_override is not None
                else None
            ),
            "max_radius": geometry.max_radius,
            "radii_source": (
                "override"
                if geometry.radii_override is not None
                else (
                    "source_plus_margin_capped"
                    if geometry.max_radius is not None
                    else "source_plus_margin"
                )
            ),
            "effective_radii": radii.detach().cpu().tolist(),
            "center": center.detach().cpu().tolist(),
            "rotation_columns": rotation.detach().cpu().tolist(),
            "max_camera_normalized_radius": float(max_camera_normalized_radius),
            "minimum_camera_surface_margin": float(1.0 - max_camera_normalized_radius),
            "scene_cutoff_distance_ignored": float(scene_cutoff_distance),
            "texture_relative_scale_margin": float(relative_scale_margin),
            "color_mask_resolution": color_mask_resolution,
            "visible_uv_texture": visible_metadata,
        }
        _write_ellipsoid_geometry_diagnostic(config, geometry_diagnostic)
        logger.info(
            "Ellipsoid background geometry: "
            f"source={geometry.source_path}, center={geometry_diagnostic['center']}, "
            f"effective_radii={geometry_diagnostic['effective_radii']}, "
            f"scale_margin={geometry.scale_margin:.4f}, "
            f"max_camera_normalized_radius={float(max_camera_normalized_radius):.6f}."
        )

        plane_scale = torch.tensor([1.0, 1.0], dtype=dtype, device=times.device)
        plane_scale_offset = torch.tensor([-0.5, -0.5], dtype=dtype, device=times.device)
        translation = center[None, :].expand(len(camera._times), -1).to(dtype)
        orientation = rotmat_to_unitquat(rotation)[None, :].expand(
            len(camera._times), -1
        )
        del color_mask_smoothing
        del correct_lens_distortion
        del depths
        del dataset

        args = dict(kwargs)
        args.update(
            is_camera_attached=False,
            plane_scale=plane_scale,
            plane_scale_offset=plane_scale_offset,
            translation=translation,
            times=camera._times,
            orientation=orientation,
            initial_rgb=initial_rgb,
            visible_uv_texture_bounds=visible_bounds if visible_enabled else None,
            visible_uv_texture_fallback_rgb=fallback_rgb if visible_enabled else None,
            visible_uv_texture_metadata=visible_metadata if visible_enabled else None,
            flow_reference_time=0,
            num_flow_control_points=num_flow_control_points,
            dtype=dtype,
            name=name,
            index=index,
            ellipsoid_radii=radii,
            ellipsoid_source_path=geometry.source_path,
            ellipsoid_geometry_metadata=geometry_diagnostic,
            sphere_uv_y_flip=sphere_uv_y_flip,
            network_dtype=(
                getattr(config, "tinycudann_network_dtype", torch.float16)
                if config is not None
                else torch.float16
            ),
        )
        if nag_model is not None:
            from nag.model.nag_functional_model import NAGFunctionalModel

            if isinstance(nag_model, NAGFunctionalModel):
                args = nag_model.patch_background_args(args)
        ellipsoid = cls(**args)
        world.add_scene_children(ellipsoid)
        return ellipsoid
