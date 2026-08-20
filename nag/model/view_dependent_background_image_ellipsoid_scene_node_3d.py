from __future__ import annotations

"""View-dependent texture model on an oriented ellipsoid background."""

import os
from typing import Any, Dict, Optional, Tuple

import torch

from nag.model.background_image_ellipsoid_scene_node_3d import (
    BackgroundImageEllipsoidSceneNode3D,
)
from nag.model.sphere_utils import normalize, to_tangent_frame
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local
from nag.model.view_dependent_background_image_sphere_scene_node_3d import (
    ViewDependentBackgroundImageSphereSceneNode3D,
)
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.util.typing import VEC_TYPE


class ViewDependentBackgroundImageEllipsoidSceneNode3D(
    ViewDependentBackgroundImageSphereSceneNode3D
):
    """View-dependent ellipsoid using normalized local coordinates for UVs."""

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
        return BackgroundImageEllipsoidSceneNode3D.intersect_rays(
            self, ray_origins, ray_directions, global_position
        )

    def get_global_surface_normals(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageEllipsoidSceneNode3D.get_global_surface_normals(
            self, points_global, global_position
        )

    def _global_points_to_sphere_uv(
        self,
        points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageEllipsoidSceneNode3D._global_points_to_sphere_uv(
            self, points_global, global_position
        )

    def _sphere_uv_to_global_points(
        self,
        uv01: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        return BackgroundImageEllipsoidSceneNode3D._sphere_uv_to_global_points(
            self, uv01, global_position
        )

    @classmethod
    def _compute_visible_uv_texture_layout(cls, *args, **kwargs):
        return BackgroundImageEllipsoidSceneNode3D._compute_visible_uv_texture_layout.__func__(
            cls, *args, **kwargs
        )

    @classmethod
    def _reproject_images_to_ellipsoid_texture(cls, *args, **kwargs):
        return BackgroundImageEllipsoidSceneNode3D._reproject_images_to_ellipsoid_texture.__func__(
            cls, *args, **kwargs
        )

    @staticmethod
    def _project_points_to_grid(*args, **kwargs):
        return BackgroundImageEllipsoidSceneNode3D._project_points_to_grid(
            *args, **kwargs
        )

    def _to_local_tangent_rays(
        self,
        ray_origins_global: torch.Tensor,
        ray_directions_global: torch.Tensor,
        hit_points_global: torch.Tensor,
        global_position: torch.Tensor,
    ) -> torch.Tensor:
        ray_targets_global = ray_origins_global + ray_directions_global
        ray_origins_local = global_to_local(
            global_position, ray_origins_global, v_include_time=True
        )[..., :3]
        ray_targets_local = global_to_local(
            global_position, ray_targets_global, v_include_time=True
        )[..., :3]
        ray_directions_local = ray_targets_local - ray_origins_local
        hit_points_local = global_to_local(
            global_position, hit_points_global, v_include_time=True
        )[..., :3]
        radii = self.get_ellipsoid_radii().to(
            dtype=hit_points_local.dtype, device=hit_points_local.device
        )
        local_normals = normalize(hit_points_local / (radii * radii))
        return to_tangent_frame(ray_directions_local, local_normals)

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
        ellipsoid_path: Optional[os.PathLike] = None,
        ellipsoid_scale_margin: float = 0.02,
        ellipsoid_radii_override: Optional[VEC_TYPE] = None,
        ellipsoid_max_radius: Optional[float] = None,
        **kwargs,
    ):
        return BackgroundImageEllipsoidSceneNode3D.for_camera.__func__(
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
            sphere_visible_uv_texture=sphere_visible_uv_texture,
            ellipsoid_visible_uv_texture=ellipsoid_visible_uv_texture,
            ellipsoid_path=ellipsoid_path,
            ellipsoid_scale_margin=ellipsoid_scale_margin,
            ellipsoid_radii_override=ellipsoid_radii_override,
            ellipsoid_max_radius=ellipsoid_max_radius,
            **kwargs,
        )
