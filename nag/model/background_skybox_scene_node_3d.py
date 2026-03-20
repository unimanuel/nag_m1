import math
from matplotlib.axes import Axes
import numpy as np
import torch
import torch.nn as nn
from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union
from tools.transforms.fittable_transform import FittableTransform

from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import compose_translation_orientation, global_to_local
from tools.transforms.geometric.mappings import unitquat_to_rotvec
from tools.logger.logging import logger
try:
    import tinycudann as tcnn
except (ModuleNotFoundError, OSError) as err:
    from tools.util.mock_import import MockImport
    if not TYPE_CHECKING:
        logger.warning(f"Could not import tinycudann: {err}")
        tcnn = MockImport(mocked_property="tcnn")
from tools.util.typing import VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.to_tensor import tensorify
from tools.transforms.mean_std import MeanStd
from tools.util.typing import DEFAULT, _DEFAULT
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims
from nag.utils import utils
from nag.transforms.transforms_timed_3d import interpolate_vector
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates
from tools.transforms.to_numpy import numpyify
from tools.transforms.geometric.quaternion import quat_average, quat_subtraction

from tools.transforms.geometric.mappings import axis_angle_to_unitquat
import torch.nn.functional as F


def ray_directions_to_skybox_sides(
    ray_directions: torch.Tensor,
    return_decomposed: bool = False
) -> torch.Tensor:
    """
    Determines which side of the skybox each ray intersects, based on its direction.

    Parameters
    ----------
    ray_directions : torch.Tensor
        Global ray directions. Shape (B, T, 3)

    Returns
    -------
    torch.Tensor
        Indices (0-5) of skybox side hit by each ray. Shape (B, T)
        Order: +X, +Y, +Z, -X, -Y, -Z
    Or if return_decomposed is True:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        1. as above
        2. Indices of the normal axis of the skybox side (ordered X, Y, Z). Shape (B, T)
        3. is_neg (0 or 1) for whether the side on the negative end of the axis was hit. Shape (B, T)
    """
    max_axis = ray_directions.abs().argmax(dim=-1) # (B, T)
    is_neg = (ray_directions.gather(-1, max_axis.unsqueeze(-1)).squeeze(-1) < 0).long() # (B, T)
    hit_side = max_axis + 3 * is_neg

    if return_decomposed:
        return hit_side, max_axis, is_neg
    return hit_side


def get_skybox_intersection_points(
    ray_directions: torch.Tensor,
    center: torch.Tensor,
    radius: float = 1000.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with a skybox.

    Parameters
    ----------
    ray_directions : torch.Tensor
        Global ray directions. Shape (B, T, 3)

    center : torch.Tensor
        Skybox center position. Shape (T, 3)
        only used to determine the output positions, all rays are assumed to originate from the center of the skybox

    radius : float
        only used to determine the output positions and intersection distances

    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        1. Intersection points of the rays with the infinite planes in the scene. Shape (N, B, T, 3)
        2. Indices of skybox side hit by each ray. Shape (B, T)
        3. Plane positions per ray as used to compute the intersection. Shape (N, B, T, 3)
        4. Plane normals per ray as used to compute the intersection. Shape (N, B, T, 3)
    """
    side_idx, max_axis, is_neg = ray_directions_to_skybox_sides(ray_directions, return_decomposed=True)
    sign = 1 - 2 * is_neg

    t = sign * radius / (ray_directions.gather(-1, max_axis.unsqueeze(-1)).squeeze(-1) + eps)
    points = center + t.unsqueeze(-1) * ray_directions

    normals = torch.zeros_like(ray_directions)
    normals.scatter_(-1, max_axis.unsqueeze(-1), -sign.float().unsqueeze(-1))
    plane_p = center.expand_as(points) - radius * normals

    return points, side_idx, plane_p, normals


@torch.jit.script
def local_to_skybox_coordinates(
    coords: torch.Tensor,
    skybox_radius: torch.Tensor
) -> torch.Tensor:
    """Converts local coordinates to plane coordinates in the skybox atlas for the corresponding skybox side.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates in object local space. Shape: ([... , B,] 3+)
        The last dimension is the coordinates in local space. (x, y, z, [...])
    skybox_radius : torch.Tensor
        The radius of the skybox.

    Returns
    -------
    torch.Tensor
        Coordinates in skybox atlas space. Shape: ([... , B,] 2)
        
    Face storage order (uv coords):
    +X: ( 0,   0 ) to (.33, 0.5)
    +Y: (.33,  0 ) to (.66, 0.5)
    +Z: (.66,  0 ) to ( 1,  0.5)
    -X: ( 0,  0.5) to (.33,  1 )
    -Y: (.33, 0.5) to (.66,  1 )
    -Z: (.66, 0.5) to ( 1,   1 )
    """
    # Remove batch dims
    coords, batch_shape = flatten_batch_dims(coords, -2)
    coords_xyz = coords[..., :3]
    unit_coords = coords_xyz / (skybox_radius + 1e-8)

    x = unit_coords[..., 0]
    y = unit_coords[..., 1]
    z = unit_coords[..., 2]

    max_axis = unit_coords.abs().argmax(dim=-1)
    axis_value = unit_coords.gather(-1, max_axis.unsqueeze(-1)).squeeze(-1)
    is_neg = (axis_value < 0).to(torch.long)

    # Per-face local UV in [0, 1] before atlas packing.
    # Axis order per row: X, Y, Z. Rows: + (top), - (bottom).
    u_face = torch.zeros_like(x)
    v_face = torch.zeros_like(x)

    mask_x = max_axis == 0
    mask_y = max_axis == 1
    mask_z = max_axis == 2

    u_face = torch.where(mask_x, 0.5 * (y + 1.0), u_face)
    v_face = torch.where(mask_x, 0.5 * (z + 1.0), v_face)

    u_face = torch.where(mask_y, 0.5 * (x + 1.0), u_face)
    v_face = torch.where(mask_y, 0.5 * (z + 1.0), v_face)

    u_face = torch.where(mask_z, 0.5 * (x + 1.0), u_face)
    v_face = torch.where(mask_z, 0.5 * (y + 1.0), v_face)

    u_face = torch.clamp(u_face, 0.0, 1.0)
    v_face = torch.clamp(v_face, 0.0, 1.0)

    # Pack 6 faces into a 3x2 atlas with face order:
    # +X,+Y,+Z on top row and -X,-Y,-Z on bottom row.
    atlas_u = (u_face + max_axis.to(u_face.dtype)) / 3.0
    atlas_v = (v_face + is_neg.to(v_face.dtype)) / 2.0
    res = torch.stack([atlas_u, atlas_v], dim=-1)

    # Restore batch dims
    return unflatten_batch_dims(res, batch_shape)


def atlas_to_face_coordinates(atlas_uv: torch.Tensor) -> torch.Tensor:
    """Converts 2D skybox atlas coordinates to 3D per-face coordinates.

    This reverses the atlas packing done by local_to_skybox_coordinates,
    producing per-face UV coordinates plus a normalized face ID.  Feeding
    these 3 values into a 3-input-dim hash grid encoding keeps each cube
    face in its own region of the encoding space, preventing cross-face
    hash collisions that cause blurriness.

    Parameters
    ----------
    atlas_uv : torch.Tensor
        Atlas UV coordinates in [0, 1]². Shape: (..., 2)

    Returns
    -------
    torch.Tensor
        Per-face coordinates in [0, 1]³. Shape: (..., 3)
        [face_u, face_v, face_id / 5]
    """
    u = atlas_uv[..., 0]
    v = atlas_uv[..., 1]

    face_col = (u * 3.0).floor().clamp(0, 2)
    face_row = (v * 2.0).floor().clamp(0, 1)

    face_u = (u * 3.0 - face_col).clamp(0.0, 1.0)
    face_v = (v * 2.0 - face_row).clamp(0.0, 1.0)
    face_id_norm = (face_col + 3.0 * face_row) / 5.0

    return torch.stack([face_u, face_v, face_id_norm], dim=-1)


class BackgroundSkyboxSceneNode3D(BackgroundImagePlaneSceneNode3D):
    """Skybox background using an image atlas for the 6 sides.
    
    Face storage order (uv coords):
    +X: ( 0,   0 ) to (.33, 0.5)
    +Y: (.33,  0 ) to (.66, 0.5)
    +Z: (.66,  0 ) to ( 1,  0.5)
    -X: ( 0,  0.5) to (.33,  1 )
    -Y: (.33, 0.5) to (.66,  1 )
    -Z: (.66, 0.5) to ( 1,   1 )
    """

    skybox_radius: torch.Tensor
    """Radius of the skybox."""

    def __init__(
        self,
        radius: Union[float, torch.Tensor] = 1000.0,
        **kwargs
    ):
        encoding_image_config = kwargs.get('encoding_image_config', 'small')
        proxy_init = kwargs.get('proxy_init', False)

        # During parent init, estimate_rgb_scaling calls self.get_rgb().
        # Keep skybox-specific 3D sampling disabled until we replace the
        # parent's temporary 2D encoding with the skybox 3D encoding.
        self._skybox_encoding_ready = False

        super().__init__(
            **kwargs
        )

        # Skybox radius
        self.register_buffer("skybox_radius", tensorify(radius, dtype=self.dtype))

        # Replace the parent's 2D image encoding with a 3D one
        # (face_u, face_v, face_id_normalized).  This prevents the hash
        # grid from bleeding across skybox face boundaries.
        enc_config = EncodingConfig.parse(encoding_image_config).to_dict()
        enc_config["log2_hashmap_size"] = enc_config.get("log2_hashmap_size", 13) + 3
        self.encoding_image = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=enc_config,
            dtype=self.network_dtype,
        )
        self._skybox_encoding_ready = True

        # Re-estimate the RGB normalization for the new encoding
        if self.rgb_rescaling and not proxy_init:
            self.rgb_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.estimate_rgb_scaling(self.rgb_normalization)

    def init_initial_rgb(
        self,
        dtype: torch.dtype,
        initial_rgb: Optional[torch.Tensor] = None,
        initial_rgb_variance: Optional[torch.Tensor] = None,
        mean_rgb_variance: Optional[torch.Tensor] = None,
    ):
        super().init_initial_rgb(
            dtype, initial_rgb,
            initial_rgb_variance=initial_rgb_variance,
            mean_rgb_variance=mean_rgb_variance,
        )
        # Split the 3×2 atlas into 6 separate face textures so that
        # get_initial_rgb can sample each face independently, avoiding
        # bilinear bleed across face boundaries in grid_sample.
        if initial_rgb is not None and initial_rgb.dim() == 3:
            C, H, W = initial_rgb.shape
            h, w = H // 2, W // 3
            faces = []
            for row in range(2):
                for col in range(3):
                    faces.append(
                        initial_rgb[:, row * h:(row + 1) * h,
                                    col * w:(col + 1) * w].clone()
                    )
            self.register_buffer(
                "initial_rgb_faces", torch.stack(faces))  # (6, C, h, w)
        else:
            self.register_buffer(
                "initial_rgb_faces",
                torch.zeros(6, 3, 1, 1, dtype=dtype))

    def get_initial_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        """Per-face initial-RGB sampling without cross-face bleeding.

        Parameters
        ----------
        uv : torch.Tensor
            Atlas UV coordinates. Shape (B, 2), range [-0.5, 0.5].

        Returns
        -------
        torch.Tensor
            RGB values. Shape (B, 3).
        """
        atlas_uv = uv + 0.5  # [0, 1]²

        face_col = (atlas_uv[..., 0] * 3.0).floor().clamp(0, 2).long()
        face_row = (atlas_uv[..., 1] * 2.0).floor().clamp(0, 1).long()
        face_id = face_col + 3 * face_row  # 0-5

        face_u = (atlas_uv[..., 0] * 3.0 - face_col.float()).clamp(0.0, 1.0)
        face_v = (atlas_uv[..., 1] * 2.0 - face_row.float()).clamp(0.0, 1.0)

        # grid_sample expects [-1, 1]
        grid = torch.stack([face_u * 2.0 - 1.0,
                            face_v * 2.0 - 1.0], dim=-1)  # (B, 2)

        out = torch.zeros(uv.shape[0], 3, device=uv.device, dtype=self.dtype)
        for fid in range(6):
            mask = face_id == fid
            if not mask.any():
                continue
            face_grid = grid[mask][None, None, ...]  # (1, 1, N, 2)
            sampled = F.grid_sample(
                self.initial_rgb_faces[fid:fid + 1],  # (1, 3, h, w)
                face_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners,
            )[0, :, 0, :].permute(1, 0)  # (N, 3)
            out[mask] = sampled
        return out

    def get_rgb(self, uv: torch.Tensor, sin_epoch: torch.Tensor) -> torch.Tensor:
        """Get RGB via 3D face-coordinate encoding.

        Parameters
        ----------
        uv : torch.Tensor
            Atlas UV coordinates. Shape (B, 2), range [-0.5, 0.5].
        sin_epoch : torch.Tensor
            Training progress marker.

        Returns
        -------
        torch.Tensor
            Network RGB prediction. Shape (B, 3).
        """
        if not getattr(self, "_skybox_encoding_ready", False):
            return super().get_rgb(uv, sin_epoch)

        atlas_uv = uv + 0.5  # [0, 1]²
        face_coords = atlas_to_face_coordinates(atlas_uv)  # [0, 1]³

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            enc = self.encoding_image(face_coords)
            if self.coarse_to_fine_color:
                enc = utils.mask(enc, sin_epoch)
            network_rgb = self.network_image(enc)

        network_rgb = network_rgb.to(dtype=self.dtype)
        if self.rgb_rescaling and self.rgb_normalization.fitted:
            network_rgb = self.rgb_normalization(network_rgb)
        return network_rgb

    @classmethod
    def _project_color_to_skybox_atlas(
        cls,
        camera: TimedCameraSceneNode3D,
        images: torch.Tensor,
        background_mask: torch.Tensor,
        times: torch.Tensor,
        color_mask_resolution: torch.Tensor,
        dtype: torch.dtype,
        color_mask_smoothing: bool = True,
    ) -> torch.Tensor:
        """Builds an initial skybox atlas by splatting unmasked frame colors into atlas bins."""
        h_atlas = int(color_mask_resolution[0].item())
        w_atlas = int(color_mask_resolution[1].item())

        sum_rgb = torch.zeros((3, h_atlas, w_atlas), dtype=dtype, device=times.device)
        sum_w = torch.zeros((1, h_atlas, w_atlas), dtype=dtype, device=times.device)

        for frame_idx, t in enumerate(times):
            frame_img = images[frame_idx].to(dtype=dtype, device=times.device)
            frame_bg_mask = background_mask[frame_idx, 0].to(device=times.device)

            if not frame_bg_mask.any():
                continue

            y, x = torch.where(frame_bg_mask)
            uv = torch.stack([
                x.to(dtype=dtype) + 0.5,
                y.to(dtype=dtype) + 0.5,
            ], dim=-1)

            _, ray_directions = camera.get_global_rays(
                uv=uv,
                t=t[None],
                uv_includes_time=False,
            )
            ray_directions = ray_directions[:, 0, :]
            ray_directions = ray_directions / (torch.norm(ray_directions, dim=-1, keepdim=True) + 1e-8)

            atlas_uv = local_to_skybox_coordinates(
                ray_directions,
                tensorify(1.0, dtype=dtype, device=ray_directions.device),
            )
            px = torch.clamp((atlas_uv[:, 0] * (w_atlas - 1)).round().long(), 0, w_atlas - 1)
            py = torch.clamp((atlas_uv[:, 1] * (h_atlas - 1)).round().long(), 0, h_atlas - 1)

            colors = frame_img[:, y, x].permute(1, 0)
            atlas_idx = py * w_atlas + px

            sum_w_flat = sum_w.view(-1)
            sum_w_flat.scatter_add_(0, atlas_idx, torch.ones_like(atlas_idx, dtype=dtype))
            for ch in range(3):
                sum_rgb_flat = sum_rgb[ch].view(-1)
                sum_rgb_flat.scatter_add_(0, atlas_idx, colors[:, ch])

        valid = sum_w > 0
        atlas = torch.where(
            valid.expand_as(sum_rgb),
            sum_rgb / sum_w.clamp_min(1e-8),
            torch.zeros_like(sum_rgb),
        )

        if valid.any():
            observed_mean = sum_rgb.view(3, -1).sum(dim=-1) / sum_w.view(-1).sum().clamp_min(1e-8)
        else:
            observed_mean = images.to(dtype=dtype, device=times.device).mean(dim=(0, -2, -1))

        atlas = torch.where(valid.expand_as(atlas), atlas, observed_mean[:, None, None])

        if color_mask_smoothing:
            smoothed = F.avg_pool2d(atlas.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
            atlas = torch.where(valid.expand_as(atlas), atlas, smoothed)

        return atlas.clamp(0.0, 1.0)

    @classmethod
    def for_camera(cls,
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
                   nag_model: Optional["Any"] = None,
                   color_mask_resolution: Optional[VEC_TYPE] = None,
                   color_mask_smoothing: bool = True,
                   num_flow_control_points: Optional[int] = None,
                   config: Any = None,
                   proxy_init: bool = False,
                   **kwargs):
        if color_mask_resolution is None:
            color_mask_resolution = [200, 300]
        if not isinstance(color_mask_resolution, _DEFAULT):
            color_mask_resolution = tensorify(
                color_mask_resolution, dtype=dtype, device=times.device)
        if num_flow_control_points is None:
            if config is not None:
                num_flow_control_points = int(
                    round(len(times) * config.plane_flow_control_points_ratio))
            else:
                num_flow_control_points = 2

        with torch.no_grad():
            if color_mask_resolution == DEFAULT:
                if config is not None and config.use_dataset_color_reprojection:
                    res_x = int(camera._image_resolution[1].item())
                    res_y = int(camera._image_resolution[0].item())
                else:
                    res_x = min(int(camera._image_resolution[1].item()), int(images.shape[3]))
                    res_y = min(int(camera._image_resolution[0].item()), int(images.shape[2]))
                color_mask_resolution = torch.tensor(
                    [res_y, res_x], dtype=torch.int64, device=times.device)

            radius = kwargs.pop("radius", scene_cutoff_distance)
            radius = tensorify(radius, dtype=dtype, device=times.device)

            if proxy_init:
                projected_color_mask = torch.zeros(
                    (3,) + tuple(color_mask_resolution.long().tolist()), dtype=dtype, device=times.device)
            else:
                background_mask = (masks.sum(dim=1, keepdim=True) == 0)
                projected_color_mask = cls._project_color_to_skybox_atlas(
                    camera=camera,
                    images=images,
                    background_mask=background_mask,
                    times=times,
                    color_mask_resolution=color_mask_resolution.long(),
                    dtype=dtype,
                    color_mask_smoothing=color_mask_smoothing,
                )

            # Keep a fixed world orientation and only track camera position over time.
            camera_positions = camera.get_global_position(t=times)[:, :3, 3].detach()
            # construct identity quaternions [0, 0, 0, 1]
            orientation_t = torch.tensor([0, 0, 0, 1], dtype=dtype, device=times.device).expand(len(times), 4)

            args = dict(kwargs)
            args.update(
                is_camera_attached=False,
                plane_scale=tensorify([radius, radius], dtype=dtype, device=times.device),
                translation=camera_positions,
                times=times,
                orientation=orientation_t,
                initial_rgb=projected_color_mask,
                flow_reference_time=0,
                num_flow_control_points=num_flow_control_points,
                network_dtype=config.tinycudann_network_dtype if config is not None else torch.float16,
                radius=radius,
                dtype=dtype,
                name=name,
                index=index,
            )
            args.pop("encoding_view_dependence_config", None)
            args.pop("network_view_dependence_config", None)

            if nag_model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(nag_model, NAGFunctionalModel):
                    args = nag_model.patch_background_args(args)

            plane = cls(**args)

            if world is not None:
                world.add_scene_children(plane)
            else:
                camera.add_scene_children(plane)

            return plane


    def forward(self,
                uv: torch.Tensor,
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                t: torch.Tensor,
                sin_epoch: torch.Tensor,
                uv_in_plane_space: bool = False,
                global_position: Optional[torch.Tensor] = None,
                plane_scale: Optional[torch.Tensor] = None,
                plane_scale_offset: Optional[torch.Tensor] = None,
                right_idx_flow: Optional[torch.Tensor] = None,
                rel_frac_flow: Optional[torch.Tensor] = None,
                next_sin_epoch: Optional[torch.Tensor] = None,
                batch_idx: Optional[int] = None,
                max_batch_idx: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, T = uv.shape[:2]

        if uv.numel() == 0:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype, device=uv.device)
            alpha = torch.zeros(B, T, 1, dtype=self.dtype, device=uv.device)
            return rgb, alpha

        if not uv_in_plane_space:
            if global_position is None:
                global_position = self.get_global_position(t=t)

            if plane_scale is None:
                plane_scale = self.get_plane_scale()

            if plane_scale_offset is None:
                plane_scale_offset = self.get_plane_scale_offset()

            # Convert the global intersection points to local coordinates
            # (B, T, 3) (x, y, z) in skybox local coordinates.
            uv_local = global_to_local(
                global_position, uv, v_include_time=True)[..., :3]

            # Convert local to plane space
            uv_plane = local_to_skybox_coordinates(
                uv_local, self.skybox_radius)  # Go from local to skybox atlas coordinate system

            if self.ray_dependent:
                # Convert ray directions to plane space
                ray_global_target = ray_origins + ray_directions
                ray_local_target = global_to_local(
                    global_position, ray_global_target, v_include_time=True)[..., :3]
                ray_local_origin = global_to_local(
                    global_position, ray_origins, v_include_time=True)[..., :3]
                ray_local_directions = ray_local_target - ray_local_origin
                ray_directions = ray_local_directions
                ray_origins = ray_local_origin
            else:
                ray_directions = None
                ray_origins = None
        else:
            uv_plane = uv[..., :2]

        # Plane_intersection_points are in range [0, 1] for x, y, we need to convert them to [-0.5, 0.5]
        uv_network = uv_plane - 0.5

        return self.compute_rgb_alpha(uv=uv_network,
                                      t=t,
                                      sin_epoch=sin_epoch,
                                      right_idx_flow=right_idx_flow,
                                      rel_frac_flow=rel_frac_flow,
                                      ray_origins=ray_origins,
                                      ray_directions=ray_directions,
                                      context=context)
        

    def _plot_plane(self,
                    ax: Axes,
                    t: Optional[torch.Tensor] = None,
                    edgecolor: str = 'black',
                    alpha: float = 1.,
                    line_width: float = 1.0,
                    facecolor: Optional[Any] = None,
                    **kwargs):
        """Plot the skybox as 6 faces.
        
        Note: This is a simplified visualization. For better visualization,
        consider plotting each face separately.
        """
        # For now, plot a simple cube at the skybox position
        # This is a placeholder - you may want to implement proper 6-face visualization
        if t is None:
            t = torch.tensor(0., dtype=self.dtype, device=self._translation.device)
        
        position = self.get_global_position(t)
        center = position[:3, 3].detach().cpu().numpy()
        radius = self.skybox_radius.item()
        
        # Plot a simple wireframe cube
        from itertools import product
        for s, e in product([0, 1], repeat=2):
            # Draw edges along each axis
            for axis in range(3):
                start = center.copy()
                end = center.copy()
                axes_idx = [i for i in range(3) if i != axis]
                start[axes_idx[0]] += radius * (2*s - 1)
                start[axes_idx[1]] += radius * (2*e - 1)
                end[axes_idx[0]] += radius * (2*s - 1)
                end[axes_idx[1]] += radius * (2*e - 1)
                start[axis] = center[axis] - radius
                end[axis] = center[axis] + radius
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]],
                       color=edgecolor, alpha=alpha, linewidth=line_width)
