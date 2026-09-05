"""Texture coordinates independent of an ellipsoid's physical axis ordering."""

import torch

from nag.model.sphere_utils import direction_to_equirect_uv, equirect_uv_to_direction


def surface_to_uv(points, y_flip=False, texture_rotation=None, u_offset=0.0):
    """Map normalized local surface points into the texture's unit sphere."""
    if texture_rotation is not None:
        points = points @ texture_rotation.to(points)
    uv = direction_to_equirect_uv(points, y_flip=y_flip)
    if u_offset != 0.0:
        uv = torch.stack((torch.remainder(uv[..., 0] + u_offset, 1.0), uv[..., 1]), -1)
    return uv


def uv_to_surface(uv, y_flip=False, texture_rotation=None, u_offset=0.0):
    """Inverse of surface_to_uv, before multiplying by physical radii."""
    if u_offset != 0.0:
        uv = torch.stack((torch.remainder(uv[..., 0] - u_offset, 1.0), uv[..., 1]), -1)
    points = equirect_uv_to_direction(uv, y_flip=y_flip)
    if texture_rotation is not None:
        points = points @ texture_rotation.to(points).T
    return points


def world_y_texture_rotation(rotation, radii):
    """Return a right-handed texture basis with physical world-Y poles.

    Physical directions must pass through inverse rotation AND inverse radii
    before defining the texture basis in normalized local coordinates.
    World +Y is a vertical axis here; it need not mean physical 'up'.
    """
    axes = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    vertical = (axes[1] @ rotation) / radii
    vertical = vertical / vertical.norm()
    forward = (axes[2] @ rotation) / radii
    forward = forward - torch.dot(forward, vertical) * vertical
    if forward.norm() < 1e-6:
        forward = (axes[0] @ rotation) / radii
        forward = forward - torch.dot(forward, vertical) * vertical
    forward = forward / forward.norm()
    right = torch.linalg.cross(vertical, forward)
    return torch.stack((right, vertical, forward), dim=1)


def choose_seam_offset(u, bins=360):
    """Place longitude seam in the largest empty arc, or least occupied arc.

    Deterministic histogram selection avoids treating tiny gaps between sampled
    rays as unobserved regions when the sequence covers a full panorama.
    """
    indices = (torch.remainder(u.flatten(), 1.0) * bins).long().clamp_max(bins - 1)
    counts = torch.bincount(indices, minlength=bins).cpu().tolist()
    if not indices.numel():
        raise ValueError("Cannot select a texture seam without valid surface observations.")
    best_start, best_length, start, length = 0, 0, 0, 0
    for i in range(2 * bins):
        if counts[i % bins] == 0:
            if length == 0:
                start = i
            length += 1
            if length > best_length and length <= bins:
                best_start, best_length = start, length
        else:
            length = 0
    if best_length:
        seam = ((best_start + best_length / 2) / bins) % 1.0
        method = "largest_empty_sampled_arc"
    else:
        occupancy = [sum(counts[(i + j) % bins] for j in range(-2, 3)) for i in range(bins)]
        seam = (min(range(bins), key=occupancy.__getitem__) + 0.5) / bins
        method = "least_occupied_sampled_arc"
    return (-seam) % 1.0, {
        "seam_selection": method,
        "seam_longitude_before_offset": seam,
        "empty_arc_degrees": best_length * 360.0 / bins,
        "histogram_bins": bins,
    }


def uv_diagnostics(uv):
    """Report sampled proximity to the coordinate singularities, not fit loss."""
    u, v = uv.unbind(-1)
    return {
        "near_pole_10deg_fraction": float(((v < 10 / 180) | (v > 170 / 180)).float().mean()),
        "near_seam_5deg_fraction": float((torch.minimum(u, 1 - u) < 5 / 360).float().mean()),
    }
