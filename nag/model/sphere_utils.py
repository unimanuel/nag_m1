"""Utilities for spherical background parameterizations.

These helpers are intentionally lightweight (pure torch) so they can be used
inside forward passes without pulling in extra dependencies.

Conventions
-----------
We use an equirectangular (lat/long) mapping.

Given a unit direction vector d = (x, y, z) in *local sphere coordinates*:

    lon = atan2(x, z)        in [-pi, pi]
    lat = asin(clamp(y))     in [-pi/2, pi/2]

We map to UV in [0, 1]^2 as:

    u = (lon + pi) / (2*pi)
    v = (pi/2 - lat) / pi

So v=0 is "north" (+y) and v=1 is "south" (-y). This matches a common HDRI
environment-map convention.
"""

from __future__ import annotations

import torch


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def direction_to_equirect_uv(
    direction: torch.Tensor,
    y_flip: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert a direction vector to equirectangular UV.

    Parameters
    ----------
    direction : torch.Tensor
        (..., 3) direction vectors (need not be normalized).
    y_flip : bool
        If True, flip v -> 1-v.
    eps : float
        Numerical stability.

    Returns
    -------
    torch.Tensor
        (..., 2) UV in [0, 1].
    """
    d = normalize(direction, eps=eps)
    x, y, z = d.unbind(dim=-1)
    lon = torch.atan2(x, z)  # [-pi, pi]
    lat = torch.asin(y.clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # [-pi/2, pi/2]
    u = (lon + torch.pi) / (2.0 * torch.pi)
    v = (0.5 * torch.pi - lat) / torch.pi
    if y_flip:
        v = 1.0 - v
    return torch.stack([u, v], dim=-1)


def equirect_uv_to_direction(
    uv: torch.Tensor,
    y_flip: bool = False,
) -> torch.Tensor:
    """Convert equirectangular UV in [0, 1] to a unit direction vector."""
    u, v = uv.unbind(dim=-1)
    if y_flip:
        v = 1.0 - v
    lon = u * (2.0 * torch.pi) - torch.pi
    lat = 0.5 * torch.pi - v * torch.pi
    cos_lat = torch.cos(lat)
    x = torch.sin(lon) * cos_lat
    y = torch.sin(lat)
    z = torch.cos(lon) * cos_lat
    return torch.stack([x, y, z], dim=-1)


def make_tangent_basis(n: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create an orthonormal tangent basis (t1, t2, n) for a unit normal n.

    Parameters
    ----------
    n : torch.Tensor
        (..., 3) normals (need not be normalized).

    Returns
    -------
    (t1, t2, n_unit) : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Each (..., 3). Columns of a right-handed basis.
    """
    n_unit = normalize(n, eps=eps)

    # Pick an "up" that is not parallel to n.
    up = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    up2 = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)

    # If |dot(n, up)| is too close to 1, swap.
    use_up2 = (torch.abs((n_unit * up).sum(dim=-1, keepdim=True)) > 0.99)
    up_vec = torch.where(use_up2, up2, up)

    t1 = torch.cross(up_vec.expand_as(n_unit), n_unit, dim=-1)
    t1 = normalize(t1, eps=eps)
    t2 = torch.cross(n_unit, t1, dim=-1)
    t2 = normalize(t2, eps=eps)
    return t1, t2, n_unit


def to_tangent_frame(v: torch.Tensor, n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Express vectors v in the tangent frame defined by normal n.

    The returned coordinates are [dot(v, t1), dot(v, t2), dot(v, n)].
    """
    t1, t2, n_unit = make_tangent_basis(n, eps=eps)
    x = (v * t1).sum(dim=-1)
    y = (v * t2).sum(dim=-1)
    z = (v * n_unit).sum(dim=-1)
    return torch.stack([x, y, z], dim=-1)


def ray_sphere_intersection(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with a sphere.

    Parameters
    ----------
    ray_origins : torch.Tensor
        (..., 3)
    ray_directions : torch.Tensor
        (..., 3) (need not be normalized)
    center : torch.Tensor
        (..., 3) or (3,) broadcastable to ray_origins
    radius : torch.Tensor
        Scalar or broadcastable to (...,)

    Returns
    -------
    (points, hit, t) : tuple
        points (..., 3) intersection points (undefined where hit==False)
        hit (...,) boolean mask
        t (...,) intersection distance along ray
    """
    d = normalize(ray_directions, eps=eps)
    o = ray_origins
    c = center
    r = radius

    oc = o - c
    b = (d * oc).sum(dim=-1)  # (...,)
    c_term = (oc * oc).sum(dim=-1) - r * r
    disc = b * b - c_term
    hit = disc >= 0

    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    # For an origin inside the sphere, the exiting intersection is -b + sqrt(disc)
    t_out = -b + sqrt_disc
    # If outside, this still gives the nearer positive root in most cases.
    t = t_out
    hit = hit & (t > eps)

    points = o + t.unsqueeze(-1) * d
    return points, hit, t
