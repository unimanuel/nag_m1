"""CPU geometry and image-region primitives; no NAG model construction required."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15)


def angle_degrees(a, b):
    a, b = unit(a), unit(b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                               np.sum(a * b, axis=-1)))


def intrinsics(camera, size):
    """Match PinholeCameraConfig.get_intrinsics, including width-based fy."""
    w, h = size
    original = camera['resolution']
    if isinstance(original, dict):
        original = original['value']
    pp = camera.get('principal_point')
    cx, cy = (w / 2, h / 2) if pp is None else (pp[0] * w / original[0], pp[1] * h / original[1])
    f = camera['focal_length'] * w
    return np.array([[f, camera.get('skew', 0), cx], [0, f, cy], [0, 0, 1.]])


def project(points, pose, k):
    """Camera-to-world pose, NAG camera +z forward, +y down; row arrays."""
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    homogeneous = local @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, local[:, 2]


def inside(uv, z, size):
    w, h = size
    return (np.isfinite(uv).all(axis=1) & (z > 1e-6) & (uv[:, 0] >= 0)
            & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))


def mask_at(mask, uv):
    good = inside(uv, np.ones(len(uv)), (mask.shape[1], mask.shape[0]))
    result = np.zeros(len(uv), dtype=bool)
    px = np.floor(uv[good]).astype(int)
    result[good] = mask[px[:, 1], px[:, 0]]
    return result


def intersect(origins, directions, geometry):
    """Nearest positive hit (exit for interior origins), in world metres."""
    directions = unit(directions)
    c, r, rot = (np.asarray(geometry[k]) for k in ('center', 'radii', 'rotation'))
    o = ((origins - c) @ rot) / r
    d = (directions @ rot) / r
    aa, bb, cc = np.sum(d*d, axis=-1), 2*np.sum(o*d, axis=-1), np.sum(o*o, axis=-1)-1
    disc = bb*bb - 4*aa*cc
    with np.errstate(invalid='ignore', divide='ignore'):
        near = (-bb-np.sqrt(np.maximum(disc, 0))) / (2*aa)
        far = (-bb+np.sqrt(np.maximum(disc, 0))) / (2*aa)
    t = np.where(near > 1e-8, near, far)
    valid = (disc >= 0) & (aa > 0) & (t > 1e-8) & np.isfinite(t)
    t = np.where(valid, t, np.nan)
    return origins + t[..., None] * directions, t, valid


def surface_coordinates(points, geometry):
    return unit(((points - geometry['center']) @ np.asarray(geometry['rotation'])) / geometry['radii'])


def surface_uv(s, mapping):
    d = unit(s @ np.asarray(mapping.get('rotation', np.eye(3))))
    u = (np.arctan2(d[:, 0], d[:, 2]) + np.pi) / (2*np.pi)
    v = (np.pi/2 - np.arcsin(np.clip(d[:, 1], -1+1e-6, 1-1e-6))) / np.pi
    if mapping.get('y_flip', False):
        v = 1-v
    return np.stack(((u + mapping.get('u_offset', 0)) % 1, v), axis=1)


def uv_errors(a, b, mapping):
    raw = np.abs(a-b)
    wrapped = raw.copy()
    wrapped[:, 0] = np.minimum(raw[:, 0], 1-raw[:, 0])
    layout = mapping.get('visible_uv', {})
    size = mapping['resolution_hw']
    coverage = np.ones(len(a), dtype=bool)
    if layout.get('enabled'):
        size = layout['resolution']
        def tile(uv):
            return np.stack((((uv[:, 0]-layout['u_start']) % 1)/layout['u_span'],
                             (uv[:, 1]-layout['v_start'])/layout['v_span']), axis=1)
        ta, tb = tile(a), tile(b)
        coverage = ((ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)).all(axis=1)
        actual = np.abs(ta-tb)
    else:
        actual = wrapped
    texels = np.linalg.norm(actual * np.array(size)[::-1], axis=1)
    texels[~coverage] = np.nan
    return np.linalg.norm(wrapped, axis=1), raw[:, 0] > .5, texels, coverage


class DepthEvidence:
    """Sparse z buffer, then conservative nearest-pixel depth evidence.

    States: 0 unknown, 1 supported, 2 occluded, 3 inconsistent. The closest
    point is chosen within a bounded pixel radius; no dense hole filling.
    """
    def __init__(self, uv, z, size):
        valid = inside(uv, z, size)
        uv, z = uv[valid], z[valid]
        pixels = np.floor(uv).astype(int)
        keys = pixels[:, 1] * size[0] + pixels[:, 0]
        order = np.lexsort((z, keys))
        _, first = np.unique(keys[order], return_index=True)
        chosen = order[first]
        self.uv, self.z = uv[chosen], z[chosen]
        self.tree = cKDTree(self.uv)

    def classify(self, uv, depth, radius=2., absolute=.5, relative=.03):
        result = np.zeros(len(uv), dtype=np.uint8)
        if not len(self.z):
            return result
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
        distance, index = self.tree.query(uv[finite], distance_upper_bound=radius)
        found = np.isfinite(distance)
        rows = np.flatnonzero(finite)[found]
        observed = self.z[index[found]]
        tolerance = absolute + relative * depth[rows]
        residual = depth[rows]-observed
        result[rows] = np.where(np.abs(residual) <= tolerance, 1,
                                np.where(residual > tolerance, 2, 3))
        return result


def balanced_indices(uv, size, count, rng):
    """Round-robin sample across 12x8 image cells, without depth weighting."""
    if len(uv) <= count:
        return np.arange(len(uv))
    cell = np.clip((uv / np.array(size) * [12, 8]).astype(int), [0, 0], [11, 7])
    keys = cell[:, 1]*12+cell[:, 0]
    buckets = [rng.permutation(np.flatnonzero(keys == k)) for k in np.unique(keys)]
    choices = []
    for level in range(max(map(len, buckets))):
        for b in rng.permutation(len(buckets)):
            if level < len(buckets[b]):
                choices.append(buckets[b][level])
                if len(choices) == count:
                    return np.sort(choices)
    return np.sort(choices)


def regions(foreground, width=3):
    if width < 1:
        raise ValueError('Boundary width must be positive')
    structure = np.ones((3, 3), dtype=bool)
    outer = binary_dilation(foreground, structure=structure, iterations=width)
    inner = binary_erosion(foreground, structure=structure, iterations=width, border_value=1)
    return {'background': ~outer, 'foreground': inner, 'boundary': outer & ~inner}


def image_statistics(render, target, selections):
    difference = (render.astype(float)-target.astype(float))/255
    mae, mse = np.abs(difference).mean(axis=2), (difference**2).mean(axis=2)
    rows = {}
    for name, mask in selections.items():
        count = int(mask.sum())
        rows[name] = {'pixels': count, 'mae': float(mae[mask].mean()) if count else None,
                      'mse': float(mse[mask].mean()) if count else None}
    return rows, mae


def signed_error_rgb(delta, cap):
    """Blue improvement, white equality, red degradation; fixed RGB-MAE cap."""
    strength = np.clip(np.abs(delta)/cap, 0, 1)[..., None]
    color = np.where((delta < 0)[..., None], [30, 90, 220], [220, 40, 30])
    return np.rint(255*(1-strength)+color*strength).astype(np.uint8)
