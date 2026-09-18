"""Offline texture sampling measurements in actual saved texture texels."""
import numpy as np

from diagnostics.parallax_core import intersect, surface_coordinates, surface_uv


def texture_coordinates(pixels, ids, poses, intrinsics, geometry, mapping):
    local = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsics).T
    directions = np.einsum('nij,nj->ni', poses[ids, :3, :3], local)
    points, _, valid = intersect(poses[ids, :3, 3], directions, geometry)
    uv = surface_uv(surface_coordinates(points, geometry), mapping)
    layout = mapping.get('visible_uv', {})
    size = mapping['resolution_hw']
    if layout.get('enabled'):
        size = layout['resolution']
        uv = np.column_stack((((uv[:, 0] - layout['u_start']) % 1) / layout['u_span'],
                              (uv[:, 1] - layout['v_start']) / layout['v_span']))
        valid &= ((uv >= 0) & (uv <= 1)).all(axis=1)
    return uv * np.asarray(size)[::-1], valid & np.isfinite(uv).all(axis=1)


def sampling_metrics(pixels, ids, poses, intrinsics, geometry, mapping, step=0.5):
    """Central differences of image pixels -> texture texels (seam aware).

    Both singular values matter: condition number alone misses uniform texture
    collapse. Invalid/tile-boundary samples remain NaN and are counted downstream.
    """
    _, valid = texture_coordinates(pixels, ids, poses, intrinsics, geometry, mapping)
    columns = []
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus, vp = texture_coordinates(pixels + offset, ids, poses, intrinsics, geometry, mapping)
        minus, vm = texture_coordinates(pixels - offset, ids, poses, intrinsics, geometry, mapping)
        delta = plus - minus
        if not mapping.get('visible_uv', {}).get('enabled'):
            width = mapping['resolution_hw'][1]
            delta[:, 0] = (delta[:, 0] + width / 2) % width - width / 2
        columns.append(delta / (2 * step))
        valid &= vp & vm
    jacobian = np.stack(columns, axis=-1)
    valid &= np.isfinite(jacobian).all(axis=(1, 2))
    singular = np.full((len(pixels), 2), np.nan)
    singular[valid] = np.linalg.svd(jacobian[valid], compute_uv=False)
    large, small = singular.T
    return {
        'texels_per_pixel_min': small,
        'texels_per_pixel_max': large,
        'texel_area_per_pixel': large * small,
        'sampling_condition': large / np.maximum(small, 1e-12),
        'valid': valid,
    }


def balanced_weights(ids, pixels, size):
    """Equal frame mass, equal occupied 12x8 image-cell mass within a frame."""
    cells = np.clip((pixels / np.asarray(size) * [12, 8]).astype(int), [0, 0], [11, 7])
    keys = ids * 96 + cells[:, 1] * 12 + cells[:, 0]
    weights = np.zeros(len(ids))
    for frame in np.unique(ids):
        selected = ids == frame
        _, inverse, counts = np.unique(keys[selected], return_inverse=True, return_counts=True)
        weights[selected] = 1 / (len(counts) * counts[inverse])
    return weights / weights.sum() if len(weights) else weights


def statistics(values, weights):
    valid = np.isfinite(values)
    result = {'count': len(values), 'valid_fraction': float(valid.mean()) if len(values) else None}
    if not valid.any():
        return {**result, **dict.fromkeys(('mean', 'balanced_mean', 'p10', 'median', 'p90', 'p95'))}
    selected = values[valid]
    return {**result, 'mean': float(selected.mean()),
            'balanced_mean': float(np.average(selected, weights=weights[valid])),
            **dict(zip(('p10', 'median', 'p90', 'p95'), np.quantile(selected, [.1, .5, .9, .95])))}
