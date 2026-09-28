"""Sparse, geometry-only texture correspondence fields; no learned-flow queries."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from diagnostics.parallax_core import intersect, surface_coordinates, surface_uv


@dataclass(frozen=True)
class NeighborhoodConfig:
    radius_pixels: float = 80.
    neighbors: int = 16
    min_support: int = 6
    depth_absolute: float = 1.
    depth_relative: float = .10
    max_chart_condition: float = 1000.


def difference(a, b, period=None):
    """Signed a-b, using a local periodic longitude chart where appropriate."""
    out = np.asarray(a) - np.asarray(b)
    if period is not None:
        out = out.copy()
        out[..., 0] = (out[..., 0] + period / 2) % period - period / 2
    return out


def correspondence_fields(pairs, geometry, mapping):
    p = pairs['points']
    ci = pairs['poses'][pairs['source'], :3, 3]
    cj = pairs['poses'][pairs['target'], :3, 3]
    qi, ti, vi = intersect(ci, p-ci, geometry)
    qj, tj, vj = intersect(cj, p-cj, geometry)
    a = surface_uv(surface_coordinates(qi, geometry), mapping)
    b = surface_uv(surface_coordinates(qj, geometry), mapping)
    valid = vi & vj & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    raw = b-a
    delta = difference(b, a, 1.)
    layout = mapping.get('visible_uv', {})
    visible = layout.get('enabled', False)
    if visible:
        def tile(uv):
            return np.column_stack((((uv[:, 0]-layout['u_start']) % 1)/layout['u_span'],
                                    (uv[:, 1]-layout['v_start'])/layout['v_span']))
        ta, tb = tile(a), tile(b)
        covered = ((ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)).all(axis=1)
        size = np.asarray(layout['resolution'])[::-1]
        tex_delta = (tb-ta) * size
    else:
        ta, tb = a, b
        size = np.asarray(mapping['resolution_hw'])[::-1]
        covered = np.ones(len(p), dtype=bool)
        tex_delta = delta * size
    tex_valid = valid & covered
    delta[~valid] = np.nan
    tex_delta[~tex_valid] = np.nan
    return dict(uv_source=a, uv_target=b, delta_uv=delta,
                texture_source=ta*size, texture_target=tb*size, delta_texels=tex_delta,
                texture_period=None if visible else float(size[0]), valid=valid,
                tile_covered=tex_valid, seam_crossing=valid & (np.abs(raw[:, 0]) > .5),
                near_seam=valid & ((np.minimum(a[:, 0], 1-a[:, 0]) < 5/360) |
                                   (np.minimum(b[:, 0], 1-b[:, 0]) < 5/360)),
                depth_relative_error=(np.abs(ti/np.linalg.norm(p-ci, axis=1)-1) +
                                      np.abs(tj/np.linalg.norm(p-cj, axis=1)-1))/2)


def pair_weights(source, target, pixels, size):
    """Equal source-frame mass, then target-pair mass, then occupied-cell mass."""
    weights = np.zeros(len(source))
    cells = np.clip((pixels / np.asarray(size) * [12, 8]).astype(int), [0, 0], [11, 7])
    keys = cells[:, 1]*12 + cells[:, 0]
    frames = np.unique(source)
    for i in frames:
        targets = np.unique(target[source == i])
        for j in targets:
            take = (source == i) & (target == j)
            _, inverse, counts = np.unique(keys[take], return_inverse=True, return_counts=True)
            weights[take] = 1 / (len(frames)*len(targets)*len(counts)*counts[inverse])
    return weights


def neighborhoods(pixels, source, target, source_depth, target_depth, config):
    """Geometry-independent neighbors, within a camera pair and depth layer.

    These are conservative depth gates, not proof of a shared semantic surface.
    No interpolation across holes, and no mixing of different camera pairs.
    """
    count = len(pixels)
    indices = np.full((count, config.neighbors), -1, dtype=np.int32)
    distances = np.full((count, config.neighbors), np.inf)
    for i, j in np.unique(np.column_stack((source, target)), axis=0):
        rows = np.flatnonzero((source == i) & (target == j))
        tree = cKDTree(pixels[rows])
        dist, local = tree.query(pixels[rows], k=config.neighbors,
                                 distance_upper_bound=config.radius_pixels)
        finite = np.isfinite(dist)
        chosen = rows[np.minimum(local, len(rows)-1)]
        for depth in (source_depth, target_depth):
            tolerance = config.depth_absolute + config.depth_relative*np.minimum(
                depth[rows, None], depth[chosen])
            finite &= np.abs(depth[chosen]-depth[rows, None]) <= tolerance
        indices[rows] = np.where(finite, chosen, -1)
        distances[rows] = np.where(finite, dist, np.inf)
    return indices, distances


def local_field(pixels, coordinates, delta, indices, distances, valid, config, period=None):
    """Robust local affine field and chart transport on sparse supported samples.

    Fit delta(p)=A p+b and source_chart(p)=J p+c, then form I+A J^-1.
    This is relative texture correspondence, not the learned network Jacobian.
    Longitude is unwrapped locally. Under-supported/ill-conditioned fits stay NaN.
    """
    count = len(pixels)
    names = ('gradient', 'affine_residual', 'transport_max', 'transport_min',
             'transport_condition', 'transport_deviation', 'transport_determinant')
    result = {name: np.full(count, np.nan) for name in names}
    result['support_count'] = np.zeros(count, dtype=np.int16)
    result['image_fit_valid'] = np.zeros(count, dtype=bool)
    result['chart_fit_valid'] = np.zeros(count, dtype=bool)
    for first in range(0, count, 2048):
        rows = np.arange(first, min(first+2048, count))
        idx = indices[rows]
        chosen = np.maximum(idx, 0)
        allowed = (idx >= 0) & valid[chosen] & valid[rows, None]
        xy = (pixels[chosen]-pixels[rows, None]) / config.radius_pixels
        offsets = difference(coordinates[chosen], coordinates[rows, None], period)
        dd = difference(delta[chosen], delta[rows, None], period)
        # Avoid ambiguous antipodal branches in either the chart or displacement.
        if period is not None:
            allowed &= (np.abs(offsets[..., 0]) < .45*period) & (np.abs(dd[..., 0]) < .45*period)
        allowed &= np.isfinite(offsets).all(axis=-1) & np.isfinite(dd).all(axis=-1)
        x = np.concatenate((np.ones((*xy.shape[:2], 1)), xy), axis=-1)
        y = np.nan_to_num(np.concatenate((dd, offsets), axis=-1))
        base = allowed * np.exp(-.5*(distances[rows]/config.radius_pixels)**2)
        support = allowed.sum(axis=1)
        result['support_count'][rows] = support
        moment = np.einsum('bki,bk,bkj->bij', x, base, x)
        eig = np.linalg.eigvalsh(moment)
        image_good = (support >= config.min_support) & (eig[:, 0] > 1e-4*np.maximum(eig[:, -1], 1e-12))
        weight = base.copy()
        for iteration in range(3):
            moment = np.einsum('bki,bk,bkj->bij', x, weight, x)
            rhs = np.einsum('bki,bk,bkj->bij', x, weight, y)
            beta = np.linalg.solve(moment + np.eye(3)[None]*1e-12, rhs)
            residual = y - np.einsum('bki,bij->bkj', x, beta)
            error = np.linalg.norm(residual[..., :2], axis=-1)
            masked = np.ma.array(error, mask=~allowed)
            scale = np.maximum(np.ma.median(masked, axis=1).filled(0)*1.5, 1e-10)
            if iteration < 2:
                weight = base*np.minimum(1, scale[:, None]/np.maximum(error, 1e-12))
        # Robust weights can remove the samples that made the initial image
        # design two-dimensional. Recheck the actual final solve as well.
        final_eig = np.linalg.eigvalsh(moment)
        image_good &= final_eig[:, 0] > 1e-4*np.maximum(final_eig[:, -1], 1e-12)
        a = beta[:, 1:, :2].transpose(0, 2, 1)/config.radius_pixels
        j = beta[:, 1:, 2:].transpose(0, 2, 1)/config.radius_pixels
        result['image_fit_valid'][rows] = image_good
        result['gradient'][rows[image_good]] = np.linalg.norm(a[image_good], axis=(1, 2))
        rms = np.sqrt(np.sum(weight*error**2, axis=1)/np.maximum(weight.sum(axis=1), 1e-12))
        result['affine_residual'][rows[image_good]] = rms[image_good]
        singular = np.linalg.svd(j, compute_uv=False)
        chart_good = image_good & (singular[:, 1] > 1e-10) & (
            singular[:, 0]/np.maximum(singular[:, 1], 1e-20) < config.max_chart_condition)
        result['chart_fit_valid'][rows] = chart_good
        if chart_good.any():
            transport = np.eye(2)[None] + a[chart_good] @ np.linalg.inv(j[chart_good])
            stretch = np.linalg.svd(transport, compute_uv=False)
            dest = rows[chart_good]
            result['transport_max'][dest] = stretch[:, 0]
            result['transport_min'][dest] = stretch[:, 1]
            result['transport_condition'][dest] = stretch[:, 0]/np.maximum(stretch[:, 1], 1e-12)
            result['transport_deviation'][dest] = np.linalg.norm(transport-np.eye(2), axis=(1, 2))
            result['transport_determinant'][dest] = np.linalg.det(transport)
    return result


def common_masks(records):
    """Identical pair/sampling/spatial subsets; missing coverage never counts as zero."""
    return {key: np.logical_and.reduce([record[key] for record in records])
            for key in ('pair_valid', 'texture_valid', 'sampling_valid', 'uv_image_fit_valid',
                        'uv_chart_fit_valid', 'texel_image_fit_valid', 'texel_chart_fit_valid')}
