"""CPU measurements of checkpoint-derived texture flow, with explicit units."""
import numpy as np
from scipy.stats import spearmanr


def uv_difference(a, b):
    delta = np.asarray(a) - np.asarray(b)
    delta = delta.copy()
    delta[..., 0] = (delta[..., 0] + .5) % 1 - .5
    return delta


def parity(render, saved, mean_limit=.5, p99_limit=2.):
    """Tolerance is in 8-bit channel values, not normalized RGB units."""
    if render.shape != saved.shape:
        raise ValueError(f'Render/image shape mismatch: {render.shape} != {saved.shape}')
    diff = np.abs(render.astype(float) - saved.astype(float))
    result = dict(mean_bytes=float(diff.mean()), p99_bytes=float(np.quantile(diff, .99)),
                  max_bytes=float(diff.max()), fraction_over_one=float((diff > 1).mean()))
    result['passed'] = result['mean_bytes'] <= mean_limit and result['p99_bytes'] <= p99_limit
    return result


def flow_fields(before, after, native, pixel, hit, projected, tile, texel_scale, step=1.):
    """Five stencils: center, x+h, x-h, y+h, y-h; axes (5,N,2).

    Pixel displacements/steps are in saved-image pixels. Native offsets retain
    their raw network representation; effective UVs wrap longitude and clamp
    latitude exactly as the renderer does. Texture texels refer only to the
    fixed initial-RGB lookup, never to the neural network's input coordinates.
    """
    if step <= 0 or before.shape != after.shape or before.shape[0] != 5:
        raise ValueError('Expected matching five-point stencils and positive step')
    valid = hit & projected & np.isfinite(before).all(-1) & np.isfinite(after).all(-1)
    valid &= np.isfinite(native).all(-1) & np.isfinite(pixel).all(-1)
    stencil = valid.all(0)
    effective = uv_difference(after, before)
    # Regular equirectangular lookup is periodic. Visible-UV tile branch
    # continuity is supplied explicitly by the caller in `tile`.
    tile_stencil = (tile & valid).all(0)
    scale = np.asarray(texel_scale)

    def jacobian(a, periodic=False):
        difference = uv_difference if periodic else np.subtract
        return np.stack((difference(a[1], a[2]), difference(a[3], a[4])), -1)/(2*step)

    j0, j1 = jacobian(before, True), jacobian(after, True)
    jf, je, jp = jacobian(native), jacobian(effective, True), jacobian(pixel)
    # I + d(effective_flow)/d(UV) is obtained from the actual flowed lookup,
    # including latitude clamping; do not interpret an invalid chart as zero.
    safe = stencil & np.isfinite(j0).all((1, 2))
    condition = np.full(len(safe), np.inf)
    condition[safe] = np.linalg.cond(j0[safe])
    safe &= (condition < 1e4) & (np.abs(np.linalg.det(j0)) > 1e-14)
    transport = np.full_like(j0, np.nan)
    transport[safe] = j1[safe] @ np.linalg.inv(j0[safe])
    sv = np.full((len(safe), 2), np.nan)
    sv[safe] = np.linalg.svd(transport[safe], compute_uv=False)
    curvature = np.stack((native[1]-2*native[0]+native[2],
                          native[3]-2*native[0]+native[4]), -1)/(step**2)
    determinant = np.full(len(safe), np.nan)
    determinant[safe] = np.linalg.det(transport[safe])
    arrays = dict(native_uv_magnitude=np.linalg.norm(native[0], axis=-1),
                  effective_uv_magnitude=np.linalg.norm(effective[0], axis=-1),
                  initial_texture_texel_magnitude=np.linalg.norm(effective[0]*scale, axis=-1),
                  image_pixel_magnitude=np.linalg.norm(pixel[0], axis=-1),
                  native_uv_gradient=np.linalg.norm(jf, axis=(1, 2)),
                  effective_uv_gradient=np.linalg.norm(je, axis=(1, 2)),
                  initial_texture_texel_gradient=np.linalg.norm(je*scale[None, :, None], axis=(1, 2)),
                  image_pixel_gradient=np.linalg.norm(jp, axis=(1, 2)),
                  native_uv_curvature=np.linalg.norm(curvature, axis=(1, 2)),
                  transport_max=sv[:, 0], transport_min=sv[:, 1],
                  transport_deviation=np.linalg.norm(transport-np.eye(2), axis=(1, 2)),
                  transport_determinant=determinant)
    with np.errstate(divide='ignore', invalid='ignore'):
        arrays['transport_condition'] = np.where(sv[:, 1] > 1e-12, sv[:, 0]/sv[:, 1], np.nan)
    for name, array in arrays.items():
        keep = (safe if name.startswith('transport_') else stencil if ('gradient' in name or 'curvature' in name) else valid[0]).copy()
        if name.startswith('initial_texture_'):
            keep &= tile_stencil if 'gradient' in name else tile[0]
        array[~keep] = np.nan
    flags = dict(valid=valid[0], stencil_valid=stencil, chart_valid=safe,
                 tile_valid=valid[0] & tile[0], tile_stencil_valid=tile_stencil)
    return arrays, flags


def statistics(values, selection, frame_ids):
    """Equal mass per contributing frame; absent observations remain absent."""
    a = np.asarray(values).ravel()
    selected = np.asarray(selection).ravel() & np.isfinite(a)
    ids = np.broadcast_to(np.asarray(frame_ids)[:, None], np.asarray(values).shape).ravel()
    count = int(selected.sum())
    if not count:
        return dict(n=0, frames=0, mean=None, median=None, p90=None)
    v, f = a[selected], ids[selected]
    frames, inv, counts = np.unique(f, return_inverse=True, return_counts=True)
    w = 1/(len(frames)*counts[inv])
    order = np.argsort(v)
    cumulative = np.cumsum(w[order])
    return dict(n=count, frames=len(frames), mean=float(v@w),
                median=float(v[order[min(np.searchsorted(cumulative, .5), len(v)-1)]]),
                p90=float(v[order[min(np.searchsorted(cumulative, .9), len(v)-1)]]))


def association(x, y, selection):
    selected = np.asarray(selection) & np.isfinite(x) & np.isfinite(y)
    if selected.sum() < 10 or np.ptp(x[selected]) == 0 or np.ptp(y[selected]) == 0:
        return None
    # Descriptive only: these are spatially correlated pixels, not independent
    # samples for a significance test.
    return float(spearmanr(x[selected], y[selected]).statistic)
