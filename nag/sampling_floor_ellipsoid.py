"""Refine original reprojection fits with a sphere-relative texture sampling floor."""
import copy
from dataclasses import asdict
import numpy as np
from scipy.optimize import minimize

from diagnostics.parallax_core import inside
from diagnostics.parallax_evaluate import batched_project
from reprojection_ellipsoid import (ReprojectionFitConfig, _array_geometry,
    _huber, _sample_indices, enforce_safety, evaluate_geometry)
from texture_sampling_diagnostics import balanced_weights, sampling_metrics


def world_y_regular_mapping(geometry, template):
    """Return the runtime-equivalent fixed-seam world-Y regular UV mapping.

    The world-Y basis depends on physical radii as well as the ellipsoid's
    rotation.  It must consequently be recalculated for every proposed
    geometry rather than copied from an initial candidate.  Keeping this
    helper beside the fitter makes that dependency explicit and lets the
    offline floor exactly use NAG's `world_y` convention.
    """
    if template.get('visible_uv', {}).get('enabled'):
        raise ValueError('World-Y sampling-floor fitting requires canonical regular UV')
    result = copy.deepcopy(template)
    rotation = np.asarray(geometry['rotation'], dtype=np.float64)
    radii = np.asarray(geometry['radii'], dtype=np.float64)
    if rotation.shape != (3, 3) or radii.shape != (3,) or not np.isfinite(rotation).all() or \
            not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('World-Y mapping requires finite 3x3 rotation and positive three-axis radii')
    # This is the NumPy expression of
    # nag.model.ellipsoid_texture.world_y_texture_rotation.  It deliberately
    # stays dependency-free because this fitter is also imported as the
    # top-level module `sampling_floor_ellipsoid`, where `nag.model` is not
    # necessarily importable.  The unit test compares it to the runtime helper.
    axes = np.eye(3, dtype=np.float64)
    vertical = (axes[1] @ rotation) / radii
    vertical /= np.linalg.norm(vertical)
    forward = (axes[2] @ rotation) / radii
    forward -= np.dot(forward, vertical) * vertical
    if np.linalg.norm(forward) < 1e-6:
        forward = (axes[0] @ rotation) / radii
        forward -= np.dot(forward, vertical) * vertical
    forward /= np.linalg.norm(forward)
    right = np.cross(vertical, forward)
    result.update(mode='world_y',
                  rotation=np.stack((right, vertical, forward), axis=1).tolist(),
                  u_offset=0.0)
    return result


def floor_statistics(sigma, reference, weights, threshold=.5):
    ratio = sigma / reference
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        return {'loss': 1e6, 'below_floor_fraction': 1., 'ratio_median': 0.}
    residual = np.maximum(0., np.log(threshold / ratio))
    return {'loss': float(np.average(_huber(residual, .2), weights=weights)),
            'below_floor_fraction': float(np.average(ratio < threshold, weights=weights)),
            'ratio_median': float(np.median(ratio))}


def fit_sampling_floor(pairs, original, sphere, mapping, size,
                       config=ReprojectionFitConfig(), max_sampling=4096,
                       preserve_zero_floor=False, mapping_for_geometry=None,
                       regularizer_version=None):
    """One fixed recipe: continuation weights 5,10,20; 2% geometric budget.

    No radius-excess prior is active. The original feasible fit and every
    feasible optimizer iterate compete under the same final objective. Final
    selection also enforces the geometric budget on all cached supported pairs.
    """
    if config.radius_excess_weight != 0:
        raise ValueError('Sampling-floor v1 isolates sampling: radius-excess weight must be zero')
    if mapping.get('visible_uv', {}).get('enabled'):
        raise ValueError('Fit with the canonical regular mapping, not Visible-UV')
    mapping_depends_on_geometry = mapping_for_geometry is not None
    if mapping_for_geometry is None:
        mapping_for_geometry = lambda geometry: mapping
    if regularizer_version is None:
        regularizer_version = 'sampling_floor_v2' if preserve_zero_floor else 'sampling_floor_v1'
    if not isinstance(regularizer_version, str) or not regularizer_version:
        raise ValueError('regularizer_version must be a non-empty string')
    poses = pairs['poses']
    original = _array_geometry(original)
    initial = enforce_safety(original, poses, config)
    if not np.allclose(initial['radii'], original['radii'], rtol=1e-6, atol=1e-6):
        raise ValueError('Original geometry is not feasible under the saved safety settings')
    supported = np.flatnonzero(pairs['visibility'] == 1)
    if len(supported) < 128:
        raise ValueError('Too few supported pairs')
    pi = supported[_sample_indices(len(supported), config.max_pairs, config.seed)]
    di = _sample_indices(len(pairs['source_points']), config.max_depth_rays, config.seed + 1)
    pixels, z = batched_project(pairs['source_points'], pairs['source_ids'], poses, pairs['intrinsics'])
    valid = np.flatnonzero(inside(pixels, z, size))
    si = valid[_sample_indices(len(valid), max_sampling, config.seed + 2)]
    if len(si) < 128:
        raise ValueError('Too few sampling rays')
    pix, ids = pixels[si], pairs['source_ids'][si]
    weights = balanced_weights(ids, pix, size)
    def candidate_mapping(geometry):
        candidate = mapping_for_geometry(geometry)
        if candidate.get('visible_uv', {}).get('enabled'):
            raise ValueError('Sampling-floor mapping factory returned Visible-UV')
        return candidate

    initial_mapping = candidate_mapping(initial)
    sphere_mapping = candidate_mapping(_array_geometry(sphere))
    reference = sampling_metrics(pix, ids, poses, pairs['intrinsics'], sphere, sphere_mapping)['texels_per_pixel_min']
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Invalid sphere sampling reference')
    rotation = initial['rotation']
    mean = poses[:, :3, 3].mean(axis=0)

    def encode(g):
        return np.r_[(g['center'] - mean) @ rotation, np.log(g['radii'])]

    def decode(x):
        return {'center': mean + rotation @ x[:3], 'radii': np.exp(x[3:]), 'rotation': rotation}

    cache = {}

    def measure(x):
        key = x.tobytes()
        if key not in cache:
            g = decode(x)
            metrics = evaluate_geometry(pairs, g, pi, di, config)
            sigma = sampling_metrics(pix, ids, poses, pairs['intrinsics'], g,
                                     candidate_mapping(g))['texels_per_pixel_min']
            cache[key] = (metrics, floor_statistics(sigma, reference, weights))
            if len(cache) > 100:
                cache.pop(next(iter(cache)))
        return cache[key]

    start = encode(initial)
    before, floor_before = measure(start)
    if (not np.isfinite(before['weighted_data_loss']) or
            min(before['valid_pair_fraction'], before['valid_depth_fraction']) < 1 or
            floor_before['loss'] >= 1e6):
        raise ValueError('Original fit must have valid projections, depth and sampling rays')
    limit = before['weighted_data_loss'] * 1.02 + 1e-6

    def constraints(x):
        m, floor = measure(x)
        values = np.array([limit - m['weighted_data_loss'],
            config.max_camera_normalized_radius - m['max_camera_normalized_radius'],
            m['conservative_camera_clearance_m'] - config.minimum_camera_surface_distance,
            m['minimum_axis_radius_m'] - config.minimum_axis_radius,
            config.max_axis_ratio - m['maximum_axis_ratio'],
            m['valid_pair_fraction'] - 1., m['valid_depth_fraction'] - 1.,
            1. - (floor['loss'] >= 1e6)])
        return np.nan_to_num(values, nan=-1e6, neginf=-1e6, posinf=1e6)

    def score(x, weight=20.):
        m, floor = measure(x)
        value = m['weighted_data_loss'] + weight * floor['loss']
        return value if np.isfinite(value) else 1e12

    candidates = [start.copy()]
    history = []
    bounds = [(-80., 160.), (-30., 30.), (-20., 20.)] + [(np.log(1.), np.log(350.))] * 3
    if any(not lo <= x <= hi for x, (lo, hi) in zip(start, bounds)):
        raise ValueError('Original fit lies outside the legacy parameter bounds')

    def retain(x):
        if np.min(constraints(x)) >= -1e-7:
            candidates.append(x.copy())

    if preserve_zero_floor:
        # Check every valid source ray, not only the optimization subsample.
        all_pix, all_ids = pixels[valid], pairs['source_ids'][valid]
        all_w = balanced_weights(all_ids, all_pix, size)
        all_ref = sampling_metrics(all_pix, all_ids, poses, pairs['intrinsics'], sphere,
                                   sphere_mapping)['texels_per_pixel_min']
        all_sigma = sampling_metrics(all_pix, all_ids, poses, pairs['intrinsics'], original,
                                     candidate_mapping(original))['texels_per_pixel_min']
        gate = floor_statistics(all_sigma, all_ref, all_w)
        full = evaluate_geometry(pairs, original, supported, np.arange(len(pairs['source_points'])), config)
        safe = (np.isfinite(full['weighted_data_loss']) and
                min(full['valid_pair_fraction'], full['valid_depth_fraction']) == 1 and
                full['max_camera_normalized_radius'] <= config.max_camera_normalized_radius + 1e-7 and
                full['conservative_camera_clearance_m'] >= config.minimum_camera_surface_distance - 1e-7 and
                full['minimum_axis_radius_m'] >= config.minimum_axis_radius - 1e-7 and
                full['maximum_axis_ratio'] <= config.max_axis_ratio + 1e-7)
        if gate['loss'] == 0 and safe:
            exact = evaluate_geometry(pairs, original, pi, di, config)
            return {'geometry': original, 'config': asdict(config),
                'regularizer_version': regularizer_version,
                'reference_max_axis_radius_m': float(original['radii'].max()),
                'optimization_pair_count': len(pi), 'optimization_depth_ray_count': len(di),
                'diagnostics': {'initial_safety_projected': exact, 'final': exact}, 'candidates': [],
                'sampling_floor': {'threshold': .5, 'huber_log': .2, 'weights': [5., 10., 20.],
                    'max_sampling_rays': max_sampling, 'sampling_ray_count': len(si),
                    'geometric_budget_fraction': .02, 'initial': gate, 'final': gate,
                    'full_sampling': {'original': gate, 'final': gate},
                    'original_combined_objective': exact['weighted_data_loss'],
                    'final_combined_objective': exact['weighted_data_loss'],
                    'full_original': full, 'full_final': full, 'sphere': sphere,
                    'canonical_mapping': initial_mapping, 'sphere_mapping': sphere_mapping,
                    'initial_mapping': initial_mapping, 'final_mapping': initial_mapping,
                    'mapping_depends_on_geometry': mapping_depends_on_geometry,
                    'retained_original': True,
                    'zero_floor_preserved': True, 'optimization_skipped': True,
                    'zero_floor_check_ray_count': len(valid)}}

    for weight in (5., 10., 20.):
        result = minimize(lambda x: score(x, weight), start, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints}, callback=retain,
                          options={'maxiter': config.max_iterations, 'ftol': 1e-8})
        retain(result.x)
        start = min(candidates, key=score).copy()
        history.append({'weight': weight, 'success': bool(result.success),
                        'message': str(result.message), 'iterations': int(result.nit),
                        'best_final_objective': score(start)})
        print(f'  sampling weight {weight:g}: {result.message}; best final objective {score(start):.5f}', flush=True)

    # Safety repair and full-observation validation precede final ranking.
    full_di = np.arange(len(pairs['source_points']))
    full_before = evaluate_geometry(pairs, initial, supported, full_di, config)
    full_limit = full_before['weighted_data_loss'] * 1.02 + 1e-6
    selected, selected_score = initial, score(encode(initial))
    seen = set()
    for x in sorted(candidates, key=score):
        g = enforce_safety(decode(x), poses, config)
        repaired = encode(g)
        key = np.round(repaired, 9).tobytes()
        if key in seen:
            continue
        seen.add(key)
        if score(repaired) >= selected_score or np.min(constraints(repaired)) < -1e-7:
            continue
        full = evaluate_geometry(pairs, g, supported, full_di, config)
        if full['weighted_data_loss'] <= full_limit and min(full['valid_pair_fraction'], full['valid_depth_fraction']) == 1:
            selected, selected_score = g, score(repaired)
            # Candidates were ordered before safety repair; continue to rank repaired scores.
    final, floor_final = measure(encode(selected))
    all_pix, all_ids = pixels[valid], pairs['source_ids'][valid]
    all_w = balanced_weights(all_ids, all_pix, size)
    all_ref = sampling_metrics(all_pix, all_ids, poses, pairs['intrinsics'], sphere,
                               sphere_mapping)['texels_per_pixel_min']
    full_floor = {}
    for name, g in [('original', initial), ('final', selected)]:
        sigma = sampling_metrics(all_pix, all_ids, poses, pairs['intrinsics'], g,
                                 candidate_mapping(g))['texels_per_pixel_min']
        full_floor[name] = floor_statistics(sigma, all_ref, all_w)
    return {'geometry': selected, 'config': asdict(config), 'regularizer_version': regularizer_version,
            'reference_max_axis_radius_m': float(initial['radii'].max()),
            'optimization_pair_count': len(pi), 'optimization_depth_ray_count': len(di),
            'diagnostics': {'initial_safety_projected': before, 'final': final}, 'candidates': history,
            'sampling_floor': {'threshold': .5, 'huber_log': .2, 'weights': [5., 10., 20.],
                'max_sampling_rays': max_sampling, 'sampling_ray_count': len(si),
                'geometric_budget_fraction': .02, 'initial': floor_before, 'final': floor_final,
                'full_sampling': full_floor, 'original_combined_objective': score(encode(initial)),
                'final_combined_objective': selected_score, 'full_original': full_before,
                'full_final': evaluate_geometry(pairs, selected, supported, full_di, config),
                'sphere': sphere, 'canonical_mapping': initial_mapping,
                'sphere_mapping': sphere_mapping, 'initial_mapping': initial_mapping,
                'final_mapping': candidate_mapping(selected),
                'mapping_depends_on_geometry': mapping_depends_on_geometry,
                'retained_original': bool(np.allclose(encode(initial), encode(selected), rtol=0, atol=1e-8))}}
