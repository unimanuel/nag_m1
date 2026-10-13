"""Seven-interval numerical regression gate, never a training experiment."""
import json
from pathlib import Path

import numpy as np

from .jobs import REPO, discover, sha
from .pipeline import prepare
from .storage import code_signature, recipe, write_json
from diagnostics.parallax_io import snapshot
from fit_reprojection_ellipsoid import _source_geometry


def acceptance_policy():
    """Explicit user-approved replay policy; never changes fitter settings."""
    return {'version': 'numerical_equivalence_1mm_v1',
            'approval': 'User approved 1 mm optimized-center/radius tolerance on 2026-10-13',
            'geometry': {'rtol': 1e-6, 'center_radius_atol_m': 1e-3, 'rotation_atol': 1e-6},
            'preserved_geometry': 'exact', 'observations': 'exact dtype/order/values',
            'sphere_reference_and_mapping': 'exact',
            'objectives': {'rtol': 1e-6, 'atol': 1e-6},
            'distributions': {'rtol': 1e-6, 'pixel_atol': 1e-3,
                              'angle_deg_atol': 1e-4, 'log_depth_atol': 1e-4},
            'sampling_statistics': {'rtol': 1e-5, 'atol': 1e-6},
            'safety': 'unchanged recipe limits and 2% full-observation data-loss budget'}


def geometry_difference(actual, expected, exact=False):
    a, b = _source_geometry(actual), _source_geometry(expected)
    errors,failures = {},[]
    for key in ('center', 'radii', 'rotation'):
        if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
            errors[key] = None
            failures.append(key)
            continue
        errors[key] = float(np.max(np.abs(a[key]-b[key])))
        same = np.array_equal(a[key], b[key]) if exact else np.allclose(a[key], b[key],
            rtol=0 if key == 'rotation' else 1e-6, atol=1e-6 if key == 'rotation' else 1e-3)
        if not same:
            failures.append(key)
    if failures:
        error=ValueError(f'Geometry parity failed: {failures}; maximum differences {errors}')
        error.maximum_errors=errors
        raise error
    return errors


def compare_fit_diagnostics(actual, expected, config):
    """Require close losses/observations plus the original hard safety policy."""
    policy = acceptance_policy()
    a, b = actual['fit']['sampling_floor'], expected['fit']['sampling_floor']
    errors, deltas = [], {}

    def close(name, x, y, rtol=1e-6, atol=1e-6):
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            errors.append(f'Nonfinite diagnostic: {name}')
            return
        deltas[name] = float(x-y)
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            errors.append(f'Diagnostic parity failed: {name}; difference={x-y}')

    for field in ['original_combined_objective', 'final_combined_objective']:
        close(field, a[field], b[field], **policy['objectives'])
    for stage in ['full_original', 'full_final']:
        x, y = a[stage], b[stage]
        for field in ['weighted_data_loss', 'reprojection_huber_loss', 'depth_huber_loss', 'radius_excess_huber_loss']:
            close(stage+'.'+field, x[field], y[field], **policy['objectives'])
        for field, atol in [('pixel_error', policy['distributions']['pixel_atol']),
                            ('camera_angle_deg', policy['distributions']['angle_deg_atol']),
                            ('absolute_log_depth_error', policy['distributions']['log_depth_atol'])]:
            for count in ['count', 'finite_count', 'invalid_count']:
                if x[field][count] != y[field][count]:
                    errors.append(f'Observation diagnostic count changed: {stage}.{field}.{count}')
            for stat in ['mean', 'median', 'p90', 'p95']:
                close(stage+'.'+field+'.'+stat, x[field][stat], y[field][stat], atol=atol)
        for field in ['valid_pair_fraction', 'valid_depth_fraction']:
            if x[field] != 1. or x[field] != y[field]:
                errors.append(f'Full-observation validity changed: {stage}.{field}')
        # Safety allowances are the existing kernel's floating-point allowances,
        # not the new 1 mm coordinate-equivalence allowance.
        limits = [(x['max_camera_normalized_radius'], '<=', config['max_camera_normalized_radius']+1e-7),
                  (x['conservative_camera_clearance_m'], '>=', config['minimum_camera_surface_distance']-1e-7),
                  (x['minimum_axis_radius_m'], '>=', config['minimum_axis_radius']-1e-7),
                  (x['maximum_axis_ratio'], '<=', config['max_axis_ratio']+1e-7)]
        for value, operator, limit in limits:
            if not np.isfinite(value) or (value > limit if operator == '<=' else value < limit):
                errors.append(f'Unchanged safety check failed: {stage}: {value} {operator} {limit}')
    for stage in ['original', 'final']:
        for field in ['loss', 'below_floor_fraction', 'ratio_median']:
            close('full_sampling.'+stage+'.'+field, a['full_sampling'][stage][field],
                  b['full_sampling'][stage][field], **policy['sampling_statistics'])
    limit = a['full_original']['weighted_data_loss']*1.02+1e-6
    if a['full_final']['weighted_data_loss'] > limit:
        errors.append('Unchanged 2% full-observation data-loss budget failed')
    return {'passed': not errors, 'errors': errors, 'differences': deltas,
            'full_data_loss_limit': limit,
            'final_safety': {k:a['full_final'][k] for k in ['max_camera_normalized_radius',
                'conservative_camera_clearance_m', 'minimum_axis_radius_m', 'maximum_axis_ratio']}}


def verify_fixtures(path):
    fixture = json.loads(Path(path).read_text())
    if len(fixture['rows'])!=7 or len({r['label']for r in fixture['rows']})!=7:
        raise ValueError('The compatibility fixture must contain all seven historical ego intervals')
    for p, h in fixture['files'].items():
        if sha(p) != h:
            raise ValueError(f'Immutable ego fixture changed: {p}')
    for row in fixture['rows']:
        if snapshot([x['path'] for x in row['input_snapshot']]) != row['input_snapshot']:
            raise ValueError(f'Historical source input metadata changed: {row["label"]}')
    return fixture


def compare_arrays(old_path, new_path):
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError('Observation field set changed')
        for key in old.files:
            if old[key].dtype != new[key].dtype or not np.array_equal(old[key], new[key], equal_nan=True):
                raise ValueError(f'Observation parity failed: {key}')
        return {key: list(old[key].shape) for key in old.files}


def compare_mapping(actual, expected):
    if actual != expected:
        raise ValueError('Canonical sphere-reference mapping changed')
    return actual


def verify(fixtures, root, cloud_root, level='all', selected=()):
    fixture = verify_fixtures(fixtures)
    all_rows = fixture['rows']
    rows = [r for r in all_rows if not selected or r['label'] in selected]
    if selected and len(rows) != len(set(selected)):
        raise ValueError('Unknown/duplicate ego verification selection')
    report_path = Path(root)/'verification.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'schema': 2, 'fixture_manifest': str(fixtures), 'fixture_sha256': sha(fixtures),
        'code': code_signature(), 'recipe': recipe(), 'acceptance_policy': acceptance_policy(),
        'cached': {}, 'end_to_end': {}}
    if (report['code'] != code_signature() or report['fixture_sha256'] != sha(fixtures)
            or report['recipe'] != recipe() or report.get('acceptance_policy') != acceptance_policy()):
        # Keep old reports: a new code revision must explicitly use another
        # verification directory rather than resetting a previous pass.
        raise ValueError('Verification report belongs to another code/input revision; select a new --verification-root')
    for row in rows:
        job = discover(config_paths=[row['config']])[0]
        if (job.segment, job.first, job.stop, job.step, list(job.size)) != (
                row['segment'], row['first'], row['stop'], 1, row['size']):
            raise ValueError('Generalized discovery differs from the historical interval')
        for mode in (['cached','end_to_end'] if level == 'all' else [level]):
            if row['label'] in report[mode] and report[mode][row['label']].get('passed'):
                print(f'[verified reuse] {mode} {row["label"]}', flush=True)
                continue
            print(f'[verify] {mode} {row["label"]}', flush=True)
            try:
                result = prepare(job, Path(root)/('autowork_'+mode), cloud_root,
                                 fixture=row, end_to_end=mode=='end_to_end')
                meta = json.loads(Path(result['prepared']).read_text())
                shapes = compare_arrays(row['pairs'], Path(meta['directory'])/'pairs.npz')
                failures=[]
                def compare(actual,expected,exact=False):
                    try:return geometry_difference(actual,expected,exact)
                    except ValueError as e:
                        failures.append(str(e));return getattr(e,'maximum_errors',{})
                reprojection = compare(json.loads(Path(result['reprojection']).read_text()),
                                       json.loads(Path(row['reprojection']).read_text()))
                floor_doc = json.loads(Path(result['floor']).read_text())
                reference = json.loads(Path(result['sphere_reference']).read_text())
                try:compare_mapping(reference['mapping'],row['mapping'])
                except ValueError as e:failures.append(str(e))
                historical = json.loads(Path(row['floor']).read_text())
                preserved = bool(floor_doc['fit']['sampling_floor'].get('zero_floor_preserved'))
                if preserved != (row['floor_version'] == 2):
                    failures.append('The v2 preservation/refinement decision changed')
                floor_diff = compare(floor_doc, historical, exact=preserved)
                diagnostic_check = compare_fit_diagnostics(floor_doc, historical, recipe()['reprojection'])
                failures.extend(diagnostic_check['errors'])
                if mode == 'end_to_end':
                    raw_diff = compare(json.loads(Path(result['raw_v4']).read_text()),
                                       json.loads(Path(row['raw_v4']).read_text()))
                    sphere = reference['geometry']
                    if any(not np.array_equal(sphere[k], row['sphere'][k]) for k in ['center','radii','rotation']):
                        failures.append('Native initialization sphere reference changed at legacy diagnostic precision')
                else:
                    raw_diff = {'source': 'same immutable initializer'}
                objective = floor_doc['fit']['sampling_floor']['full_final']['weighted_data_loss']
                old_objective = historical['fit']['sampling_floor']['full_final']['weighted_data_loss']
                report[mode][row['label']] = {'passed': not failures, 'errors':failures, 'arrays': shapes, 'reprojection_maximum_errors': reprojection,
                    'floor_maximum_errors': floor_diff, 'raw_maximum_errors': raw_diff, 'preserved': preserved,
                    'objective_difference': objective-old_objective, 'diagnostic_check': diagnostic_check,
                    'result': result}
                print(f'[{"FAIL" if failures else "PASS"}] {mode} {row["label"]}; floor differences={floor_diff}', flush=True)
            except Exception as e:
                report[mode][row['label']] = {'passed': False, 'error': str(e)}
                write_json(report_path, report)
                print(f'[FAIL] {mode} {row["label"]}: {e}',flush=True)
            write_json(report_path, report)
    report['released'] = all(len(report[m])==len(all_rows) and all(r.get('passed') for r in report[m].values())
                             for m in ['cached','end_to_end'])
    write_json(report_path, report)
    return report


def release_gate(path):
    if not Path(path).is_file():
        raise ValueError('Missing current seven-interval release report; run verify_ego_ellipsoid_preparation.sh first')
    report = json.loads(Path(path).read_text())
    if (not report.get('released') or report['code'] != code_signature() or report['recipe'] != recipe()
            or report.get('acceptance_policy') != acceptance_policy()
            or sha(report['fixture_manifest']) != report['fixture_sha256']):
        raise ValueError('Static preparation/training requires the current full seven-interval ego compatibility gate')
    labels={r['label']for r in json.loads(Path(report['fixture_manifest']).read_text())['rows']}
    if len(labels)!=7 or any(set(report[m])!=labels or not all(r.get('passed') and
                            r.get('diagnostic_check',{}).get('passed') for r in report[m].values())
                            for m in ['cached','end_to_end']):
        raise ValueError('The release gate is incomplete')
    return report
