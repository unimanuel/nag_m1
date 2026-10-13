"""Seven-interval numerical regression gate, never a training experiment."""
import json
from pathlib import Path

import numpy as np

from .jobs import REPO, discover, sha
from .pipeline import prepare
from .storage import code_signature, recipe, write_json
from diagnostics.parallax_io import snapshot
from fit_reprojection_ellipsoid import _source_geometry


def geometry_difference(actual, expected, exact=False):
    a, b = _source_geometry(actual), _source_geometry(expected)
    errors,failures = {},[]
    for key in ('center', 'radii', 'rotation'):
        errors[key] = float(np.max(np.abs(a[key]-b[key])))
        same = np.array_equal(a[key], b[key]) if exact else np.allclose(a[key], b[key],
            rtol=0 if key == 'rotation' else 1e-6, atol=1e-6 if key == 'rotation' else 1e-4)
        if not same:
            failures.append(key)
    if failures:
        error=ValueError(f'Geometry parity failed: {failures}; maximum differences {errors}')
        error.maximum_errors=errors
        raise error
    return errors


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
        'schema': 1, 'fixture_manifest': str(fixtures), 'fixture_sha256': sha(fixtures),
        'code': code_signature(), 'recipe': recipe(), 'cached': {}, 'end_to_end': {}}
    if report['code'] != code_signature() or report['fixture_sha256'] != sha(fixtures) or report['recipe'] != recipe():
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
                    'objective_difference': objective-old_objective, 'result': result}
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
            or sha(report['fixture_manifest']) != report['fixture_sha256']):
        raise ValueError('Static preparation/training requires the current full seven-interval ego compatibility gate')
    labels={r['label']for r in json.loads(Path(report['fixture_manifest']).read_text())['rows']}
    if len(labels)!=7 or any(set(report[m])!=labels or not all(r.get('passed')for r in report[m].values())
                            for m in ['cached','end_to_end']):
        raise ValueError('The release gate is incomplete')
    return report
