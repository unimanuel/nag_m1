"""Exact launcher-only compatibility identities; no numerical-kernel waiver."""
import json

from .jobs import REPO

PATH = 'nag/diagnostics/parallax_observations.py'
OLD = '5d492510a0174d42ac0e415aa2291925dcf582d2f5ea24b2d275a0d145eb6527'
NEW = 'd949bd84fcdf9be93b69cea31b6202d3c4002603273b66bcb7bcf4c4906a1ddf'


def allowed_preparation_change(path, old, new):
    # A future edit, even to this same path, is not automatically compatible.
    return path == PATH and old == OLD and new == NEW


VALIDATION_LEDGER = REPO / 'docs/compatibility/optional_mask_validation_v1.json'
VALIDATION_PATHS = {
    'nag/preparation/compatibility.py',
    'nag/preparation/storage.py',
    'nag/scripts/generalized_ellipsoid_preparation.py',
    'nag/scripts/paper_sphere_baselines.py',
}
WORKER_LEDGER = REPO / 'docs/compatibility/final_worker_policy_v1.json'
WORKER_PATHS = VALIDATION_PATHS | {
    'nag/preparation/jobs.py',
    'nag/scripts/paper_final_baselines.py',
}
RELEASE_CLEANUP_LEDGER = REPO / 'docs/compatibility/release_cleanup_v1.json'
RELEASE_CLEANUP_PATHS = {
    'nag/preparation/compatibility.py',
    'nag/scripts/make_result_comparison_video.py',
}
LINE_ENDING_LEDGER = REPO / 'docs/compatibility/historical_line_endings_v1.json'
LINE_ENDING_PATHS = {
    'nag/callbacks/nag_callback.py',
    'nag/dataset/nag_dataset.py',
    'nag/model/background_image_plane_scene_node_3d.py',
    'nag/model/background_image_sphere_scene_node_3d.py',
    'nag/model/learned_image_plane_scene_node_3d.py',
    'nag/model/nag_functional_model.py',
    'nag/model/nag_model.py',
    'nag/model/sphere_utils.py',
    'nag/model/view_dependent_background_image_plane_scene_node_3d.py',
    'nag/model/view_dependent_background_image_sphere_scene_node_3d.py',
    'nag/model/view_dependent_image_plane_scene_node_3d.py',
    'nag/run/nag_runner.py',
    'nag/strategy/in_image_plane_position_strategy.py',
    'nag/strategy/plane_position_strategy.py',
}


def reviewed_validation_snapshot(snapshot):
    """Canonicalize only exact reviewed launcher revisions, never fitting code.

    Historical metadata stays byte-for-byte unchanged. The ledger records both
    actual source hashes and the earlier compatibility identities. Every file
    in a reviewed revision must match before any of its aliases can apply;
    an additional edit disables that allowance rather than broadening it.
    Worker-policy changes affect training and are explicit in job metadata;
    this is not a claim of identical four- and ten-worker random streams.
    """
    from .jobs import sha
    reviewed = dict(snapshot)
    cleanup = json.loads(RELEASE_CLEANUP_LEDGER.read_text())
    if cleanup.get('schema') != 1 or set(cleanup.get('files', {})) != RELEASE_CLEANUP_PATHS:
        raise ValueError('Invalid release cleanup compatibility ledger')
    files = cleanup['files']
    if all(reviewed.get(p) == entry['after'] and sha(REPO / p) == snapshot.get(p)
           for p, entry in files.items()):
        for p, entry in files.items():
            if entry['before'] is None:
                reviewed.pop(p)
            else:
                reviewed[p] = entry['before']
    line_endings = json.loads(LINE_ENDING_LEDGER.read_text())
    if line_endings.get('schema') != 1 or set(line_endings.get('files', {})) != LINE_ENDING_PATHS:
        raise ValueError('Invalid historical line-ending compatibility ledger')
    files = line_endings['files']
    if all(reviewed.get(p) in (entry['before'], entry['after']) and
           sha(REPO / p) == snapshot.get(p) for p, entry in files.items()):
        reviewed.update({p: entry['before'] for p, entry in files.items()})
    for path, expected in [(WORKER_LEDGER, WORKER_PATHS), (VALIDATION_LEDGER, VALIDATION_PATHS)]:
        ledger = json.loads(path.read_text())
        if ledger.get('schema') != 1 or set(ledger.get('files', {})) != expected:
            raise ValueError('Invalid launcher compatibility ledger')
        files = ledger['files']
        if all(reviewed.get(p) == entry['after'] and sha(REPO / p) == snapshot.get(p)
               for p, entry in files.items()):
            return {p: files[p]['before'] if p in files and digest == files[p]['after'] else digest
                    for p, digest in reviewed.items()}
    return reviewed
