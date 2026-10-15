"""Exact, reviewed preparation-only hash allowance; no model/training waiver."""
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


def reviewed_validation_snapshot(snapshot):
    """Canonicalize only an exact reviewed validator patch, never fitting code.

    Historical metadata stays byte-for-byte unchanged. The ledger records both
    actual source hashes and the earlier compatibility identities. All four
    current files must match the reviewed patch before any alias can apply;
    an additional edit disables the allowance rather than broadening it.
    """
    from .jobs import sha
    ledger = json.loads(VALIDATION_LEDGER.read_text())
    if ledger.get('schema') != 1 or set(ledger.get('files', {})) != VALIDATION_PATHS:
        raise ValueError('Invalid optional-mask validation compatibility ledger')
    files = ledger['files']
    if any(sha(REPO / path) != entry['after'] for path, entry in files.items()):
        return dict(snapshot)
    return {path: files[path]['before'] if path in files and digest == files[path]['after'] else digest
            for path, digest in snapshot.items()}
