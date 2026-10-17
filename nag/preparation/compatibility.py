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
    for path, expected in [(WORKER_LEDGER, WORKER_PATHS), (VALIDATION_LEDGER, VALIDATION_PATHS)]:
        ledger = json.loads(path.read_text())
        if ledger.get('schema') != 1 or set(ledger.get('files', {})) != expected:
            raise ValueError('Invalid launcher compatibility ledger')
        files = ledger['files']
        if all(sha(REPO / p) == entry['after'] for p, entry in files.items()):
            return {p: files[p]['before'] if p in files and digest == files[p]['after'] else digest
                    for p, digest in snapshot.items()}
    return dict(snapshot)
