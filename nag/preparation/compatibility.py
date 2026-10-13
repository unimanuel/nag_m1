"""Exact, reviewed preparation-only hash allowance; no model/training waiver."""
from .jobs import REPO

PATH = 'nag/diagnostics/parallax_observations.py'
OLD = '5d492510a0174d42ac0e415aa2291925dcf582d2f5ea24b2d275a0d145eb6527'
NEW = 'd949bd84fcdf9be93b69cea31b6202d3c4002603273b66bcb7bcf4c4906a1ddf'


def allowed_preparation_change(path, old, new):
    # A future edit, even to this same path, is not automatically compatible.
    return path == PATH and old == OLD and new == NEW
