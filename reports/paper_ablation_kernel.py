"""Isolated, hash-pinned zero-reprojection ablation of the released fitter.

The production source is never edited or monkey-patched. Only its weight
validation expression is changed in a private module; all objective, sampling,
multistart, solver, bounds, safety and selection statements stay identical.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
import types

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'nag/reprojection_ellipsoid.py'
SOURCE_SHA256 = 'b4fcb7036efec29c014e41d3ed4fb291996ea5d917196e23c404fa38c9a6e8b6'
OLD = '''if min(config.reprojection_weight, config.depth_weight, config.reprojection_huber_px,
           config.depth_huber_log, config.radius_excess_huber_log) <= 0.0 or config.radius_excess_weight < 0.0:'''
NEW = '''if config.reprojection_weight < 0.0 or min(config.depth_weight, config.reprojection_huber_px,
           config.depth_huber_log, config.radius_excess_huber_log) <= 0.0 or config.radius_excess_weight < 0.0:'''


def ablation_source(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Released reprojection source changed; review/version this ablation instead of widening its pin')
    text = raw.decode('utf-8')
    if text.count(OLD) != 1:
        raise ValueError('Expected exactly one released positive-weight validation')
    return text.replace(OLD, NEW, 1)


@lru_cache(maxsize=1)
def kernel():
    sys.path.insert(0, str(REPO/'nag'))
    name = '_paper_loss_ablation_kernel_v1'
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module  # dataclasses resolves annotations through this entry
    try:
        exec(compile(ablation_source(SOURCE.read_bytes()), str(SOURCE)+'[ablation-v1]', 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def fit_depth_only(pairs, initial, config=None):
    module = kernel()
    config = config or module.ReprojectionFitConfig(reprojection_weight=0.)
    if config.reprojection_weight != 0. or config.depth_weight <= 0 or config.radius_excess_weight != 0.:
        raise ValueError('D0 requires zero reprojection, positive depth and no radius-scale regularizer')
    result = module.fit_reprojection_depth_ellipsoid(pairs, initial, config)
    result['regularizer_version'] = 'ablation_depth_only_v1'
    return result
