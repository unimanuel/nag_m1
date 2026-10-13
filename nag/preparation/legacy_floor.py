"""Publication recipe pinned before the later upright/compact experiments.

The kernel is the exact source at 07cbd74, SHA-256
5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603.
Its nonzero-floor refinement is the original v1 implementation, while its
generic exact-zero preservation rule is v2. No sequence identity is consulted.
"""
from ._legacy_floor_kernel import fit_sampling_floor as kernel


def fit_sampling_floor(pairs, original, sphere, mapping, size, config,
                       max_sampling=4096, floor_threshold=.5, preserve_zero_floor=True,
                       compactness_weight=0., axis_multistart=False):
    if floor_threshold!=.5 or compactness_weight!=0 or axis_multistart:
        raise ValueError('The frozen legacy publication recipe cannot enable experimental options')
    result=kernel(pairs,original,sphere,mapping,size,config,max_sampling=max_sampling,
                  preserve_zero_floor=preserve_zero_floor)
    result['regularizer_version']='sampling_floor_v2' if preserve_zero_floor else 'sampling_floor_v1'
    result['sampling_floor']['kernel_commit']='07cbd74'
    result['sampling_floor']['kernel_sha256']='5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603'
    return result
