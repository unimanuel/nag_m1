#!/usr/bin/env python3
"""Extend the frozen loss ablation to the four remaining ego intervals.

Reuse the original audited driver and numerical kernel without editing either
file or any historical manifest. Only cohort selection, output namespaces and
additional launcher provenance differ from the completed five-interval batch.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import paper_loss_ablations as driver


INTERVALS = ('W191_40_128', 'W191_128_198', 'W254_0_37', 'W254_139_195')
BATCHES = {
    'part1of2': INTERVALS[:2],
    'part2of2': INTERVALS[2:],
}
DEFAULT_PREP = Path('/mnt/d/CGProjectData/waymo/ellipsoids/autowork_remaining_ego_loss_ablations_v1')
DEFAULT_TRAIN = driver.REPO / 'runs/autowork_remaining_ego_loss_ablations_v1'
LAUNCHERS = (
    'prepare_remaining_ego_loss_ablations.sh',
    'run_remaining_ego_loss_ablations.sh',
    'run_remaining_ego_loss_ablations_part1of2.sh',
    'run_remaining_ego_loss_ablations_part2of2.sh',
)


@contextmanager
def remaining_intervals():
    """Scope the legacy driver's cohort constant; never change its source."""
    original = driver.ABLATION_INTERVALS
    driver.ABLATION_INTERVALS = INTERVALS
    try:
        yield
    finally:
        driver.ABLATION_INTERVALS = original


def inventory(catalog):
    with remaining_intervals():
        plan = driver.inventory(catalog)
    if tuple(item['label'] for item in plan['intervals']) != INTERVALS:
        raise ValueError('Expected exactly the four remaining ego intervals')
    plan['cohort'] = 'remaining_ego_loss_ablations_v1'
    # These pins supplement, never replace or alias, the original driver,
    # numerical kernel, release-fixture and training-environment signatures.
    for path in (Path(__file__).resolve(), *(driver.REPO / name for name in LAUNCHERS)):
        plan['diagnostic_code'][str(path)] = driver.sha(path)
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'prepare', 'train', 'summary'))
    parser.add_argument('--catalog', type=Path, default=driver.DEFAULT_CATALOG)
    parser.add_argument('--preparation-root', type=Path, default=DEFAULT_PREP)
    parser.add_argument('--training-root', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--intervals', nargs='+', choices=INTERVALS)
    parser.add_argument('--max-hours', type=float, default=8.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args(argv)
    if args.summary:
        args.stage = 'summary'
    if args.intervals and args.stage not in ('train', 'summary'):
        parser.error('--intervals selects training/summary queues; preparation audits all four')
    if not np.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error('Positive finite hours required')
    historical = (driver.DEFAULT_PREP.resolve(), driver.DEFAULT_TRAIN.resolve())
    for root in (args.preparation_root, args.training_root):
        resolved = root.resolve()
        if not root.name.startswith('autowork_') or resolved == driver.REPO.resolve():
            parser.error('Use separate autowork_ namespaces')
        if any(resolved == old or old in resolved.parents for old in historical):
            parser.error('Keep the completed original ablation roots read-only')
    if args.preparation_root.resolve() == args.training_root.resolve():
        parser.error('Preparation/training roots must be distinct')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = inventory(args.catalog)
    with remaining_intervals():
        # Match the original immutable-manifest and retry contracts. In
        # particular, read-only commands neither create roots nor fit/train.
        driver.freeze(args.preparation_root, plan, dry_run=True)
        cells = driver.matrix(plan, args.preparation_root,
                              require_prepared=args.stage in ('train', 'summary'))
        driver.show(plan, cells)
        if args.stage == 'audit' or args.dry_run and args.stage == 'prepare':
            return
        if args.stage == 'prepare':
            driver.prepare(plan, args.preparation_root)
        else:
            driver.train_or_summary(args, plan, cells)


if __name__ == '__main__':
    main()
