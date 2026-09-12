#!/usr/bin/env python3
"""Offline parallax and spatial-error evaluation of completed W191/W254 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'nag'))  # Avoid nag.__init__ importing the training stack.

from diagnostics.parallax_io import SEQUENCES, input_files, inventory, snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','evaluate','spatial','summarize'])
    p.add_argument('--sequences',nargs='+',choices=list(SEQUENCES),default=list(SEQUENCES))
    p.add_argument('--output',type=Path,default=Path('/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1'))
    p.add_argument('--data-root',type=Path,default=REPO/'data/datasets/waymo')
    p.add_argument('--cloud-root',type=Path,default=Path('/mnt/d/CGProjectData/waymo/pointclouds'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--rays-per-frame',type=int,default=256)
    p.add_argument('--targets-per-frame',type=int,default=8)
    p.add_argument('--frame-limit',type=int,default=0,help='Smoke-test only: evenly spaced subset; report labels it partial')
    p.add_argument('--visibility-radius',type=float,default=2.,help='Nearest target-LiDAR support radius in rendered pixels')
    p.add_argument('--depth-absolute',type=float,default=.5,help='Depth evidence tolerance in metres')
    p.add_argument('--depth-relative',type=float,default=.03)
    p.add_argument('--alignment-p90-limit',type=float,default=15.,help='Stop if distorted-label vs projected pixel P90 exceeds this')
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--boundary-width',type=int,default=3,help='Dilation/erosion radius in rendered pixels')
    p.add_argument('--display-cap',type=float,default=.1,help='Signed MAE display saturation')
    args=p.parse_args();args.repo=REPO
    if (args.rays_per_frame<1 or args.targets_per_frame<1 or args.frame_limit<0 or args.frame_limit==1 or
        args.visibility_radius<=0 or args.depth_absolute<0 or args.depth_relative<0 or
        args.alignment_p90_limit<=0 or args.boundary_width<1 or not 0<args.display_cap<=1):
        p.error('Invalid sampling, tolerance, frame-limit, boundary-width, or display-cap value')
    inv=inventory(REPO,args.sequences,args.data_root)
    if args.dry_run:
        for label in args.sequences:
            runs=[r for r in inv['runs'] if r['sequence']==label]
            segment,start,stop=SEQUENCES[label]
            files=input_files(args.data_root/f'segment-{segment}',args.cloud_root/f'segment-{segment}',range(start,stop))
            snapshot(files)  # Validate existence and readability before processing.
            print(label+': '+', '.join(r['id'] for r in runs))
        for m in inv['missing']:print('MISSING BASELINE:',m)
        print(f'{len(inv["runs"])} runs; output {args.output}; no files written')
        return
    if args.stage=='prepare':
        from diagnostics.parallax_observations import prepare_sequence
        for label in args.sequences:prepare_sequence(args,inv,label)
    elif args.stage=='evaluate':
        from diagnostics.parallax_evaluate import evaluate_sequence
        for label in args.sequences:evaluate_sequence(args,inv,label)
    elif args.stage=='spatial':
        from diagnostics.parallax_spatial import spatial_sequence
        for label in args.sequences:spatial_sequence(args,inv,label)
    else:
        from diagnostics.parallax_report import report
        report(args,inv)


if __name__=='__main__':
    main()
