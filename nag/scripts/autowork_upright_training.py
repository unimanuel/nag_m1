#!/usr/bin/env python3
"""Run controlled training for the two axis-neutral upright-fit candidates.

Preparation and geometric screening are separate. This script is intentionally
not invoked by the automatic geometry/diagnostic workflow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'nag'))
sys.path.insert(0, str(REPO / 'nag/scripts'))

from diagnostics.parallax_io import SEQUENCES, write_json
from upright_sampling_floor_validation import _complete
from autowork_upright_compactness import paths, strong_floor_version, version

RECIPES = {'compact': version(.15), 'strong_floor': strong_floor_version(.8)}
DEFAULT_SEQUENCES = ('W191_0_40', 'W254_0_37', 'W254_139_195')
REFERENCE_MANIFEST = REPO / 'runs/2026-09-28_upright_sampling_floor/manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs_for(args):
    jobs = []
    reference = json.loads(REFERENCE_MANIFEST.read_text())['jobs']
    for label in args.sequences:
        segment, first, stop = SEQUENCES[label]
        for recipe in args.recipes:
            source, _, folder = paths(args.ellipsoid_root, label, RECIPES[recipe])
            path = folder / 'ellipsoid.json'
            report_path = folder / 'sampling_floor_diagnostics.json'
            document = json.loads(path.read_text())
            report = json.loads(report_path.read_text())
            original = json.loads(source.read_text())
            recorded_recipe = report['provenance'].get('recipe')
            if recorded_recipe is None and report.get('compactness_weight', 0) > 0:
                # The first compactness preview predates the explicit recipe
                # field; its version and weight remain in the saved report.
                recorded_recipe = 'compact'
            if (document['sequence'] != f'segment-{segment}' or
                    document['frame_slice'] != original['frame_slice'] or
                    document['fit']['regularizer_version'] != RECIPES[recipe] or
                    recorded_recipe != recipe or
                    report['provenance']['source_sha256'] != sha(source)):
                raise ValueError(f'{label} {recipe}: geometry document/provenance mismatch')
            full = report['full_final']
            initial = report['full_original']
            if (full['weighted_data_loss'] > initial['weighted_data_loss'] * 1.02 + 1e-6
                    or full['valid_pair_fraction'] < 1 or full['valid_depth_fraction'] < 1):
                raise ValueError(f'{label} {recipe}: invalid full-data geometry or 2% budget breach')
            if report['final_mapping']['mode'] != 'world_y' or report['final_mapping']['u_offset'] != 0:
                raise ValueError(f'{label} {recipe}: not a fixed-seam world-Y geometry')
            radii = np.asarray(document['fit']['radii'])
            if not np.isfinite(radii).all() or np.any(radii <= 0):
                raise ValueError(f'{label} {recipe}: invalid radii')
            for mode in args.modes:
                suffix = '_sph_visuv' if mode == 'visuv' else ''
                config = (REPO / 'config/waymo/large-ego' /
                          f'segment-{segment}_{first}_{stop-1}_large-undistorted{suffix}.yaml')
                settings = yaml.safe_load(config.read_text())['NAGConfig']
                if (settings.get('checkpoint_path') or settings.get('deterministic', False)
                        or settings.get('max_epochs') != 80):
                    raise ValueError(f'{config}: expected fresh nondeterministic epoch-80 training')
                matches = [job for job in reference if job['label'] == label and job['mode'] == mode]
                if len(matches) != 1 or sha(config) != matches[0]['config_sha256']:
                    raise ValueError(f'{config}: YAML differs from the completed upright-aware control')
                job_id = f'autowork_{label}_{recipe}_{mode}_seed42'
                jobs.append({'id': job_id, 'sequence': label, 'recipe': recipe, 'mode': mode,
                             'frame_count': stop-first, 'config': str(config),
                             'config_sha256': sha(config), 'geometry': str(path),
                             'geometry_sha256': sha(path), 'report_sha256': sha(report_path)})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', nargs='+', choices=list(SEQUENCES),
                        default=list(DEFAULT_SEQUENCES))
    parser.add_argument('--recipes', nargs='+', choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument('--modes', nargs='+', choices=('regular', 'visuv'),
                        default=['regular', 'visuv'])
    parser.add_argument('--ellipsoid-root', type=Path,
                        default=Path('/mnt/d/CGProjectData/waymo/ellipsoids'))
    parser.add_argument('--output-root', type=Path,
                        default=REPO / 'runs/autowork_upright_optimizer_validation')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(len(values) != len(set(values)) for values in
           (args.sequences, args.recipes, args.modes)):
        parser.error('duplicate sequences, recipes or modes are not allowed')
    jobs = jobs_for(args)
    root = args.output_root.resolve()
    if not root.name.startswith('autowork_'):
        parser.error('output-root folder name must begin with autowork_')
    signature = {str(p.relative_to(REPO)): sha(p) for p in
                 (Path(__file__), REPO / 'nag/sampling_floor_ellipsoid.py',
                  REPO / 'nag/model/ellipsoid_texture.py', REPO / 'nag/scripts/run_nag.py')}
    manifest = {'schema': 1, 'purpose': 'upright optimizer geometry validation; no best-run selection',
                'seed': 42, 'texture_orientation': 'world_y', 'jobs': jobs, 'code': signature,
                'reference_manifest_sha256': sha(REFERENCE_MANIFEST)}
    path = root / 'manifest.json'
    if root.exists() and any(root.iterdir()):
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            raise ValueError(f'{root}: existing output has different or missing manifest')
    print(f'{len(jobs)} new training jobs in {root}; dry_run={args.dry_run}', flush=True)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
    for job in jobs:
        parent = root / job['id']
        attempts = sorted(parent.glob('autowork_attempt_*')) if parent.exists() else []
        if any(_complete(attempt, job['frame_count']) is not None for attempt in attempts):
            print(f'[skip completed] {job["id"]}', flush=True)
            continue
        number = max([int(p.name.split('_')[-1]) for p in attempts] or [0]) + 1
        output = parent / f'autowork_attempt_{number:03d}'
        cmd = [sys.executable, str(REPO / 'nag/scripts/run_nag.py'),
               '--config-path', job['config'], '--background-plane-type',
               'nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D',
               '--background-ellipsoid-path', job['geometry'],
               '--background-ellipsoid-scale-margin', '0',
               '--background-ellipsoid-texture-orientation', 'world_y',
               '--seed', '42', '--name', job['id'], '--output-path', str(output)]
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        if sha(job['config']) != job['config_sha256'] or sha(job['geometry']) != job['geometry_sha256']:
            raise ValueError(f'{job["id"]}: training input changed during batch')
        subprocess.run(cmd, cwd=REPO, check=True)
        if _complete(output, job['frame_count']) is None:
            raise RuntimeError(f'{job["id"]}: training returned without complete epoch-80 output')
    if args.dry_run:
        print('Dry run only; no folders or trainings created.', flush=True)


if __name__ == '__main__':
    main()
