"""Select completed upright checkpoints for the learned-flow diagnostic.

The fixed-legacy-geometry upright repeats and the upright-aware fitted runs
are existing trainings. This module creates diagnostic jobs only; it never
starts or modifies training.
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from diagnostics.parallax_io import SEQUENCES, mask_paths

REPO = Path(__file__).resolve().parents[2]
FAMILIES = ('upright_fixed', 'upright_fit')


def jobs_from_upright_sources(args):
    from ego_learned_background_flow import file_record, saved_layers
    from upright_sampling_floor_validation import _complete

    catalog = json.loads(args.upright_sources.read_text())
    records = catalog['runs']
    jobs = []
    for label in args.sequences:
        segment, first, stop = SEQUENCES[label]
        frames = list(range(first, stop))
        data = (REPO / 'data/datasets/waymo' / f'segment-{segment}').resolve()
        for mode in args.modes:
            for family in args.upright_families:
                source_family = 'upright' if family == 'upright_fixed' else 'upright_fit'
                selected = [r for r in records if r['sequence'] == label and r['mode'] == mode
                            and r['family'] == source_family]
                if family == 'upright_fixed':
                    selected = [r for r in selected if 'repeat01' in r['path']]
                if len(selected) != 1:
                    raise ValueError(f'{label} {mode} {family}: expected one completed source run')
                run = Path(selected[0]['path']).resolve()
                if _complete(run, len(frames)) is None:
                    raise ValueError(f'{run}: incomplete epoch-80 training')
                checkpoints = list((run / 'checkpoints').glob('final-*.ckpt'))
                if len(checkpoints) != 1 or checkpoints[0].name != 'final-80.ckpt':
                    raise ValueError(f'{run}: exactly final-80.ckpt is required')
                config_path = run / 'training_config.yaml'
                config = yaml.safe_load(config_path.read_text())['NAGConfig']
                if (config['background_ellipsoid_texture_orientation'] != 'world_y' or
                        float(config['background_ellipsoid_scale_margin']) != 0 or
                        config['background_ellipsoid_radii_override'] is not None):
                    raise ValueError(f'{run}: not the expected fixed-geometry upright treatment')
                value = config['background_ellipsoid_path']['value']
                geometry_path = Path(value)
                if not geometry_path.is_absolute():
                    geometry_path = REPO / geometry_path
                geometry_path = geometry_path.resolve()
                geometry_document = json.loads(geometry_path.read_text())
                frame_slice = geometry_document['frame_slice']
                if (geometry_document['sequence'] != f'segment-{segment}' or
                        (frame_slice.get('start'), frame_slice.get('stop'), frame_slice.get('step'))
                        != (first, stop, 1)):
                    raise ValueError(f'{run}: training geometry sequence mismatch')
                fit = geometry_document['fit']
                geometry = {'center': fit['center'], 'radii': fit['radii'],
                            'rotation': fit['rotation_columns']}
                layers = saved_layers(run, frames)
                with Image.open(layers[0]['composite']) as image:
                    size = list(image.size)
                if args.max_frames:
                    indices = np.unique(np.linspace(0, len(frames)-1,
                        min(args.max_frames, len(frames))).round().astype(int))
                else:
                    indices = np.arange(len(frames))
                sources = [file_record(config_path, True), file_record(geometry_path, True),
                           file_record(checkpoints[0])]
                for local in indices:
                    sources.extend(file_record(path) for path in layers[local].values())
                    sources.append(file_record(data / 'images_undistorted' / f'{frames[local]:03d}.png'))
                    sources.extend(file_record(path) for path in mask_paths(data, frames[local]))
                metrics = {}
                for metric in ('MPSNR', 'SSIM', 'LPIPS'):
                    path = run / 'tracker' / f'train_epoch_Mean{metric}.csv'
                    with path.open() as stream:
                        last = list(csv.DictReader(stream, delimiter=';'))[-1]
                    if float(last['step']) != 80 or not np.isfinite(float(last['value'])):
                        raise ValueError(f'{run}: missing final {metric}')
                    metrics[metric] = float(last['value'])
                    sources.append(file_record(path, True))
                job_id = f'{label}_{family}_{mode}_repeat01'
                jobs.append({'id': job_id, 'sequence': label, 'family': family,
                             'mode': mode, 'repeat': 1, 'run': str(run),
                             'checkpoint': str(checkpoints[0]), 'size': size,
                             'frames': frames, 'indices': indices.tolist(),
                             'layers': layers, 'data': str(data), 'metrics': metrics,
                             'geometry': geometry,
                             'mapping': {'texture_orientation': 'world_y',
                                         'visible_uv': mode == 'visuv',
                                         'geometry_document': str(geometry_path)},
                             'inputs': sources})
    return jobs[:args.limit_jobs] if args.limit_jobs else jobs
