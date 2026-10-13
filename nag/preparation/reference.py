"""Native initialization-only sphere references; never instantiate a trainer."""
import json
from pathlib import Path
import sys

import numpy as np

from .jobs import REPO, SPHERE, settings, sha


def mapping_from_diagnostic(diag, config):
    if diag['radius_source'] != 'visible_plane_centers' or diag['visible_uv_texture']['enabled']:
        raise ValueError('Requires the visible-plane-center regular sphere reference')
    return {'mode': 'legacy', 'rotation': np.eye(3).tolist(), 'u_offset': 0,
            'y_flip': config.get('background_plane_kwargs', {}).get('sphere_uv_y_flip', False),
            'visible_uv': diag['visible_uv_texture'], 'resolution_hw': diag['color_mask_resolution']}


def _native_config(path, job, output):
    sys.path.insert(0, str(REPO / 'tools'))
    sys.path.insert(0, str(REPO))
    from nag.config.nag_config import NAGConfig
    cfg = NAGConfig.load_from_file(str(path))
    cfg.output_path = str(output)
    cfg.data_path = job.data_dir
    for field, suffix in [('images_path','images_undistorted'),('masks_path','masks'),('depths_path','depth'),
                          ('boxes_path','boxes'),('boxes_object_id_mask_mapping_path','box_id_mapping.json'),
                          ('mask_ids_filter_path','mask_indices_filter.json')]:
        setattr(cfg,field,str(Path(job.data_dir)/suffix))
    cfg.pinhole_camera_config = job.pose_path
    cfg.prepare(create_output_path=False)
    cfg.background_plane_type = SPHERE
    cfg.background_ellipsoid_path = None
    cfg.background_sphere_radius_override = None
    cfg.background_sphere_radius_from_visible_plane_centers = True
    if cfg.checkpoint_path or cfg.checkpoint_tracker_path:
        raise ValueError('Initialization-only export must not load checkpoints')
    if str(Path(str(cfg.data_path)).resolve()) != job.data_dir:
        raise ValueError('Resolved native dataset root differs from the inspected job')
    return cfg


def initialization_signature(cfg, job):
    """Normalize only native initialization inputs, not optimizer/update metadata."""
    def value(v):
        if isinstance(v, type):
            return v.__module__ + '.' + v.__name__
        if isinstance(v, dict):
            return {k: value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [value(x) for x in v]
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (str, int, bool, float)) or v is None:
            return v
        return str(v)
    explicit = {'seed', 'model_type', 'dtype', 'tinycudann_network_dtype', 'init_max_image_size',
                'max_image_size', 'learn_resolution_factor', 'final_plot_resolution_factor',
                'normalize_camera', 'new_homography', 'skip_object_on_missing_box',
                'mask_background_coverage_threshold', 'has_background_plane', 'has_camera_aberration_plane'}
    excluded = {'background_ellipsoid_path', 'background_ellipsoid_scale_margin',
                'background_ellipsoid_texture_orientation', 'background_sphere_radius_override'}
    keys = [k for k in vars(cfg) if k in explicit or k.startswith(('plane_', 'object_', 'is_', 'background_'))]
    result = {k: value(getattr(cfg, k)) for k in keys if k not in excluded}
    if cfg.object_rigid_control_points is None or not isinstance(cfg.object_rigid_control_points, int):
        result['object_rigid_control_points'] = int(round(len(job.frames) * cfg.object_rigid_control_points_ratio))
    result['plane_init_strategy_kwargs'] = dict(result.get('plane_init_strategy_kwargs') or {})
    result['plane_init_strategy_kwargs']['position_spline_fitting'] = cfg.object_position_spline_approximation
    result.update(data_dir=job.data_dir, pose_sha256=sha(job.pose_path), frames=list(job.frames), size=list(job.size))
    return result


def initialize_runner(runner):
    # Do not call build(): it constructs a sampler, dataloader and Trainer.
    runner.dataset = runner.create_bundle_dataset(None)
    return runner.create_model(None)


def export_reference(job, output):
    output = Path(output)
    cfg = _native_config(job.regular_config, job, output)
    signature = initialization_signature(cfg, job)
    from nag.run.nag_runner import NAGRunner
    from diagnostics.parallax_io import diagnostic
    runner = NAGRunner(cfg)
    model = initialize_runner(runner)
    nodes = list(model.world.get_scene_children())
    sphere = next(n for n in nodes if getattr(n, 'is_spherical_background', False))
    full_radius = float(sphere.sphere_radius.detach().cpu())
    centers = sphere.get_global_position(t=runner.dataset.frame_timestamps)[:, :3, 3].detach().cpu().numpy()
    diag_path = output / 'initial/background_diagnostics/sphere_geometry.txt'
    diag = diagnostic(diag_path)
    actual_size = (model.camera._image_resolution.flip(-1).detach().cpu().numpy() * cfg.final_plot_resolution_factor).astype(int)
    if list(actual_size) != list(job.size):
        raise ValueError('Descriptor render size differs from native NAG final resolution')
    geometry = {'center': diag['center'], 'radii': [float(diag['radius'])]*3, 'rotation': np.eye(3).tolist()}
    return {'schema': 1, 'job': job.document(), 'geometry': geometry,
            'full_precision': {'center': centers[0].tolist(), 'radius': full_radius},
            'mapping': mapping_from_diagnostic(diag, settings(job.regular_config)),
            'initialization_signature': signature, 'precision': 'legacy diagnostic 0.0001 m; full precision retained separately',
            'source': 'native_initialization_only', 'native_diagnostic': str(diag_path),
            'trainer_constructed': False, 'training_performed': False}


def completed_reference(job, run, output):
    from diagnostics.parallax_io import diagnostic, intrinsics
    sys.path.insert(0, str(REPO / 'nag/scripts'))
    from ego_repeatability import completed
    run = Path(run)
    if (completed(run, len(job.frames)) is None or
            not (run / 'exit_000.txt').read_text().startswith('Exit code: 0.')):
        raise ValueError('Sphere reference run is incomplete')
    saved = settings(run / 'training_config.yaml')
    recorded_root = saved['data_path']
    while isinstance(recorded_root,dict):recorded_root=recorded_root['value']
    if str((REPO/recorded_root).resolve()) != job.data_dir:
        raise ValueError('Reference dataset root mismatch')
    if saved['background_plane_type'] != SPHERE or not saved.get('background_sphere_radius_from_visible_plane_centers'):
        raise ValueError('Reference run is not the requested sphere policy')
    sl = saved['frame_indices_filter']
    if (sl['start'], sl['stop'], sl.get('step') or 1) != (job.first, job.stop, job.step):
        raise ValueError('Reference interval mismatch')
    camera = json.loads(Path(job.pose_path).read_text())
    if (not np.array_equal(saved['pinhole_camera_config']['position'], camera['position']) or
            not np.array_equal(intrinsics(saved['pinhole_camera_config'], job.size), intrinsics(camera, job.size))):
        raise ValueError('Reference camera calibration mismatch')
    wanted = _native_config(job.regular_config, job, output)
    recorded = _native_config(run / 'training_config.yaml', job, output)
    if initialization_signature(wanted, job) != initialization_signature(recorded, job):
        raise ValueError('Reference native initialization settings mismatch')
    diag_path = run / 'initial/background_diagnostics/sphere_geometry.txt'
    diag = diagnostic(diag_path)
    return {'schema': 1, 'job': job.document(),
            'geometry': {'center': diag['center'], 'radii': [float(diag['radius'])]*3, 'rotation': np.eye(3).tolist()},
            'mapping': mapping_from_diagnostic(diag, settings(job.regular_config)),
            'initialization_signature': initialization_signature(wanted, job),
            'source': 'validated_completed_sphere', 'source_run': str(run),
            'source_config_sha256': sha(run / 'training_config.yaml'), 'source_diagnostic_sha256': sha(diag_path),
            'full_precision': None, 'precision': 'legacy diagnostic 0.0001 m; full precision unavailable from this text export'}
