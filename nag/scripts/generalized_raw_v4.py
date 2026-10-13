"""Run the unchanged raw-v4 fitter in the processing repo's Python environment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'libs/nag-waymo-processing'))
from nag_waymo_processing.ellipsoid import EllipsoidJob, fit_ellipsoid_job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job-json', type=Path, required=True)
    p.add_argument('--cloud-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    job = json.loads(args.job_json.read_text())
    task = EllipsoidJob('segment-' + job['segment'], job['first'], job['stop'], job['step'],
                        Path(job['pose_path']).name, tuple(job['source_configs']))
    result = fit_ellipsoid_job(task, args.cloud_root, str(Path(job['data_dir']).parent), args.output,
        correct_pose_mismatches=True, fit_mode='ray_depth_proxy', ray_depth_background_only=True,
        ray_depth_max_samples=24000, ray_depth_max_camera_normalized_radius=.8,
        ray_depth_minimum_camera_surface_distance=2., output_suffix='raw_v4')
    meta = {'executable': sys.executable, 'python': sys.version,
            'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'scipy']}, 'result': result}
    (Path(args.output) / 'raw_environment.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
