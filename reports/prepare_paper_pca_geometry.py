#!/usr/bin/env python3
"""Export the unchanged all-FRONT-point PCA enclosure in its processing Python.

This worker never imports NAG training code. Its temporary export is flattened
into a newly owned attempt directory; historical geometry is never edited.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def export(item, cloud_root, output):
    sys.path.insert(0, str(REPO / 'libs/nag-waymo-processing'))
    from nag_waymo_processing.ellipsoid import EllipsoidJob, fit_ellipsoid_job

    output = Path(output)
    job = EllipsoidJob('segment-' + item['segment'], item['first'], item['stop'], 1,
                       Path(item['pose_path']).name, (item['config'],))
    # The caller owns this new attempt. Never reuse the processing command's
    # permissive existing-output shortcut or overwrite a historical PCA mesh.
    with tempfile.TemporaryDirectory(prefix='pca_export_', dir=output) as temporary:
        result = Path(fit_ellipsoid_job(job, str(cloud_root), str(Path(item['data_dir']).parent),
            temporary, correct_pose_mismatches=True, fit_mode='pca_enclosing'))
        for path in sorted(result.parent.iterdir()):
            if not path.is_file():
                raise ValueError('Unexpected directory in PCA export: ' + str(path))
            target = output / path.name
            if target.exists():
                raise FileExistsError('Refuse to overwrite PCA attempt artifact: ' + str(target))
            path.replace(target)
    return output / 'ellipsoid.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-json', type=Path, required=True)
    parser.add_argument('--cloud-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_dir():
        parser.error('Output must be a newly owned existing attempt directory')
    export(json.loads(args.job_json.read_text()), args.cloud_root, args.output)


if __name__ == '__main__':
    main()
