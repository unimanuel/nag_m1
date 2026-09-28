"""Launcher contracts for the separate upright-aware sampling-floor cohort."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nag/scripts'))
import upright_sampling_floor_validation as driver


class UprightSamplingFloorDriverTests(unittest.TestCase):
    def test_paths_are_separate_from_legacy_floors(self):
        root = Path('/tmp/ellipsoids')
        source, output = driver.paths(root, 'W254_86_137')
        self.assertTrue(str(source).endswith('frames_086_137_reprojection_depth_v1/ellipsoid.json'))
        self.assertTrue(str(output).endswith('frames_086_137_reprojection_depth_sampling_floor_world_y_v2'))
        self.assertNotEqual(output / 'ellipsoid.json', driver.legacy_floor_path(root, 'W254_86_137'))

    def test_geometry_comparison_allows_runtime_float32_rounding(self):
        a = {'fit': {'center': [1., 2., 3.], 'radii': [4., 5., 6.],
                     'rotation_columns': np.eye(3).tolist()}}
        b = {'fit': {'center': [1.00001, 2., 3.], 'radii': [4., 5., 6.],
                     'rotation_columns': np.eye(3).astype(np.float32).tolist()}}
        self.assertTrue(driver._same_geometry(a, b))

    def test_training_dry_run_emits_explicit_world_y_seeded_jobs(self):
        label = 'W254_86_137'
        segment, first, stop = driver.SEQUENCES[label]
        source_doc = {'sequence': f'segment-{segment}',
                      'frame_slice': {'start': first, 'stop': stop, 'step': 1}}
        geometry_doc = {**source_doc, 'fit': {'regularizer_version': driver.VERSION,
                        'center': [0., 0., 0.], 'radii': [20., 20., 20.],
                        'rotation_columns': np.eye(3).tolist()}}
        report = {'full_final': {'weighted_data_loss': 1.},
                  'full_original': {'weighted_data_loss': 1.},
                  'final_mapping': {'mode': 'world_y', 'u_offset': 0.}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'ellipsoids'
            source, folder = driver.paths(root, label)
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(source_doc))
            folder.mkdir()
            (folder / 'ellipsoid.json').write_text(json.dumps(geometry_doc))
            (folder / 'sampling_floor_diagnostics.json').write_text(json.dumps(report))
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['driver', 'train', '--sequences', label,
                    '--ellipsoid-root', str(root), '--output-root', str(Path(directory) / 'runs'),
                    '--dry-run']), contextlib.redirect_stdout(stream):
                driver.main()
        text = stream.getvalue()
        self.assertEqual(text.count('--background-ellipsoid-texture-orientation world_y'), 2)
        self.assertEqual(text.count('--seed 42'), 2)
        self.assertIn('2 new training jobs', text)


if __name__ == '__main__':
    unittest.main()
