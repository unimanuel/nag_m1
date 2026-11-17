"""CPU-only tests for presentation comparisons; no NAG model imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'nag/scripts/make_result_comparison_video.py'
SPEC = importlib.util.spec_from_file_location('comparison_video', SCRIPT)
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='autowork_video_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_fixture(self, name='left', psnr=24.3):
        p = self.root / name
        (p / 'final').mkdir(parents=True)
        (p / 'tracker').mkdir()
        config = {'frame_indices_filter': {'start': 86, 'stop': 89, 'step': None},
                  'data_path': {'value': {'value': 'data/waymo/segment-2547899409721197155'}},
                  'images_path': {'value': 'data/waymo/segment-2547899409721197155/images_undistorted'},
                  'max_epochs': 80, 'seed': 42, 'background_plane_type': 'example.Plane',
                  'final_plot_resolution_factor': .5}
        (p / 'training_config.yaml').write_text(yaml.safe_dump({'NAGConfig': config}))
        (p / 'exit_000.txt').write_text('Exit code: 0.\n')
        (p / 'final/complete_scene.mp4').write_bytes(b'video fixture')
        (p / 'tracker/train_epoch_MeanMPSNR.csv').write_text(f'step;value;global_step\n80;{psnr};80\n')
        return p

    def info(self, name='left'):
        result = video.read_run(self.run_fixture(name), name.upper())
        result['probe'] = {'width': 960, 'height': 640, 'frames': 3, 'fps': '5'}
        return result

    def test_run_metric_and_explicit_frame_identity(self):
        result = self.info()
        self.assertEqual(result['frames'], [86, 87, 88])
        self.assertEqual(result['psnr'], 24.3)
        self.assertEqual(result['segment'], 'segment-2547899409721197155')

    def test_mismatched_scene_interval_source_or_count_fails(self):
        left = self.info()
        for field, value in [('segment', 'segment-999'), ('frames', [0, 1, 2]),
                             ('image_source', 'other/images'), ('epoch', 79)]:
            right = copy.deepcopy(left)
            right[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                video.validate_pair(left, right, None)
        right = copy.deepcopy(left)
        right['probe']['frames'] = 2
        with self.assertRaises(ValueError):
            video.validate_pair(left, right, None)

    def test_fps_preserved_or_explicitly_retimed(self):
        left = self.info()
        right = copy.deepcopy(left)
        self.assertEqual(str(video.validate_pair(left, right, None)), '5')
        right['probe']['fps'] = '10'
        with self.assertRaises(ValueError):
            video.validate_pair(left, right, None)
        self.assertEqual(str(video.validate_pair(left, right, 10)), '10')
        for fps in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                video.validate_pair(left, left, fps)

    def test_missing_final_epoch_duplicate_and_nonfinite_metric_fail(self):
        p = self.root / 'metric.csv'
        for rows in ['79;24;79\n', '80;24;80\n80;25;80\n', '80;nan;80\n']:
            p.write_text('step;value;global_step\n' + rows)
            with self.assertRaises(ValueError):
                video.final_psnr(p, 80)

    def test_failed_run_rejected(self):
        p = self.run_fixture()
        (p / 'exit_000.txt').write_text('Exit code: 1.\n')
        with self.assertRaises(ValueError):
            video.read_run(p, 'PLANE')

    def test_header_handles_literal_titles(self):
        p = self.root / 'header.png'
        video.make_header(p, ["PLANE: A's 100%", 'ELLIPSOID | [test]'], 480, 64, video.choose_font(None))
        with Image.open(p) as image:
            self.assertEqual(image.size, (960, 64))

    def test_dry_run_writes_nothing_and_uses_each_runs_metric(self):
        left, right = self.run_fixture('left', 24.3), self.run_fixture('right', 27.6)
        output = self.root / 'absent_output_dir' / 'test.mp4'
        probe = {'width': 960, 'height': 640, 'frames': 3, 'fps': '5'}
        with patch.object(video, 'executable', return_value='/test/ffmpeg'), \
             patch.object(video, 'probe', return_value=probe), patch.object(video, 'print'):
            plan = video.main(['--left-run', str(left), '--right-run', str(right), '--output', str(output), '--dry-run'])
        self.assertFalse(output.parent.exists())
        self.assertIn('24.3000', plan['labels'][0])
        self.assertIn('27.6000', plan['labels'][1])
        self.assertEqual(plan['playback_seconds'], .6)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / 'result.mp4'
        output.write_bytes(b'keep')
        with self.assertRaises(ValueError):
            video.main(['--left-run', 'x', '--right-run', 'y', '--output', str(output)])
        self.assertEqual(output.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
