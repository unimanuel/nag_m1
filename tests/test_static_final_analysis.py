"""CPU-only aggregation and launcher-selection checks for the static cohort."""
import copy
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'reports'))
import analyze_static_final_ellipsoids as analysis


def rows():
    return [{'label': label, 'mode': mode, 'repeat': repeat, 'workers': 10, 'seed': 42,
             'MPSNR': base + repeat + (1 if mode == 'visuv' else 0),
             'SSIM': .8 + (.01 if mode == 'visuv' else 0),
             'LPIPS': .3 - (.02 if mode == 'visuv' else 0), 'frames': frames}
            for label, base, frames in [('W125_0_40', 20., 41), ('W952_64_140', 30., 77)]
            for mode in ('regular', 'visuv') for repeat in (1, 2, 3)]


class StaticFinalAnalysisTests(unittest.TestCase):
    def test_repeat_means_precede_equal_interval_macro(self):
        result = analysis.aggregate(rows(), {'W125_0_40', 'W952_64_140'})
        self.assertEqual(result['macro']['regular']['MPSNR'], 27.)
        self.assertEqual(result['macro']['visuv']['MPSNR'], 28.)
        self.assertEqual(result['intervals'][0]['regular_sd']['MPSNR'], 1.)
        self.assertEqual(result['visible_uv_interval_wins'], {'MPSNR': 2, 'SSIM': 2, 'LPIPS': 2})

    def test_missing_duplicate_and_unexpected_repeats_are_rejected(self):
        source = rows()
        for bad in [source[:-1], source + [source[0]],
                    [dict(r, repeat=4) if i == 0 else r for i, r in enumerate(source)]]:
            with self.assertRaises(ValueError):
                analysis.aggregate(bad, {'W125_0_40', 'W952_64_140'})

    def test_wrong_worker_seed_and_nonfinite_metric_are_rejected(self):
        for field, value in [('workers', 4), ('seed', 43), ('MPSNR', math.nan)]:
            bad = copy.deepcopy(rows()); bad[0][field] = value
            with self.assertRaises(ValueError):
                analysis.aggregate(bad, {'W125_0_40', 'W952_64_140'})

    def test_numeric_interval_order_and_delta_sign(self):
        result = analysis.aggregate(rows(), {'W952_64_140', 'W125_0_40'})
        self.assertEqual(result['intervals'][0]['label'], 'W125_0_40')
        delta = result['intervals'][0]['delta_uv_minus_regular']
        self.assertEqual(delta['MPSNR'], 1.)
        self.assertAlmostEqual(delta['LPIPS'], -.02)

    def test_documented_queues_cover_exactly_twenty_intervals_and_180_jobs(self):
        # Replace only the executable with echo: exercise real shell forwarding
        # without Python training, dataset writes, locks or result directories.
        expected = {
            'W125': {'W125_0_40', 'W125_40_93', 'W125_93_124', 'W125_124_149'},
            'W141': {'W141_2_53', 'W141_53_101', 'W141_102_173', 'W141_174_197'},
            'W203_ready': {'W203_3_58'},
            'W324': {'W324_0_42', 'W324_42_96', 'W324_161_197'},
            'W344': {'W344_0_51', 'W344_52_95', 'W344_95_135', 'W344_135_197'},
            'W952': {'W952_0_63', 'W952_64_140', 'W952_141_198'},
            'W975_ready': {'W975_175_195'},
        }
        seen = set()
        for suffix, labels in expected.items():
            script = REPO / f'run_paper_final_baselines_{suffix}.sh'
            text = subprocess.check_output(['bash', str(script), '--max-hours', '3'],
                env={**os.environ, 'PYTHON_BIN': '/bin/echo', 'DRY_RUN': '1'}, text=True)
            tokens = shlex.split(text)
            self.assertIn('--dry-run', tokens)
            self.assertEqual(tokens[-2:], ['--max-hours', '3'])
            if '--intervals' in tokens:
                pos = tokens.index('--intervals') + 1
                selected = []
                while pos < len(tokens) and not tokens[pos].startswith('--'):
                    selected.append(tokens[pos]); pos += 1
                self.assertEqual(set(selected), labels)
            else:
                self.assertEqual(tokens[tokens.index('--segment') + 1], suffix)
            self.assertFalse(seen & labels)
            self.assertFalse(labels & analysis.OMITTED)
            seen.update(labels)
        self.assertEqual(len(seen), 20)
        self.assertEqual(len(seen) * 9, 180)


if __name__ == '__main__':
    unittest.main()
