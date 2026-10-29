"""Pure reporting statistics and vector-table tests; no model/training imports."""
import contextlib
import copy
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preliminary_stats', REPO / 'reports/generate_preliminary_background_statistics.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def records():
    return [dict(group='ego', label=l, segment='segment-'+s, treatment=t, repeat=r, frames=f,
                 seed=42, workers=10, mean=[b+r, .8+r*.001, .2-r*.001],
                 frame_sd=[float(r), .01*r, .02*r])
            for l,s,b,f in [('W191_0_40','1918764220984209654',20.,41),
                            ('W191_40_128','1918764220984209654',20.,89),
                            ('W254_0_37','2547899409721197155',50.,38)]
            for t in report.TREATMENTS for r in (1,2,3)]


class PreliminaryStatisticsTests(unittest.TestCase):
    def test_frame_sd_uses_population_divisor(self):
        mean, sd = report.frame_statistics('[1 2\n 3]', 3)
        self.assertEqual(mean, 2.)
        self.assertAlmostEqual(sd, np.sqrt(2/3))
        self.assertNotAlmostEqual(sd, 1.)

    def test_truncated_nonfinite_and_wrong_length_vectors_fail(self):
        for text,n in [('[1 ... 3]',3),('[1 nan 3]',3),('[1 inf 3]',3),
                       ('[1 2]',3),('[1]',1),('1 2 3',3),('[1, 2, 3]',3)]:
            with self.assertRaises(ValueError): report.frame_statistics(text,n)

    def test_interval_frame_sd_and_repeat_sd_are_separate(self):
        g=report.aggregate(records(),{'ego':3})['ego']
        v=g['intervals']['W191_0_40']['plane']
        self.assertEqual(v['mean'][0],22.)
        self.assertEqual(v['frame_sd'][0],2.)
        self.assertEqual(v['repeat_sd'][0],1.)
        self.assertEqual(g['interval_macro']['plane']['mean'][0],32.)
        self.assertEqual(g['scene_macro']['plane']['mean'][0],37.)
        self.assertEqual(g['interval_macro']['plane']['frame_sd'][0],2.)

    def test_duplicate_missing_or_wrong_seed_runs_fail(self):
        source=records()
        badseed=copy.deepcopy(source);badseed[0]['seed']=43
        badworker=copy.deepcopy(source);badworker[0]['workers']=4
        for rows in [source+[source[0]],source[:-1],badseed,badworker]:
            with self.assertRaises(ValueError):report.aggregate(rows,{'ego':3})

    def test_conflicting_frame_identity_fails(self):
        rows=records();rows[0]['frames']=2
        with self.assertRaises(ValueError):report.aggregate(rows,{'ego':3})

    def test_saved_means_are_not_replaced_with_pooled_psnr(self):
        g=report.aggregate(records(),{'ego':3})['ego']
        expected=(22+22+52)/3
        self.assertEqual(g['interval_macro']['plane']['mean'][0],expected)

    def test_completion_requires_success_coverage_and_epoch80(self):
        with tempfile.TemporaryDirectory(prefix='autowork_complete_test_') as d:
            root=Path(d);(root/'final/complete').mkdir(parents=True);(root/'tracker').mkdir()
            (root/'exit_000.txt').write_text('Exit code: 0.\n')
            for n in (0,1):(root/'final/complete'/f'{n:02d}_t.png').write_bytes(b'not decoded by report')
            for m in report.METRICS:
                with (root/'tracker'/f'train_epoch_Mean{m}.csv').open('w') as f:
                    f.write('step;value\n80;1\n')
            self.assertEqual(report.complete_means(root,2),[1.,1.,1.])
            (root/'exit_000.txt').write_text('Exit code: 1.\n')
            self.assertIsNone(report.complete_means(root,2))
            (root/'exit_000.txt').write_text('Exit code: 0.\n')
            (root/'tracker/train_epoch_MeanSSIM.csv').write_text('step;value\n79;1\n')
            self.assertIsNone(report.complete_means(root,2))

    def test_effective_config_accepts_only_expected_layout_and_budget(self):
        c=dict(background_plane_type=report.SPHERE,seed=42,num_workers=10,max_epochs=80,
               background_sphere_radius_from_visible_plane_centers=True,
               background_plane_kwargs={'color_mask_resolution':[960,1920]})
        report.normalized_config(c,'sphere_regular',REPO)
        bad=copy.deepcopy(c);bad['background_plane_kwargs']['color_mask_resolution']=[1920,3840]
        with self.assertRaises(ValueError):report.normalized_config(bad,'sphere_regular',REPO)
        bad=copy.deepcopy(c);bad['background_plane_kwargs']['sphere_visible_uv_texture']=report.UV
        with self.assertRaises(ValueError):report.normalized_config(bad,'sphere_regular',REPO)

    def test_existing_output_is_rejected_before_reading_runs(self):
        with tempfile.TemporaryDirectory(prefix='autowork_report_test_') as d, \
                patch.object(sys,'argv',['report','--output-root',d]), \
                patch.object(report,'collect') as collect, \
                contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            report.main()
        collect.assert_not_called()

    def test_pdf_preliminary_marker_plusminus_and_page_bounds(self):
        with tempfile.TemporaryDirectory(prefix='autowork_pdf_test_') as d:
            path=Path(d)/'test.pdf';p=report.PDF('2026-10-29')
            p.page('Test table','Frame SD and repeat SD are different.')
            p.table(['Name','PSNR','SSIM','LPIPS'],[['Plane','35.075 ± 0.142','0.94749 ± 0.00065','0.20431 ± 0.00149']],
                    [206,192,192,192])
            p.save(path)
            with pymupdf.open(path) as doc:
                text=doc[0].get_text()
                self.assertIn('PRELIMINARY',text);self.assertIn('±',text)
                for block in doc[0].get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line['spans']:
                            self.assertEqual(span['color'],0)
                            self.assertTrue(doc[0].rect.contains(pymupdf.Rect(span['bbox'])))

    def test_table_and_paragraph_overflow_is_not_silently_clipped(self):
        p=report.PDF('2026-10-29');p.page('Bounds')
        with self.assertRaises(ValueError):p.table(['a','b'],[],[1,1])
        p.y=550
        with self.assertRaises(ValueError):p.para('too tall')
        with self.assertRaises(ValueError):p.table(['a','b'],[['c','d']],[391,391])
        p.doc.close()


if __name__=='__main__':unittest.main()
