"""Completed-cohort reporting checks; no training, model or fitting imports."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reports'))
import generate_main_background_statistics as report


def records():
    ego=['W191_0_40','W191_40_128','W191_128_198','W254_0_37','W254_37_81','W254_86_137','W254_139_195']
    final=['W125_0_40','W125_40_93','W125_93_124','W125_124_149','W141_2_53','W141_53_101',
        'W141_102_173','W141_174_197','W203_3_58','W203_60_107','W324_0_42','W324_42_96',
        'W324_161_197','W344_0_51','W344_52_95','W344_95_135','W344_135_197','W952_0_63',
        'W952_64_140','W952_141_198','W975_0_68','W975_69_99','W975_99_162','W975_175_195']
    return [dict(group=g,label=l,segment='segment-'+l.split('_')[0][1:]+'0000000000',
        treatment=t,repeat=n,frames=20,seed=42,workers=10,mean=[20+n,.8+n*.001,.2-n*.001],
        frame_sd=[.5,.01,.02],path='/example/'+l+'/'+t+'/'+str(n),physical_path='/example/'+l+'/'+t+'/'+str(n))
        for g,labels in [('ego',ego),('final',final)] for l in labels for t in report.base.TREATMENTS for n in (1,2,3)]


class MainStatisticsTests(unittest.TestCase):
    def test_restored_manifests_are_added_without_changing_old_list(self):
        original=report.base.manifests(REPO)
        full=report.main_manifests(REPO)
        self.assertEqual(full[:len(original)],original)
        self.assertEqual(len(full),len(original)+4)
        self.assertEqual(report.base.OMITTED,set(report.RESTORED)|{report.EXCLUSION})

    def test_full_cohort_has_all_restored_and_no_resource_exclusion(self):
        groups=report.aggregate_main(records())
        self.assertEqual(len(groups['final']['intervals']),24)
        self.assertEqual(len(groups['ego']['intervals']),7)
        self.assertTrue(set(report.RESTORED)<=set(groups['final']['intervals']))

    def test_missing_repeat_duplicate_and_substituted_restored_interval_fail(self):
        rows=records()
        bad=copy.deepcopy(rows)
        for r in bad:
            if r['label']=='W975_99_162':r['label']='W975_999_1000'
        for sample in (rows[:-1],rows+[rows[0]],bad):
            with self.assertRaises(ValueError):report.aggregate_main(sample)

    def test_existing_output_rejected_before_collect(self):
        with tempfile.TemporaryDirectory(prefix='autowork_main_test_') as d,patch.object(sys,'argv',['main','--output-root',d]), \
                patch.object(report,'collect_main') as collect,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            report.main()
        collect.assert_not_called()

    def test_compact_triple_keeps_requested_metric_order(self):
        self.assertEqual(report.triple({'mean':[33.12345,.912345,.123456]}),'33.123 / 0.91234 / 0.12346')

    def test_paired_wins_respect_lpips_direction(self):
        rows=records()
        for r in rows:
            if r['treatment']=='ellipsoid_regular':r['mean']=[x+d for x,d in zip(r['mean'],[1,.01,-.02])]
        groups=report.aggregate_main(rows)
        found=[x for x in report.comparisons(groups) if x['group']=='ego' and x['candidate']=='ellipsoid_regular' and x['reference']=='plane']
        self.assertEqual([x['wins'] for x in found],[7,7,7])
        self.assertAlmostEqual(found[2]['mean_delta'],-.02)

    def test_c_yaml_matches_historical_safe_parser(self):
        text='NAGConfig:\n  seed: 42\n  deterministic: false\n  radii: [1.25, 3.2e-05, null]\n  path: {value: /mnt/d/test}\n'
        self.assertEqual(report.safe_yaml(text),yaml.safe_load(text))

    def test_complete_text_and_pdf_are_not_preliminary_or_pending(self):
        rows=records();groups=report.aggregate_main(rows)
        provenance={'incomplete_attempts':[],'files_sha256':{}}
        text=report.text_report(groups,rows,provenance,'2026-11-01')
        self.assertIn('465 completed',text);self.assertIn('W975_99_162',text)
        self.assertNotIn('trainings remain pending',text)
        with tempfile.TemporaryDirectory(prefix='autowork_main_pdf_test_') as d:
            path=Path(d)/'main.pdf'
            report.pdf_report(path,groups,rows,'2026-11-01')
            with pymupdf.open(path) as doc:
                self.assertGreater(len(doc),35)
                for page in doc:
                    self.assertIn('MAIN BENCHMARK COMPLETE',page.get_text())
                    self.assertNotIn('PRELIMINARY',page.get_text())
                    for block in page.get_text('dict')['blocks']:
                        for line in block.get('lines',[]):
                            for span in line['spans']:
                                self.assertEqual(span['color'],0)
                                self.assertTrue(page.rect.contains(pymupdf.Rect(span['bbox'])))


if __name__=='__main__':unittest.main()
