"""Small CPU tests; never run the full frozen-data experiment in unit tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'nag'))
import controlled_floor_replay as replay
from reprojection_ellipsoid import ReprojectionFitConfig,enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs


class ControlledReplayTests(unittest.TestCase):
    def test_balanced_fresh_process_schedule(self):
        jobs=replay.schedule(['W191_0_40'],3)
        self.assertEqual(len(jobs),9)
        self.assertEqual(len({j['id']for j in jobs}),9)
        for i in range(3):self.assertEqual(jobs[i*3]['arm'],replay.ARMS[i])
        for arm in replay.ARMS:self.assertEqual(sum(j['arm']==arm for j in jobs),3)

    def test_historical_kernel_hashes_and_dependencies(self):
        import hashlib
        sources=replay.historical_sources()
        self.assertEqual(hashlib.sha256(sources['historical_v1']).hexdigest(),
                         '2532348a6f10fe6bf981f3b6725da56e4f7478c784b557d529ac04e2ad99e278')
        self.assertEqual(sources['historical_v2'],(REPO/'nag/preparation/_legacy_floor_kernel.py').read_bytes())

    def test_all_engines_accept_the_same_synthetic_inputs(self):
        pairs,g=synthetic_pairs()
        cfg=ReprojectionFitConfig(max_iterations=1,max_pairs=128,max_depth_rays=128,
            minimum_camera_surface_distance=3.,minimum_axis_radius=5.)
        g=enforce_safety(g,pairs['poses'],cfg)
        row={'sphere':{k:v.tolist()for k,v in g.items()},
             'mapping':{'resolution_hw':[640,960],'visible_uv':{'enabled':False}},'size':[960,640]}
        functions=replay.engines()
        snapshot={k:v.copy()for k,v in pairs.items()}
        results={a:replay.fit_arm(functions,a,pairs,g,row,cfg) for a in replay.ARMS}
        for k in pairs:np.testing.assert_array_equal(pairs[k],snapshot[k])
        self.assertTrue(results['historical_v2']['sampling_floor']['zero_floor_preserved'])
        self.assertTrue(results['generalized_v2']['sampling_floor']['zero_floor_preserved'])
        self.assertTrue(replay.difference(results['historical_v2']['geometry'],results['generalized_v2']['geometry'])['exact'])

    def test_array_fingerprints_detect_reordering_and_dtype_changes(self):
        g={'center':np.zeros(3),'radii':np.ones(3),'rotation':np.eye(3)}
        row={'sphere':{},'mapping':{},'size':[960,640]}
        a={'indices':np.array([0,1],dtype=np.int32)}
        b={'indices':a['indices'][::-1]}
        c={'indices':a['indices'].astype(np.int64)}
        values=[replay.digest(replay.input_fingerprint(p,g,row,{}))for p in [a,b,c]]
        self.assertEqual(len(set(values)),3)

    def test_existing_tolerance_is_not_widened(self):
        a={'center':[0,0,0],'radii':[10,10,10],'rotation':np.eye(3).tolist()}
        self.assertTrue(replay.difference(a,copy.deepcopy(a))['exact'])
        b=copy.deepcopy(a);b['radii'][0]+=.00053
        self.assertFalse(replay.difference(a,b)['passes_existing_tolerance'])

    def test_summary_separates_within_arm_between_arm_and_stored_error(self):
        with tempfile.TemporaryDirectory(prefix='autowork_replay_test_') as d:
            p=Path(d)/'stored.json';p.write_text(json.dumps({'fit':{'sampling_floor':{'final_combined_objective':1.}}}))
            g={'center':[0,0,0],'radii':[10,10,10],'rotation':np.eye(3).tolist()}
            records=[{'job':j,'input_digest':'same','geometry':g,'versus_stored':replay.difference(g,g),
                      'fit':{'sampling_floor':{'final_combined_objective':1.}}}for j in replay.schedule(['test'],3)]
            summary=replay.summarize(records,[{'label':'test','floor':str(p)}])['test']
            self.assertEqual(summary['within_arm']['historical_v1']['exact_pair_count'],3)
            self.assertEqual(summary['between_arms']['historical_v2_vs_generalized_v2']['pair_count'],9)
            self.assertEqual(len(summary['versus_stored']),9)
            records[-1]['input_digest']='changed'
            with self.assertRaises(ValueError):replay.summarize(records,[{'label':'test','floor':str(p)}])

    def test_preserved_interval_is_not_mislabeled_as_v1_refinement(self):
        with tempfile.TemporaryDirectory(prefix='autowork_replay_test_') as d:
            p=Path(d)/'fixture.json';p.write_text(json.dumps({'rows':[{'label':'test','floor_version':2}]}))
            with self.assertRaises(ValueError):replay.load_rows(p,['test'])

    def test_missing_or_modified_frozen_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='autowork_replay_test_') as d:
            p=Path(d)/'fixture.json';p.write_text(json.dumps({'files':{},'rows':[{'label':'test','floor_version':1,
                'pairs':'missing','reprojection':'missing','floor':'missing','sphere':{'provenance':'missing'},'sphere_run':d}]}))
            with self.assertRaises(ValueError):replay.load_rows(p,['test'])


if __name__=='__main__':unittest.main()
