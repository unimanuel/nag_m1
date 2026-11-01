"""Synthetic fitting and read-only launcher/retry tests; no real training."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'reports'),str(REPO/'nag'),str(REPO/'tests')]
import paper_ablation_kernel as adapter
import paper_loss_ablations as driver
from reprojection_ellipsoid import ReprojectionFitConfig,fit_reprojection_depth_ellipsoid
from test_reprojection_ellipsoid import synthetic_pairs


def plan():
    return dict(intervals=[dict(label=l,D1_geometry='/missing/d1.json',D2_geometry='/missing/d2.json',
        D2_runs=['/main/one','/main/two','/main/three'],D1_same_as_D2=i>=2,
        estimated_training_seconds=1200,pins={}) for i,l in enumerate(driver.ABLATION_INTERVALS)],
        recipes={'D0':{**asdict(ReprojectionFitConfig()),'reprojection_weight':0.}})


class AblationKernelTests(unittest.TestCase):
    def test_adapter_changes_only_the_expected_validation(self):
        raw=adapter.SOURCE.read_bytes();text=raw.decode()
        self.assertEqual(adapter.ablation_source(raw).replace(adapter.NEW,adapter.OLD,1),text)
        with self.assertRaises(ValueError):adapter.ablation_source(raw+b'\n')

    def test_released_fitter_still_rejects_zero(self):
        pairs,g=synthetic_pairs()
        with self.assertRaises(ValueError):fit_reprojection_depth_ellipsoid(pairs,g,ReprojectionFitConfig(reprojection_weight=0))

    def test_positive_mode_matches_released_numerics_exactly(self):
        pairs,g=synthetic_pairs();c=ReprojectionFitConfig(max_pairs=128,max_depth_rays=128,max_iterations=2)
        expected=fit_reprojection_depth_ellipsoid(pairs,g,c)
        actual=adapter.kernel().fit_reprojection_depth_ellipsoid(pairs,g,c)
        for k in g:np.testing.assert_array_equal(expected['geometry'][k],actual['geometry'][k])
        self.assertEqual(expected['candidates'],actual['candidates'])
        self.assertEqual(expected['diagnostics'],actual['diagnostics'])

    def test_depth_only_is_exactly_zero_not_an_epsilon(self):
        module=adapter.kernel();pairs,g=synthetic_pairs()
        def fake_minimize(fun,x,*args,**kwargs):
            return SimpleNamespace(x=x,fun=fun(x),nit=0,success=True,message='unit-test fixed candidates')
        def residual(error):
            def call(p,g,indices):
                n=len(indices)
                return dict(valid=np.ones(n,bool),pixel_error=np.full(n,error),camera_angle_deg=np.zeros(n))
            return call
        cfg=ReprojectionFitConfig(reprojection_weight=0,max_pairs=128,max_depth_rays=128,max_iterations=2)
        with patch.object(module,'minimize',fake_minimize),patch.object(module,'_pair_residuals',residual(1.)):
            first=adapter.fit_depth_only(pairs,g,cfg)
        with patch.object(module,'minimize',fake_minimize),patch.object(module,'_pair_residuals',residual(1e8)):
            second=adapter.fit_depth_only(pairs,g,cfg)
        self.assertEqual(first['candidates'],second['candidates'])
        self.assertEqual(first['config']['reprojection_weight'],0)
        for k in g:np.testing.assert_array_equal(first['geometry'][k],second['geometry'][k])

    def test_depth_only_rejects_nonzero_or_excess_regularizers(self):
        pairs,g=synthetic_pairs()
        for cfg in (ReprojectionFitConfig(),ReprojectionFitConfig(reprojection_weight=0,radius_excess_weight=1)):
            with self.assertRaises(ValueError):adapter.fit_depth_only(pairs,g,cfg)


class AblationLauncherTests(unittest.TestCase):
    def test_live_tee_preserves_progress_stderr_and_exit_status(self):
        import subprocess
        real_popen=subprocess.Popen
        processes=[];seen_alive=[]
        def spawn(*args,**kwargs):
            p=real_popen(*args,**kwargs);processes.append(p);return p
        class Console(io.StringIO):
            def write(self,text):
                if 'first' in text:seen_alive.append(processes[0].poll() is None)
                return super().write(text)
        script="import os,sys,time; sys.stdout.write('first\\r'); time.sleep(.3); os.write(1,b'\\xe2'); time.sleep(.05); os.write(1,b'\\x82\\xac\\n'); sys.stderr.write('error\\n'); sys.exit(3)"
        with tempfile.TemporaryDirectory(prefix='autowork_console_test_') as d:
            path=Path(d)/'console.log';console=Console()
            with patch.object(driver.subprocess,'Popen',spawn),contextlib.redirect_stdout(console):
                code=driver.stream_training([sys.executable,'-c',script],path)
            self.assertEqual(code,3)
            self.assertEqual(console.getvalue(),'first\r€\nerror\n')
            self.assertEqual(path.read_bytes(),'first\r€\nerror\n'.encode())
            self.assertEqual(seen_alive,[True])

    def test_terminal_tee_keeps_child_stdout_a_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):return True
            def fileno(self):raise OSError('unit-test terminal has no real console fd')
        with tempfile.TemporaryDirectory(prefix='autowork_console_test_') as d:
            path=Path(d)/'console.log';console=Terminal()
            with contextlib.redirect_stdout(console):
                code=driver.stream_training([sys.executable,'-c',"import sys; print('TTY',sys.stdout.isatty()); print('stderr',file=sys.stderr)"],path)
            self.assertEqual(code,0)
            self.assertIn('TTY True',console.getvalue());self.assertIn('stderr',console.getvalue())
            self.assertIn(b'TTY True',path.read_bytes())

    def test_existing_console_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='autowork_console_test_') as d:
            path=Path(d)/'console.log';path.write_bytes(b'keep previous output')
            with patch.object(driver.subprocess,'Popen') as spawn,self.assertRaises(FileExistsError):
                driver.stream_training(['unused'],path)
            spawn.assert_not_called();self.assertEqual(path.read_bytes(),b'keep previous output')

    def test_console_compatibility_is_exact_and_launcher_only(self):
        with tempfile.TemporaryDirectory(prefix='autowork_console_test_') as d:
            ledger=Path(d)/'ledger.json';source=Path(driver.__file__)
            data=dict(scope='console_streaming_only_v1',before=driver.ORIGINAL_LAUNCHER_SHA,after=driver.sha(source))
            ledger.write_text(json.dumps(data))
            with patch.object(driver,'CONSOLE_LEDGER',ledger):
                self.assertEqual(driver.code_identity(source),driver.ORIGINAL_LAUNCHER_SHA)
                driver.assert_pins({str(source):driver.ORIGINAL_LAUNCHER_SHA})
                other=Path(d)/'other.py';other.write_text('not the launcher')
                with self.assertRaises(ValueError):driver.assert_pins({str(other):driver.ORIGINAL_LAUNCHER_SHA})
                for bad in ({**data,'scope':'anything'}, {**data,'before':'0'*64}, {**data,'after':'0'*64}):
                    ledger.write_text(json.dumps(bad))
                    self.assertEqual(driver.code_identity(source),driver.sha(source))

    def test_matrix_uses_exact_geometry_reuse_not_quality(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            cells=driver.matrix(plan(),Path(d))
        self.assertEqual(len(cells),45)
        self.assertEqual(sum(c['reused_run'] is None for c in cells),21)
        self.assertEqual(sum(c['recipe']=='D0' and c['reused_run'] is None for c in cells),15)
        self.assertEqual(sum(c['recipe']=='D1' and c['reused_run'] is not None for c in cells),9)
        self.assertEqual([c['reused_run'] for c in cells if c['label']=='W254_86_137' and c['recipe']=='D1'],
                         ['/main/one','/main/two','/main/three'])

    def test_float32_runtime_equality_requires_center_radii_and_rotation(self):
        a=dict(center=[0.,1.,2.],radii=[10.,20.,30.],rotation_columns=np.eye(3))
        b={**a,'radii':[10.+1e-13,20.,30.]}
        self.assertTrue(driver.same_runtime(a,b))
        self.assertFalse(driver.same_runtime(a,{**a,'radii':[10.01,20.,30.]}))
        self.assertFalse(driver.same_runtime(a,{**a,'center':[.1,1.,2.]}))
        self.assertFalse(driver.same_runtime(a,{**a,'rotation_columns':-np.eye(3)}))

    def test_old_original_config_allows_only_known_absent_zero_fields(self):
        c=asdict(ReprojectionFitConfig());del c['radius_excess_weight'];del c['radius_excess_huber_log']
        self.assertEqual(driver.original_config(c),asdict(ReprojectionFitConfig()))
        for bad in ({**c,'depth_weight':1.},{k:v for k,v in c.items() if k!='max_pairs'},{**c,'radius_excess_weight':1.}):
            with self.assertRaises(ValueError):driver.original_config(bad)

    def test_missing_preparation_fails_without_creating_paths(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            p=Path(d)/'autowork_not_prepared'
            with self.assertRaises(ValueError):driver.matrix(plan(),p,True)
            self.assertFalse(p.exists())

    def test_unowned_and_conflicting_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            r=Path(d)/'new'
            driver.freeze(r,{'x':1},dry_run=True);self.assertFalse(r.exists())
            r.mkdir();(r/'user.txt').write_text('keep')
            with self.assertRaises(ValueError):driver.freeze(r,{'x':1})
            self.assertEqual((r/'user.txt').read_text(),'keep')
            other=Path(d)/'owned';driver.freeze(other,{'x':1})
            with self.assertRaises(ValueError):driver.freeze(other,{'x':2})
            self.assertEqual(driver.load(other/'manifest.json'),{'x':1})

    def test_failed_launch_records_reserve_their_attempt_number(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            r=Path(d)
            self.assertEqual(driver.next_attempt(r),1)
            (r/'attempt_001.launch.json').write_text('{}')
            self.assertEqual(driver.next_attempt(r),2)
            (r/'attempt_002').mkdir()
            self.assertEqual(driver.next_attempt(r),3)

    def test_budget_accounts_for_next_job_and_never_kills_an_active_job(self):
        self.assertTrue(driver.fits_budget(100,200,1))
        self.assertFalse(driver.fits_budget(3500,200,1))
        self.assertFalse(driver.fits_budget(0,4000,1))

    def test_command_keeps_all_inputs_and_does_not_enable_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            cfg=Path(d)/'source.yaml';cfg.write_text('NAGConfig:\n  save_intermediate_checkpoints: false\n')
            item=dict(config=str(cfg),data_dir='/data/segment-191',pose_path='/data/segment-191/calibration.json')
            cell=dict(id='test',geometry='/geometry/ellipsoid.json')
            cmd=driver.command(item,cell,Path(d)/'run')
            self.assertEqual(cmd[cmd.index('--mask-ids-filter-path')+1],'/data/segment-191/mask_indices_filter.json')
            self.assertEqual(cmd[cmd.index('--num-workers')+1],'10')
            self.assertEqual(cmd[cmd.index('--background-ellipsoid-scale-margin')+1],'0')
            self.assertNotIn('--save-intermediate-checkpoints',cmd)
            self.assertNotIn('--checkpoint-path',cmd)
            cfg.write_text('NAGConfig:\n  save_intermediate_checkpoints: true\n')
            with self.assertRaises(ValueError):driver.command(item,cell,Path(d)/'run')

    def test_prepare_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_test_') as d:
            root=Path(d)/'autowork_prepare';train=Path(d)/'autowork_train'
            args=['paper_loss_ablations','prepare','--dry-run','--preparation-root',str(root),'--training-root',str(train)]
            with patch.object(sys,'argv',args),patch.object(driver,'inventory',return_value=plan()),\
                    patch.object(driver,'prepare') as prepare,contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            prepare.assert_not_called();self.assertFalse(root.exists());self.assertFalse(train.exists())

    def test_synthetic_preparation_exports_and_reuses_without_training(self):
        pairs,g=synthetic_pairs();pairs['frames']=np.arange(3)
        fit=adapter.fit_depth_only(pairs,g,ReprojectionFitConfig(reprojection_weight=0,max_iterations=2))
        fit['config']['max_iterations']=200  # mock's planned run metadata
        with tempfile.TemporaryDirectory(prefix='autowork_ablation_smoke_') as d:
            root=Path(d);source=root/'source';source.mkdir()
            raw=source/'ellipsoid.json'
            raw.write_text(json.dumps(dict(sequence='segment-test',frame_slice=dict(start=0,stop=3,step=1),
                fit=dict(center=g['center'].tolist(),radii=g['radii'].tolist(),rotation_columns=g['rotation'].tolist()))))
            for name in ('front_points.ply','background_points.ply','camera_centers.ply'):(source/name).write_text('test inspection asset')
            np.savez(root/'pairs.npz',**pairs)
            (root/'prepared.json').write_text('{}')
            item=dict(label='W191_0_40',raw=str(raw),prepared=str(root/'prepared.json'),pairs=str(root/'pairs.npz'),
                first=0,stop=3,size=[960,640],D2_runs=['/main/one','/main/two','/main/three'],pins={},
                D1_geometry=str(raw),D2_geometry=str(raw),D1_same_as_D2=False)
            tiny=dict(intervals=[item],recipes=plan()['recipes'])
            config={'pinhole_camera_config':{'position':pairs['poses'].tolist()}}
            target=root/'autowork_geometry'
            with patch.object(driver,'check_plan'),patch.object(driver,'run_config',return_value=config),\
                    patch.object(driver,'fit_depth_only',return_value=fit) as fit_mock,\
                    patch('diagnostics.parallax_io.intrinsics',return_value=pairs['intrinsics']),\
                    patch.object(driver.subprocess,'run',side_effect=AssertionError('Must not train')),\
                    contextlib.redirect_stdout(io.StringIO()):
                driver.prepare(tiny,target)
                driver.prepare(tiny,target)
            self.assertEqual(fit_mock.call_count,1)
            stage=driver.load(target/'W191_0_40/D0/stage.json');document=driver.load(stage['result'])
            self.assertEqual(document['fit']['reprojection_weight'],0)
            self.assertEqual(document['fit']['regularizer_version'],'ablation_depth_only_v1')
            self.assertTrue(Path(stage['result']).with_suffix('.ply').is_file())
            self.assertEqual(len(driver.load(target/'matrix.json')['cells']),9)
            # A changed inspection artifact is a conflict, never silently erased.
            Path(stage['result']).with_suffix('.ply').write_bytes(b'changed')
            with patch.object(driver,'check_plan'),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):
                driver.prepare(tiny,target)


if __name__=='__main__':unittest.main()
