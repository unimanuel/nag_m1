"""CPU-only unit tests for generalized discovery, references and stage safety."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'nag'))
sys.path.insert(1,str(REPO/'nag/scripts'))
from preparation.jobs import PreparationJob, discover, audit, select, SPHERE
from preparation.reference import initialize_runner, mapping_from_diagnostic, completed_reference
from preparation.storage import Stage, recipe, write_json
from preparation.verification import geometry_difference, compare_arrays, release_gate
from preparation.compatibility import allowed_preparation_change, PATH, OLD, NEW
from diagnostics.parallax_observations import targets_for
import generalized_ellipsoid_preparation as cli
from preparation import _legacy_floor_kernel as legacy_kernel
from preparation.legacy_floor import fit_sampling_floor
from reprojection_ellipsoid import ReprojectionFitConfig, enforce_safety
from test_reprojection_ellipsoid import synthetic_pairs


def dataset(base, segment='123456'):
    root=base/('segment-'+segment)
    for name in ['images_undistorted','masks','depth','boxes']:(root/name).mkdir(parents=True,exist_ok=True)
    poses=np.repeat(np.eye(4)[None],3,axis=0).tolist()
    camera={'position':poses,'resolution':{'__class__':'tuple','value':[32,24]},'lens_distortion':[0]*5,
            'focal_length':1.,'center':[.5,.5]}
    for name in ['waymo_pose_config.json','waymo_pose_config_no_distortion.json']:(root/name).write_text(json.dumps(camera))
    for n in range(3):
        Image.new('RGB',(32,24)).save(root/'images_undistorted'/f'{n:03d}.png')
        Image.new('L',(32,24)).save(root/'masks'/f'img_{n:03d}_ov_0.png')
        (root/'depth'/f'{n}.tiff').write_bytes(b'depth')
    (root/'boxes/77.json').write_text('{}');(root/'box_id_mapping.json').write_text('{}')
    return root


def configs(base, root):
    c={'__class__':'nag.config.nag_config.NAGConfig','data_path':str(root),
       'pinhole_camera_config':'{data_path}/waymo_pose_config_no_distortion.json',
       'frame_indices_filter':{'__class__':'slice','start':0,'stop':3,'step':None},
       'final_plot_resolution_factor':.5,'background_plane_type':SPHERE,
       'background_plane_kwargs':{'color_mask_resolution':[960,1920]},'seed':42,'max_epochs':80,
       'name':'regular','sampler_kwargs':{'num_rays':100000}}
    regular=base/'regular.yaml';visible=base/'visible.yaml'
    regular.write_text(yaml.safe_dump({'NAGConfig':c}))
    uv=copy.deepcopy(c);uv['name']='visible'
    uv['background_plane_kwargs']['sphere_visible_uv_texture']={'enabled':True,'margin_uv':[.05,.05]}
    visible.write_text(yaml.safe_dump({'NAGConfig':uv}))
    return regular,visible


class GeneralizedPreparationTests(unittest.TestCase):
    def test_discovery_deduplicates_variants_and_uses_full_identity(self):
        with tempfile.TemporaryDirectory(prefix='autowork_job_test_') as d:
            base=Path(d);root=dataset(base);paths=configs(base,root)
            jobs=discover(config_paths=paths)
            self.assertEqual(len(jobs),1)
            job=jobs[0]
            self.assertEqual(job.frames,(0,1,2));self.assertEqual(job.size,(16,12))
            self.assertEqual(job.uid,'segment-123456_frames_000_002_step1')
            self.assertEqual(select(jobs,[job.uid]),jobs)
            self.assertEqual(len(job.source_configs),2)

    def test_conflicting_preprocessing_or_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='autowork_job_test_') as d:
            base=Path(d);root=dataset(base);regular,visible=configs(base,root)
            c=yaml.safe_load(visible.read_text());c['NAGConfig']['sampler_kwargs']['num_rays']=70000
            visible.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):discover(config_paths=[regular,visible])

    def test_geometry_override_is_not_silently_carried_into_frozen_recipe(self):
        with tempfile.TemporaryDirectory(prefix='autowork_job_test_') as d:
            base=Path(d);root=dataset(base);regular,_=configs(base,root)
            c=yaml.safe_load(regular.read_text());c['NAGConfig']['background_ellipsoid_radii_override']=[30.,30.,30.]
            regular.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):discover(config_paths=[regular])

    def test_ambiguous_short_label_cannot_select_another_segment(self):
        jobs=[PreparationJob(s,0,3,1,'','','',(32,24),(16,12),'','',())for s in ['123456','123999']]
        with self.assertRaises(ValueError):select(jobs,['W123_0_2'])
        self.assertEqual(select(jobs,[jobs[1].uid]),[jobs[1]])

    def test_frame_step_is_explicit_and_never_silently_dropped(self):
        with tempfile.TemporaryDirectory(prefix='autowork_job_test_') as d:
            base=Path(d);root=dataset(base);regular,_=configs(base,root)
            c=yaml.safe_load(regular.read_text());c['NAGConfig']['frame_indices_filter']['step']=2
            regular.write_text(yaml.safe_dump(c))
            self.assertEqual(discover(config_paths=[regular])[0].frames,(0,2))

    def test_static_camera_audit_and_missing_depth(self):
        with tempfile.TemporaryDirectory(prefix='autowork_job_test_') as d:
            base=Path(d);root=dataset(base);regular,_=configs(base,root)
            job=discover(config_paths=[regular])[0];cloud=base/'cloud'/job.sequence;cloud.mkdir(parents=True)
            for n in job.frames:(cloud/f'{n:03d}.npz').write_bytes(b'cloud')
            a=audit(job,base/'cloud');self.assertTrue(a['ready']);self.assertTrue(a['weak_parallax'])
            self.assertEqual(a['maximum_camera_baseline_m'],0.)
            (root/'depth/1.tiff').unlink();self.assertEqual(audit(job,base/'cloud')['missing'],{'depth':[1]})

    def test_static_target_pairs_remain_available_without_changing_loss(self):
        poses=np.repeat(np.eye(4)[None],5,axis=0)
        targets=targets_for(2,poses,8)
        self.assertNotIn(2,targets);self.assertTrue(targets)
        self.assertEqual(recipe()['reprojection']['depth_weight'],.15)

    def test_mapping_uses_native_diagnostic_resolution_and_flip(self):
        diag={'radius_source':'visible_plane_centers','visible_uv_texture':{'enabled':False},
              'color_mask_resolution':[640,1280]}
        mapping=mapping_from_diagnostic(diag,{'background_plane_kwargs':{'sphere_uv_y_flip':True}})
        self.assertEqual(mapping['resolution_hw'],[640,1280]);self.assertTrue(mapping['y_flip'])
        diag['visible_uv_texture']['enabled']=True
        with self.assertRaises(ValueError):mapping_from_diagnostic(diag,{})

    def test_initialization_never_calls_build_trainer_train_or_checkpoints(self):
        runner=MagicMock();model=initialize_runner(runner)
        self.assertIs(model,runner.create_model.return_value)
        runner.build.assert_not_called();runner.create_trainer.assert_not_called()
        runner.train.assert_not_called();runner.save.assert_not_called()
        runner.create_dataloader.assert_not_called();runner.create_callbacks.assert_not_called()
        self.assertEqual([call[0]for call in runner.mock_calls],['create_bundle_dataset','create_model'])

    def test_completed_reference_interval_calibration_and_initialization_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='autowork_reference_test_') as d:
            base=Path(d);root=dataset(base);regular,_=configs(base,root)
            job=discover(config_paths=[regular])[0];run=base/'completed';run.mkdir()
            saved=yaml.safe_load(regular.read_text())
            saved['NAGConfig']['background_sphere_radius_from_visible_plane_centers']=True
            saved['NAGConfig']['pinhole_camera_config']=json.loads(Path(job.pose_path).read_text())
            (run/'exit_000.txt').write_text('Exit code: 0.\n')
            for mismatch in ['interval','calibration','initialization']:
                c=copy.deepcopy(saved)
                if mismatch=='interval':c['NAGConfig']['frame_indices_filter']['stop']=2
                if mismatch=='calibration':c['NAGConfig']['pinhole_camera_config']['focal_length']=2.
                (run/'training_config.yaml').write_text(yaml.safe_dump(c))
                with patch('ego_repeatability.completed',return_value={}), \
                        patch('preparation.reference._native_config') as native, \
                        patch('preparation.reference.initialization_signature',side_effect=[{'seed':42},{'seed':43}]):
                    with self.assertRaisesRegex(ValueError,'mismatch'):completed_reference(job,run,base/'out')
                    if mismatch!='initialization':native.assert_not_called()

    def test_legacy_floor_kernel_is_the_exact_historical_source(self):
        self.assertEqual(hashlib.sha256(Path(legacy_kernel.__file__).read_bytes()).hexdigest(),
                         '5228d936fb4ef98fbd235848aa3adaa0db9017529cbfe4102052c2604f6c3603')

    def test_pinned_kernel_preserves_zero_floor_exactly_and_does_not_optimize(self):
        pairs,geometry=synthetic_pairs()
        cfg=ReprojectionFitConfig(max_pairs=128,max_depth_rays=128,
            minimum_camera_surface_distance=3.,minimum_axis_radius=5.,max_iterations=1)
        geometry=enforce_safety(geometry,pairs['poses'],cfg)
        mapping={'resolution_hw':[640,960],'visible_uv':{'enabled':False}}
        with patch('preparation._legacy_floor_kernel.minimize') as optimizer:
            result=fit_sampling_floor(pairs,geometry,geometry,mapping,(960,640),cfg,max_sampling=128)
        optimizer.assert_not_called()
        self.assertTrue(result['sampling_floor']['zero_floor_preserved'])
        for name in ['center','radii','rotation']:np.testing.assert_array_equal(result['geometry'][name],geometry[name])

    def test_pinned_kernel_tiny_positive_full_ray_loss_is_not_zero(self):
        pairs,geometry=synthetic_pairs()
        cfg=ReprojectionFitConfig(max_pairs=128,max_depth_rays=128,
            minimum_camera_surface_distance=3.,minimum_axis_radius=5.,max_iterations=1)
        geometry=enforce_safety(geometry,pairs['poses'],cfg)
        sphere={k:v.copy()for k,v in geometry.items()}
        mapping={'resolution_hw':[640,960],'visible_uv':{'enabled':False}}
        def density(pixels,ids,poses,intrinsics,g,mapping):
            result={'texels_per_pixel_min':np.ones(len(pixels))}
            if len(pixels)>128 and g is not sphere:result['texels_per_pixel_min'][-1]=.5-1e-8
            return result
        def no_step(fun,start,**kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(x=start.copy(),success=True,message='unit-test identity step',nit=0)
        with patch('preparation._legacy_floor_kernel.sampling_metrics',side_effect=density), \
                patch('preparation._legacy_floor_kernel.minimize',side_effect=no_step) as optimizer:
            result=fit_sampling_floor(pairs,geometry,sphere,mapping,(960,640),cfg,max_sampling=128)
        self.assertEqual(optimizer.call_count,3)
        self.assertFalse(result['sampling_floor'].get('zero_floor_preserved',False))
        self.assertGreater(result['sampling_floor']['full_sampling']['original']['loss'],0.)

    def test_pinned_recipe_rejects_experimental_regularizers(self):
        pairs,geometry=synthetic_pairs();mapping={'visible_uv':{'enabled':False}}
        for options in [{'floor_threshold':.8},{'compactness_weight':.1},{'axis_multistart':True}]:
            with self.assertRaises(ValueError):fit_sampling_floor(pairs,geometry,geometry,mapping,(960,640),
                                                                 ReprojectionFitConfig(),**options)

    def test_stage_reuse_validates_hash_and_preserves_partial_attempt(self):
        with tempfile.TemporaryDirectory(prefix='autowork_stage_test_') as d:
            root=Path(d);stage=Stage(root,'example',{'frames':(0,1)})
            one=stage.attempt();(one/'partial').write_text('preserve')
            two=stage.attempt();result=two/'result.json';result.write_text('{}')
            stage.complete(result,two)
            self.assertEqual(Stage(root,'example',{'frames':(0,1)}).result,result)
            self.assertTrue((one/'partial').exists())
            with self.assertRaises(ValueError):Stage(root,'example',{'frames':(0,2)})
            result.write_text('changed')
            with self.assertRaises(ValueError):Stage(root,'example',{'frames':(0,1)})

    def test_unowned_nonempty_stage_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='autowork_stage_test_') as d:
            root=Path(d);(root/'example').mkdir();(root/'example/user.txt').write_text('preserve')
            with self.assertRaises(ValueError):Stage(root,'example',{'frames':[0,1]})
            self.assertEqual((root/'example/user.txt').read_text(),'preserve')

    def test_wrong_sphere_radius_policy_is_rejected(self):
        diag={'radius_source':'computed','visible_uv_texture':{'enabled':False},'color_mask_resolution':[960,1920]}
        with self.assertRaises(ValueError):mapping_from_diagnostic(diag,{})

    def test_observation_array_reordering_fails_exact_parity(self):
        with tempfile.TemporaryDirectory(prefix='autowork_array_test_') as d:
            a,b=Path(d)/'old.npz',Path(d)/'new.npz'
            np.savez(a,source=np.array([0,1]),points=np.ones((2,3)))
            np.savez(b,source=np.array([0,1]),points=np.ones((2,3)))
            compare_arrays(a,b)
            np.savez(b,source=np.array([1,0]),points=np.ones((2,3)))
            with self.assertRaises(ValueError):compare_arrays(a,b)

    def test_preserved_geometry_requires_exact_arrays(self):
        a={'fit':{'center':[0.,0.,0.],'radii':[30.,20.,10.],'rotation_columns':np.eye(3).tolist()}}
        b=copy.deepcopy(a);b['fit']['radii'][0]+=1e-7
        geometry_difference(a,b)
        with self.assertRaises(ValueError):geometry_difference(a,b,exact=True)

    def test_gate_cannot_be_bypassed_by_cached_only_report(self):
        with tempfile.TemporaryDirectory(prefix='autowork_gate_test_') as d:
            p=Path(d)/'verification.json';p.write_text(json.dumps({'released':False}))
            with self.assertRaises(ValueError):release_gate(p)

    def test_compatibility_allowance_is_exact_and_preparation_only(self):
        self.assertTrue(allowed_preparation_change(PATH,OLD,NEW))
        self.assertFalse(allowed_preparation_change('nag/model/nag_model.py',OLD,NEW))
        self.assertFalse(allowed_preparation_change(PATH,OLD,'unreviewed'))

    def test_cloud_lookup_uses_full_segment_and_training_list_index(self):
        jobs=discover(config_paths=[REPO/'config/waymo/autowork_paper-final/segment-12511696717465549299_0_40_large-undistorted.yaml'])
        commands=cli.cloud_commands(jobs,Path('/tmp/raw'),Path('/tmp/cloud'))
        self.assertEqual(commands[0][commands[0].index('--scene-ids')+1],'115')
        self.assertEqual(len(commands),2)

    def test_cli_dry_reference_and_audit_never_write_or_initialize(self):
        with tempfile.TemporaryDirectory(prefix='autowork_cli_test_') as d:
            base=Path(d);root=dataset(base);regular,_=configs(base,root)
            job=discover(config_paths=[regular])[0];cloud=base/'cloud'/job.sequence;cloud.mkdir(parents=True)
            for n in job.frames:(cloud/f'{n:03d}.npz').write_bytes(b'cloud')
            for stage in ['audit','reference']:
                out=base/'autowork_output'
                with patch.object(sys,'argv',['cli',stage,'--config-paths',str(regular),'--cloud-root',str(base/'cloud'),
                                            '--output-root',str(out),'--dry-run']), \
                        patch.object(cli,'reference_stage') as reference,patch.object(cli.subprocess,'run') as run, \
                        contextlib.redirect_stdout(io.StringIO()):cli.main()
                reference.assert_not_called();run.assert_not_called();self.assertFalse(out.exists())

    def test_six_training_commands_have_fixed_geometry_and_both_layouts(self):
        for mode in ['regular','visuv']:
            for repeat in [1,2,3]:
                cmd=cli.training_command({'id':f'id{repeat}','config':f'{mode}.yaml','geometry':'ellipsoid.json'},Path('out'))
                self.assertEqual(cmd[cmd.index('--background-ellipsoid-scale-margin')+1],'0')
                self.assertEqual(cmd[cmd.index('--background-ellipsoid-texture-orientation')+1],'legacy')
                self.assertFalse(any('sphere-radius' in x for x in cmd))


if __name__=='__main__':unittest.main()
