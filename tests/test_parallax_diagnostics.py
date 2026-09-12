"""Scientific invariants for the offline parallax evaluator (CPU only)."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nag'))
from diagnostics.parallax_core import (DepthEvidence, angle_degrees, balanced_indices, image_statistics,
    inside, intersect, intrinsics, project, regions, surface_coordinates, surface_uv, uv_errors)
from diagnostics.parallax_evaluate import evaluate_pairs
from diagnostics.parallax_io import ROOTS, classify_run, diagnostic, digest, inventory, snapshot
from diagnostics.parallax_observations import targets_for
from diagnostics.parallax_spatial import spatial_comparisons


def geometry(radii=(10,10,10)):
    return {'center':np.zeros(3),'radii':np.array(radii),'rotation':np.eye(3)}


def mapping():
    return {'rotation':np.eye(3),'resolution_hw':[100,200], 'visible_uv':{'enabled':False}}


class ParallaxDiagnosticsTests(unittest.TestCase):
    def test_intersections_analytic_sphere_and_outside(self):
        origins=np.array([[0,0,0],[0,0,20],[0,0,20.]])
        directions=np.array([[0,0,1],[0,0,-1],[0,0,1.]])
        q,t,valid=intersect(origins,directions,geometry())
        np.testing.assert_allclose(t[:2],[10,10])
        np.testing.assert_array_equal(valid,[True,True,False])
        np.testing.assert_allclose(q[:2],[[0,0,10],[0,0,10]])

    def test_oriented_ellipsoid_hits_surface(self):
        g=geometry((9,4,2));g['rotation']=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        g['center']=np.array([2.,-3,1])
        q,t,valid=intersect(np.tile(g['center'],(3,1)),np.eye(3),g)
        np.testing.assert_allclose(t,[4,9,2])
        np.testing.assert_allclose(np.linalg.norm(((q-g['center'])@g['rotation'])/g['radii'],axis=1),1)

    def test_intersection_and_uv_agree_with_nag_runtime(self):
        import torch
        path=Path(__file__).resolve().parents[1]/'nag/model/sphere_utils.py'
        spec=importlib.util.spec_from_file_location('sphere_utils_for_diagnostics_test',path)
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        g=geometry((9,4,2));g['center']=np.array([1.,2,3])
        g['rotation']=np.array([[0,-1.,0],[1,0,0],[0,0,1]])
        rng=np.random.default_rng(1)
        o=g['center']+rng.normal(0,.1,(100,3));d=rng.normal(size=(100,3))
        q,t,hit=intersect(o,d,g)
        tensor=lambda x:torch.tensor(x,dtype=torch.float64)
        rq,rhit,rt=runtime.ray_ellipsoid_intersection(tensor(o),tensor(d),tensor(g['center']),tensor(g['radii']),tensor(g['rotation']))
        np.testing.assert_allclose(q,rq.numpy(),atol=1e-9)
        np.testing.assert_allclose(t,rt.numpy(),atol=1e-9)
        s=surface_coordinates(q,g)
        np.testing.assert_allclose(surface_uv(s,mapping()),runtime.direction_to_equirect_uv(tensor(s)).numpy(),atol=1e-9)

    def test_invalid_and_off_image_proxies_are_not_dropped(self):
        poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,3]=1
        pairs={'poses':poses,'intrinsics':np.array([[1000,0,10],[0,1000,10],[0,0,1.]]),
               'points':np.array([[0,0,5.]]),'source':np.array([0]),'target':np.array([1])}
        r=evaluate_pairs(pairs,geometry(),mapping(),(20,20))
        self.assertTrue(r['valid'][0]);self.assertTrue(r['proxy_outside_image'][0])
        g=geometry();g['center']=np.array([0,0,-100.])
        r=evaluate_pairs(pairs,g,mapping(),(20,20))
        self.assertFalse(r['valid'][0]);self.assertTrue(np.isnan(r['pixel_error'][0]))

    def test_exact_surface_and_pure_rotation_zero_parallax(self):
        poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,3]=1
        p=np.array([[0,0,10.],[2,0,np.sqrt(96)]])
        pairs={'poses':poses,'intrinsics':np.array([[100,0,100],[0,100,100],[0,0,1.]]),
               'points':p,'source':np.array([0,0]),'target':np.array([1,1])}
        result=evaluate_pairs(pairs,geometry(),mapping(),(200,200))
        np.testing.assert_allclose(result['pixel_error'],0,atol=1e-10)
        np.testing.assert_allclose(result['surface_angle_deg'],0,atol=1e-10)
        poses[1,:3,3]=0
        a=.4;poses[1,:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]
        pairs['points']=p/2
        result=evaluate_pairs(pairs,geometry(),mapping(),(200,200))
        np.testing.assert_allclose(result['pixel_error'],0,atol=1e-10)
        np.testing.assert_allclose(result['surface_angle_deg'],0,atol=1e-10)

    def test_translation_makes_wrong_depth_nonzero(self):
        poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,3]=1
        pairs={'poses':poses,'intrinsics':np.array([[100,0,100],[0,100,100],[0,0,1.]]),
               'points':np.array([[0,0,5.]]),'source':np.array([0]),'target':np.array([1])}
        r=evaluate_pairs(pairs,geometry(),mapping(),(200,200))
        self.assertTrue(r['valid'][0]);self.assertGreater(r['pixel_error'][0],5)
        self.assertGreater(r['surface_angle_deg'][0],1)

    def test_pixel_projection_scales_with_resolution(self):
        c={'resolution':{'value':[1920,1280]},'focal_length':1.,'principal_point':[900,600]}
        p=np.array([[1,2,10.]])
        uv,_=project(p,np.eye(4),intrinsics(c,(1920,1280)))
        half,_=project(p,np.eye(4),intrinsics(c,(960,640)))
        np.testing.assert_allclose(uv,half*2)
        self.assertEqual(intrinsics(c,(960,640))[1,1],960)

    def test_sparse_visibility_no_hole_fill(self):
        e=DepthEvidence(np.array([[10.,10],[10.1,10.1],[30,30]]),np.array([5.,8.,10.]),(100,100))
        states=e.classify(np.array([[10,10],[10,10],[10,10],[90,90]]),np.array([5.,8.,2.,5.]),1,.1,0)
        np.testing.assert_array_equal(states,[1,2,3,0])

    def test_uv_seam_and_visible_tile(self):
        a=np.array([[.99,.5]]);b=np.array([[.01,.5]])
        distance,seam,texels,covered=uv_errors(a,b,mapping())
        np.testing.assert_allclose(distance,.02);self.assertTrue(seam[0]);np.testing.assert_allclose(texels,4)
        m=mapping();m['visible_uv']={'enabled':True,'resolution':[100,200],'u_start':.9,'u_span':.2,'v_start':.4,'v_span':.2}
        _,_,texels,covered=uv_errors(a,b,m)
        np.testing.assert_allclose(texels,20);self.assertTrue(covered[0])
        _,_,texels,covered=uv_errors(np.array([[.5,.5]]),b,m)
        self.assertFalse(covered[0]);self.assertTrue(np.isnan(texels[0]))

    def test_angle_is_invariant_under_texture_rotation(self):
        a=np.array([[1.,0,0]]);b=np.array([[0.,1,0]])
        rot=np.array([[0.,0,1],[1,0,0],[0,1,0]])
        np.testing.assert_allclose(angle_degrees(a,b),angle_degrees(a@rot,b@rot))

    def test_region_partition_and_errors(self):
        mask=np.zeros((20,20),bool);mask[5:15,5:15]=True
        parts=regions(mask,2)
        np.testing.assert_array_equal(sum(x.astype(int) for x in parts.values()),np.ones((20,20)))
        self.assertFalse(np.any(parts['background']&mask))
        r=np.full((20,20,3),255,np.uint8);t=np.zeros_like(r)
        stats,mae=image_statistics(r,t,parts)
        self.assertEqual(stats['background']['mae'],1);self.assertEqual(stats['foreground']['mse'],1)
        empty=regions(np.zeros_like(mask),2)
        self.assertEqual(empty['background'].sum(),400)
        self.assertEqual(empty['boundary'].sum(),0)

    def test_snapshot_invalidation_and_effective_radii(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'diagnostic.txt'
            p.write_text('source_radii: [80, 30, 20]\neffective_radii: [40, 30.6, 20.4]\n')
            before=digest(snapshot([p]));d=diagnostic(p)
            self.assertEqual(d['effective_radii'][0],40)
            p.write_text(p.read_text()+'scale_margin: 0.02\n')
            self.assertNotEqual(before,digest(snapshot([p])))

    def test_windows_carriage_return_run_classification(self):
        p=Path('2026_W254_37_81_large-undistorted\r_ellipsoid_ray_depth_proxy_yup_v4_moderate_y_visuv')
        self.assertEqual(classify_run(p),('moderate_y','visuv','benchmark'))

    def test_reprojection_depth_run_classification(self):
        p=Path('2026_W254_37_81_large-undistorted_ellipsoid_reprojection_depth_v1_visuv')
        self.assertEqual(classify_run(p),('reprojection_v1','visuv','benchmark'))

    def test_reprojection_spatial_comparisons_include_direct_geometry_references(self):
        def run(family, mode):
            return {'id': f'{family}_{mode}_benchmark', 'family': family,
                    'mode': mode, 'cohort': 'benchmark'}
        runs = [run(family, mode) for mode in ('regular', 'visuv')
                for family in ('sphere', 'raw_v4', 'moderate_y', 'reprojection_v1')]
        comparisons, missing = spatial_comparisons(runs)
        self.assertEqual(missing, [])
        for mode in ('regular', 'visuv'):
            candidate = f'reprojection_v1_{mode}_benchmark'
            self.assertIn((candidate, f'sphere_{mode}_benchmark'), comparisons)
            self.assertIn((candidate, f'moderate_y_{mode}_benchmark'), comparisons)
            self.assertIn((candidate, f'raw_v4_{mode}_benchmark'), comparisons)

    def test_extra_run_root_is_additive(self):
        self.assertIn('2026-08-19_ellipsoid_non_center', ROOTS)
        self.assertIn('2026-09-12_regular_sphere', ROOTS)
        # The CLI passes roots through to inventory; avoid a filesystem fixture
        # here because inventory deliberately validates complete saved runs.
        import inspect
        self.assertEqual(inspect.signature(inventory).parameters['extra_roots'].default, ())

    def test_sampling_deterministic_and_targets_exclude_source(self):
        uv=np.random.default_rng(1).uniform(0,100,(100,2))
        a=balanced_indices(uv,(100,100),20,np.random.default_rng(4))
        b=balanced_indices(uv,(100,100),20,np.random.default_rng(4))
        np.testing.assert_array_equal(a,b);self.assertEqual(len(set(a)),20)
        poses=np.repeat(np.eye(4)[None],40,axis=0);poses[:,0,3]=np.arange(40)
        targets=targets_for(20,poses,8)
        self.assertNotIn(20,targets);self.assertEqual(len(targets),8)


if __name__=='__main__':unittest.main()
