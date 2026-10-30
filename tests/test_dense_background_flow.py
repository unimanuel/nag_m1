"""CPU-only checks of dense checkpoint-flow postprocessing."""
import sys
from pathlib import Path
import unittest
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'reports'),str(REPO/'nag/scripts'),str(REPO/'nag')]
from dense_background_flow import dense_fields, magnitude_mean, heatmap
from learned_flow_diagnostics import flow_fields


class DenseFlowTests(unittest.TestCase):
    def test_magnitude_average_does_not_cancel_opposite_vectors(self):
        vectors=np.array([[[3.,4.]], [[-3.,-4.]]])
        mean,count=magnitude_mean(np.linalg.norm(vectors,axis=-1),np.ones((2,1),bool))
        self.assertEqual(mean[0],5);self.assertEqual(count[0],2)
        self.assertEqual(np.linalg.norm(vectors.mean(0)),0)

    def test_absent_support_is_not_zero(self):
        mean,count=magnitude_mean(np.array([[5.,7.],[np.nan,9.]]),np.array([[True,False],[True,False]]))
        self.assertEqual(mean[0],5);self.assertTrue(np.isnan(mean[1]));self.assertEqual(count[1],0)

    def test_dense_gradient_and_magnitude_have_saved_pixel_units(self):
        y,x=np.mgrid[:7,:9];flow=np.stack((2*x,3*y),-1).astype(np.float32)
        fields,valid=dense_fields(flow,flow/100,flow/100,np.ones((7,9),bool))
        np.testing.assert_allclose(fields['image_pixel_gradient'][1:-1,1:-1],np.sqrt(13))
        self.assertAlmostEqual(float(fields['image_pixel_magnitude'][4,3]),np.sqrt(180),places=5)
        self.assertTrue(np.isnan(fields['image_pixel_gradient'][0]).all())

    def test_invalid_projection_is_excluded_with_its_gradient_stencil(self):
        flow=np.ones((5,5,2),np.float32);valid=np.ones((5,5),bool);valid[2,2]=False
        fields,_=dense_fields(flow,flow,flow,valid)
        self.assertTrue(np.isnan(fields['image_pixel_magnitude'][2,2]))
        self.assertTrue(np.isnan(fields['image_pixel_gradient'][2,1]))
        self.assertEqual(fields['image_pixel_gradient'][1,1],0)

    def test_matches_existing_five_point_diagnostic(self):
        rng=np.random.default_rng(42)
        pixel=rng.normal(size=(7,9,2)).astype(np.float32)
        native=pixel/100
        dense,_=dense_fields(pixel,native,native,np.ones((7,9),bool))
        def stencil(a):
            return np.stack([a[1:-1,1:-1],a[1:-1,2:],a[1:-1,:-2],
                             a[2:,1:-1],a[:-2,1:-1]]).reshape(5,-1,2)
        y,x=np.mgrid[:7,:9]
        before=np.stack((x,y),-1)*.001+.3
        flags=np.ones((5,35),bool)
        old,_=flow_fields(stencil(before),stencil(before+native),stencil(native),
                          stencil(pixel),flags,flags,flags,np.ones(2))
        for key in ('image_pixel_magnitude','image_pixel_gradient','native_uv_magnitude'):
            np.testing.assert_allclose(dense[key][1:-1,1:-1].ravel(),old[key],rtol=1e-6,atol=1e-7)

    def test_heatmap_does_not_mutate_metrics_or_colour_invalid_as_zero(self):
        a=np.array([[0.,4.],[np.nan,2.]]);before=a.copy()
        im=heatmap(a,np.isfinite(a),1.,'Flow')
        np.testing.assert_equal(a,before)
        self.assertEqual(im.size,(2,50))
        self.assertEqual(tuple(np.asarray(im)[49,0]),(175,175,175))


if __name__=='__main__':unittest.main()
