import unittest
import numpy as np
import pandas as pd
from engine import design, context_features

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(dict(cell_id=['a','b','c'], year=[2024]*3, lat=[28.,28.001,29.], lon=[77.,77.001,78.], NDVI=[.1,.3,.8], NTL=[0.,1.,2.], Pop=[10.,20.,30.], SO2=[4.,5.,6.]))
        self.bundle = dict(context_columns=['NDVI'], context_radius_km=3., year_means={2024:{'SO2':4.}}, base_winsor_bounds={'NDVI':(.15,.7)}, context_winsor_bounds={}, base_features=['NDVI','SO2','NTL_log'], context_features=['ctx_NDVI','ctx_count','ctx_fallback'], features=['NDVI','SO2','NTL_log','ctx_NDVI','ctx_count','ctx_fallback'])
    def test_self_exclusion_and_fallback(self):
        out = context_features(self.raw,self.bundle)
        np.testing.assert_allclose(out.ctx_NDVI,[.3,.1,.2])
        np.testing.assert_array_equal(out.ctx_fallback,[0,0,1])
    def test_transform_order_and_saved_means(self):
        out=design(self.raw,self.bundle)
        self.assertEqual(list(out),self.bundle['features'])
        np.testing.assert_allclose(out.NDVI,[.15,.3,.7])
        np.testing.assert_allclose(out.SO2,[0,1,2])
        np.testing.assert_allclose(out.NTL_log,np.log1p([0,1,2]))
    def test_scenario_recomputes_neighbours(self):
        changed=self.raw.copy(); changed.loc[0,'NDVI']=.5
        before=context_features(self.raw,self.bundle); after=context_features(changed,self.bundle)
        self.assertAlmostEqual(after.loc[0,'ctx_NDVI'],before.loc[0,'ctx_NDVI'])
        self.assertAlmostEqual(after.loc[1,'ctx_NDVI'],.5)
        self.assertAlmostEqual(after.loc[2,'ctx_NDVI'],.4)
    def test_unknown_year_rejected(self):
        self.raw['year']=2027
        with self.assertRaises(ValueError): design(self.raw,self.bundle)

if __name__=='__main__': unittest.main()
