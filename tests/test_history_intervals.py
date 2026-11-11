"""Interval calibration must use prior residuals and score later outcomes."""
import unittest
import numpy as np
import pandas as pd

from fund_analytics.history_intervals import make_ranges, score


class HistoryIntervalTests(unittest.TestCase):
    def test_signed_error_quantiles_and_nonnegative_clipping(self):
        cal=pd.DataFrame({'actual':[0.,10.,20.,30.,40.], 'prediction':[10.]*5})
        test=pd.DataFrame({'date':pd.to_datetime(['2021-01-01']), 'actual':[20.], 'prediction':[10.],
                           'target':['subscriptions'],'model':['ridge'],'approach':['rolling']})
        result=make_ranges(cal,test,'daily_amount')
        self.assertEqual(result.calibration_n.iloc[0],5)
        self.assertAlmostEqual(result.lower.iloc[0],4.)
        self.assertAlmostEqual(result.upper.iloc[0],36.)
        metrics,yearly,scored=score(result,'daily_amount')
        self.assertEqual(metrics.coverage.iloc[0],1.)
        self.assertEqual(metrics.mean_width_gbp.iloc[0],32.)
        self.assertEqual(yearly.year.iloc[0],2021)

    def test_monthly_ranges_use_aggregated_residuals_without_clipping(self):
        cal=pd.DataFrame({'actual':[100.,120.,80.,110.], 'prediction':[90.,100.,100.,100.]})
        test=pd.DataFrame({'month':['2022-02'], 'actual':[0.], 'prediction':[5.],
                           'target':['net_flow'],'model':['ridge'],'approach':['rolling']})
        result=make_ranges(cal,test,'net_flow_monthly')
        self.assertLess(result.lower.iloc[0],0.)
        self.assertEqual(result.calibration_n.iloc[0],4)
        metrics,yearly,_=score(result,'net_flow_monthly')
        self.assertEqual(yearly.year.iloc[0],2022)
        self.assertEqual(metrics.observations.iloc[0],1)
