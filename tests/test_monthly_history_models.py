"""Monthly history selection, target availability and aggregation invariants."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fund_analytics.monthly_forecast import (
    MonthlySpec, history_examples, monthly_weights, history_prediction,
    selection_rank, compute,
)


class MonthlyHistoryTests(unittest.TestCase):
    def setUp(self):
        self.months = pd.period_range('2000-01', periods=240, freq='M')
        self.values = np.arange(240, dtype=float)+100

    def test_targets_obey_window_and_cutoff_and_weights_use_target_age(self):
        x,y,ages = history_examples(self.values,self.months,120,60)
        self.assertEqual(y.min(),self.values[61])
        self.assertEqual(y.max(),self.values[120])
        np.testing.assert_array_equal(ages,120-(y-100))
        changed = self.values.copy(); changed[121:] = 1e12
        xx,yy,aa = history_examples(changed,self.months,120,60)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(y,yy)
        np.testing.assert_array_equal(ages,aa)
        weights = monthly_weights(np.array([0,12,24]),1.)
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[0]/weights[1],2.)
        _,expanded,_ = history_examples(self.values,self.months,120,None)
        self.assertEqual(expanded.min(),self.values[12])

    def test_prediction_ignores_future_for_each_family(self):
        dates = pd.date_range('2000-01-01','2019-12-31')
        series = pd.Series(np.arange(len(dates))%13+1.,index=dates)
        values = series.groupby(series.index.to_period('M')).sum().to_numpy()
        origin = 120
        changed = series.copy(); changed.loc[changed.index>self.months[origin].end_time] = 1e12
        changed_values = changed.groupby(changed.index.to_period('M')).sum().to_numpy()
        for model in ('ridge','random_forest'):
            config = MonthlySpec(model,'weighted',half_life=1.)
            a = history_prediction(series,values,self.months,origin,config)
            b = history_prediction(changed,changed_values,self.months,origin,config)
            np.testing.assert_allclose(a,b)
            self.assertTrue((a>=0).all())

    def test_selection_prefers_total_error_and_horizon_changes_rank(self):
        actual = np.zeros((1,12))
        first = np.array([[1.,-1.]+[0.]*10])
        second = np.array([[0.,3.]+[0.]*10])
        self.assertGreater(selection_rank(first,actual,1,0),selection_rank(second,actual,1,1))
        self.assertLess(selection_rank(first,actual,6,0),selection_rank(second,actual,6,1))
        self.assertLess(selection_rank(first,actual,6,0),selection_rank(first,actual,6,1))

    def test_compute_keeps_horizons_and_aggregate_calibration_separate(self):
        dates = pd.date_range('2000-01-01','2019-12-31')
        history = pd.DataFrame(dict(date=dates,client_sector='A',subscriptions=10.,redemptions=5.))
        def candidates(family):
            return [MonthlySpec(family,'expanding')]
        def forecast(series,values,months,origin,config,prepared=None):
            return np.repeat(values[origin]*.9,12)
        def baselines(series,values,months,origin):
            return {name:np.repeat(values[origin],12) for name in ('seasonal_naive','mean_12','weekday_12')}
        with patch('fund_analytics.monthly_forecast.model_candidates',side_effect=candidates), patch(
                'fund_analytics.monthly_forecast.history_prediction',side_effect=forecast), patch(
                'fund_analytics.monthly_forecast.baseline_predictions',side_effect=baselines):
            result = compute(history,'test',workers=1)
        self.assertEqual(result['metadata']['version'],'monthly-history-v2')
        detail = result['detail']; cumulative = result['cumulative']
        keys = ['horizon','scope','sector','target','model','phase','origin']
        sums = detail.groupby(keys).prediction.sum().sort_index()
        actual = cumulative.set_index(keys).prediction.sort_index()
        pd.testing.assert_series_equal(sums,actual)
        fund = detail.query("scope=='fund'")
        sectors = detail.query("scope=='sector'")
        self.assertAlmostEqual(fund.prediction.sum(),sectors.prediction.sum())
        for horizon,g in detail.query("phase=='outlook'").groupby('horizon'):
            self.assertTrue((g.groupby(['scope','sector','target','model']).size()==horizon).all())
        self.assertEqual(len(result['models']),24)
        self.assertTrue(detail.query("phase=='outlook'").actual.isna().all())
