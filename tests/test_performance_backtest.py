"""Performance features must match information available at monthly origins."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fund_analytics.performance_backtest import CONFIGS, performance_features, fit_month


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2006-11-01', '2014-12-31')
        valuation_dates = pd.bdate_range(self.dates[0], self.dates[-1])
        self.prices = pd.Series(100*np.exp(np.arange(len(valuation_dates))*.0001), index=valuation_dates)
        self.series = pd.Series(np.arange(len(self.dates)) % 11, index=self.dates, dtype=float)
        self.perf = performance_features(self.prices, self.dates)

    def test_prior_month_returns_and_early_eligibility(self):
        month = self.perf.loc['2013-12']  # November 30 was a Saturday.
        self.assertEqual(month.price_date.nunique(), 1)
        self.assertEqual(month.price_date.iloc[0], pd.Timestamp('2013-11-29'))
        self.assertTrue((month.feature_cutoff == pd.Timestamp('2013-11-30')).all())
        for n in (5,21,63):
            np.testing.assert_allclose(month[f'return_{n}'], np.exp(n*.0001)-1)
        # This fixture excludes weekends only; the supplied valuations also omit holidays.
        self.assertEqual(self.perf.dropna().index.min(), pd.Timestamp('2007-02-01'))

    def test_bad_prices_and_insufficient_history(self):
        for values in (self.prices*0, self.prices*np.nan, pd.concat([self.prices,self.prices.iloc[:1]])):
            with self.assertRaises(ValueError):
                performance_features(values,self.dates)
        with self.assertRaises(ValueError):
            fit_month(self.series,self.perf,pd.Period('2007-01','M'),CONFIGS['subscriptions'][1],True)

    def test_future_information_and_matched_training(self):
        cutoff = pd.Timestamp('2013-11-30')
        changed_prices = self.prices.copy()
        changed_prices.loc[changed_prices.index > cutoff] *= 100
        changed_series = self.series.copy()
        changed_series.loc[changed_series.index > cutoff] = 1e12
        changed_perf = performance_features(changed_prices,self.dates)
        pd.testing.assert_frame_equal(self.perf.loc[:'2013-12-31'],changed_perf.loc[:'2013-12-31'])
        for config in [c for configs in CONFIGS.values() for c in configs]:
            outputs=[]
            for augmented in (False,True):
                a=fit_month(self.series,self.perf,pd.Period('2013-12','M'),config,augmented)
                b=fit_month(changed_series,changed_perf,pd.Period('2013-12','M'),config,augmented)
                np.testing.assert_allclose(a.prediction,b.prediction)
                self.assertTrue((a.prediction>=0).all())
                outputs.append(a)
            pd.testing.assert_frame_equal(outputs[0][['date','training_start','training_end','training_days']],
                                          outputs[1][['date','training_start','training_end','training_days']])

    def test_ridge_scaling_uses_training_rows_only(self):
        captured={}
        class CaptureRidge:
            def __init__(self,alpha):
                captured['alpha']=alpha
            def fit(self,x,y,sample_weight):
                captured['train']=x.copy()
            def predict(self,x):
                captured['future']=x.copy()
                return np.zeros(len(x))
        perf=self.perf.copy()
        for n in (5,21,63):
            perf[f'return_{n}']=np.arange(len(perf))*n*.001
        perf.loc['2013-12', ['return_5','return_21','return_63']]=1e8
        with patch('fund_analytics.performance_backtest.Ridge',CaptureRidge):
            fit_month(self.series,perf,pd.Period('2013-12','M'),CONFIGS['subscriptions'][1],True)
        np.testing.assert_allclose(captured['train'][:,-3:].mean(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(captured['train'][:,-3:].std(axis=0),1)
        self.assertTrue((captured['future'][:,-3:]>100).all())
        self.assertEqual(captured['alpha'],.1)
