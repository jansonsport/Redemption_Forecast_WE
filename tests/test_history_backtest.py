"""History strategy boundaries, temporal isolation and scoring."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from fund_analytics.forecast import Spec, predict, predict_forest, recency_weights, training_slice
from fund_analytics.history_backtest import Configuration, evaluate, select, split_months, summarize


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2006-11-01', '2026-10-30')
        self.history = pd.DataFrame({'date': self.dates, 'subscriptions': 10., 'redemptions': 20.})

    def test_split_and_missing_data(self):
        warmup, val, test = split_months(self.history)
        self.assertEqual((str(warmup[0]),str(warmup[-1])), ('2006-11','2011-10'))
        self.assertEqual((str(val[0]),str(val[-1])), ('2011-11','2013-10'))
        self.assertEqual((str(test[0]),str(test[-1]),len(test)), ('2013-11','2026-09',155))
        for broken in (self.history.drop(index=1), pd.concat([self.history,self.history.iloc[:1]])):
            with self.assertRaises(ValueError):
                split_months(broken)

    def test_boundaries_and_weights(self):
        series = self.history.set_index('date').subscriptions
        cutoff = pd.Timestamp('2015-10-31')
        self.assertEqual(training_slice(series,cutoff,60).index.min(), pd.Timestamp('2010-11-01'))
        self.assertEqual(training_slice(series,cutoff,None).index.min(), self.dates[0])
        dates = pd.to_datetime(['2011-10-31','2015-10-31'])
        weights = recency_weights(dates,cutoff,4.)
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[1]/weights[0],2.)
        for half in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                predict(series,[cutoff+pd.Timedelta(days=1)],cutoff,Spec('ridge',None),half)

    def test_leap_year_twelve_month_history_status(self):
        series = self.history.set_index('date').subscriptions
        _,status = predict(series, pd.date_range('2017-03-01','2017-03-31'),
                           pd.Timestamp('2017-02-28'), Spec('weekday',12))
        self.assertEqual(status,'ok')

    def test_all_strategies_ignore_future(self):
        series = pd.Series(np.arange(len(self.dates)) % 19, index=self.dates,dtype=float)
        cutoff = pd.Timestamp('2014-10-31')
        dates = pd.date_range('2014-11-01','2014-11-30')
        changed = series.copy()
        changed.loc[changed.index>cutoff] = 1e12
        for window,half in ((60,None),(None,None),(None,3.)):
            for method in ('ridge','forest'):
                def fit(s):
                    return (predict(s,dates,cutoff,Spec('ridge',window),half)[0] if method=='ridge'
                            else predict_forest(s,dates,cutoff,window,3,12,half))
                a,b = fit(series),fit(changed)
                np.testing.assert_allclose(a,b)
                self.assertTrue((a>=0).all())
        old = series.copy()
        old.loc[old.index < '2009-11-01'] = 1e12
        np.testing.assert_allclose(predict(series,dates,cutoff,Spec('ridge',60))[0],
                                   predict(old,dates,cutoff,Spec('ridge',60))[0])

    def test_metrics_aggregate_errors_before_absolute_value(self):
        rows = []
        for target,actual in (('subscriptions',[10.,30.]),('redemptions',[0.,0.])):
            for model in ('ridge','mean','weekday'):
                for date,a,p in zip(pd.to_datetime(['2020-01-01','2020-01-02']),actual,[20.,20.]):
                    rows.append(dict(date=date,month='2020-01',actual=a,prediction=p,
                                     target=target,model=model,approach='rolling' if model=='ridge' else 'baseline'))
        scored = summarize(pd.DataFrame(rows))
        row = scored.query("period=='overall' and target=='subscriptions' and model=='ridge'").iloc[0]
        self.assertEqual(row.daily_mae_gbp,10.)
        self.assertEqual(row.monthly_total_mae_gbp,0.)
        self.assertEqual(row.daily_bias_gbp,0.)
        self.assertEqual(row.daily_mae_pct_of_mean_actual,50.)
        zero = scored.query("target=='redemptions'")
        self.assertTrue(zero.daily_mae_pct_of_mean_actual.isna().all())
        self.assertTrue(scored.monthly_total_mae_gbp_improvement_pct_vs_mean.isna().loc[scored.target=='subscriptions'].all())

    def test_selection_uses_daily_then_monthly_then_grid_order(self):
        configs = [Configuration('ridge','rolling',alpha=a) for a in (.1,1.,10.)]
        def frame(prediction):
            return pd.DataFrame({'month':['2020-01','2020-01'], 'actual':[10.,10.], 'prediction':prediction})
        # Equal daily MAE, but cancellation makes the second monthly total correct.
        outcomes = [frame([12.,12.]), frame([12.,8.]), frame([12.,8.])]
        with patch('fund_analytics.history_backtest.candidates', return_value=configs), patch(
                'fund_analytics.history_backtest.evaluate', side_effect=outcomes):
            config,_ = select(None,None,'ridge','rolling')
        self.assertEqual(config.alpha,1.)

    def test_validation_isolated_and_zero_history(self):
        series = self.history.set_index('date').subscriptions
        months = pd.period_range('2011-11','2011-12',freq='M')
        grid = [Configuration('ridge','weighted',alpha=a,half_life=1.) for a in (.1,1.)]
        altered = series.copy()
        altered.loc[altered.index>months[-1].end_time] = 1e9
        with patch('fund_analytics.history_backtest.candidates', return_value=grid):
            a,_ = select(series,months,'ridge','weighted')
            b,_ = select(altered,months,'ridge','weighted')
        self.assertEqual(a,b)
        dates = []
        for model in ('ridge','random_forest','mean','weekday'):
            for approach in (('rolling','expanding','weighted') if model in ('ridge','random_forest') else ('baseline',)):
                frame = evaluate(series*0,months,Configuration(model,approach,half_life=1. if approach=='weighted' else None))
                self.assertTrue((frame.prediction==0).all())
                self.assertTrue((frame.status=='all_zero_history').all())
                dates.append(frame.date)
        for d in dates[1:]:
            pd.testing.assert_series_equal(dates[0],d)
