"""Temporal availability and monthly forecast invariants."""
import unittest
import numpy as np
import pandas as pd

from fund_analytics.monthly_forecast import (
    feature_rows, training_examples, forest_prediction, split_origins,
    monthly_history, add_derived, summarise, baseline_predictions,
)


class MonthlyTests(unittest.TestCase):
    def setUp(self):
        self.months = pd.period_range('2000-01', periods=240, freq='M')
        self.values = np.arange(240, dtype=float)+1

    def test_future_values_cannot_change_features_training_or_prediction(self):
        cutoff = 120
        altered = self.values.copy()
        altered[cutoff+1:] = 1e12
        np.testing.assert_array_equal(feature_rows(self.values,self.months,cutoff),
                                      feature_rows(altered,self.months,cutoff))
        x,y = training_examples(self.values,self.months,cutoff)
        xx,yy = training_examples(altered,self.months,cutoff)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(y,yy)
        a = forest_prediction(self.values,self.months,cutoff,(3,6))
        b = forest_prediction(altered,self.months,cutoff,(3,6))
        np.testing.assert_array_equal(a,b)
        cached = forest_prediction(self.values,self.months,cutoff,(3,6),(x,y))
        np.testing.assert_array_equal(a,cached)
        self.assertTrue((a>=0).all())
        # At lead 12 the latest known month is the seasonal reference.
        self.assertEqual(feature_rows(self.values,self.months,cutoff)[11,-1],self.values[cutoff])

    def test_validation_and_calibration_targets_are_known_before_next_phase(self):
        val,cal,test = split_origins(240)
        self.assertLessEqual(max(val)+12,min(cal))
        self.assertLessEqual(max(cal)+12,min(test))
        self.assertLess(max(test)+12,240)
        self.assertEqual(len({self.months[i].month for i in test}),12)

    def test_complete_months_only_and_missing_coverage_rejected(self):
        dates = pd.date_range('2020-01-02','2020-03-30')
        frame = pd.DataFrame(dict(date=dates,client_sector='A',subscriptions=0.,redemptions=0.))
        _,months = monthly_history(frame)
        self.assertEqual(list(map(str,months)),['2020-02'])
        with self.assertRaises(ValueError):
            monthly_history(frame.drop(index=40))

    def test_fund_and_net_are_derived_from_matched_sector_predictions(self):
        rows = []
        for s in ('A','B'):
            for target,amount in [('subscriptions',100.),('redemptions',40.)]:
                rows.append(dict(scope='sector',sector=s,target=target,phase='evaluation',
                    origin='2020-01',target_month='2020-02',lead_month=1,model='random_forest',
                    actual=amount,prediction=amount))
        result=add_derived(pd.DataFrame(rows))
        fund=result[(result.scope=='fund')&(result.target=='net_flow')]
        self.assertEqual(fund.prediction.iloc[0],120.)

    def test_baselines_use_only_pre_origin_values(self):
        dates=pd.date_range('2000-01-01','2019-12-31')
        series=pd.Series(np.arange(len(dates))%100.,index=dates)
        values=series.groupby(series.index.to_period('M')).sum().to_numpy()
        cutoff=120
        changed=series.copy()
        changed.loc[changed.index>self.months[cutoff].end_time]=1e12
        changed_values=changed.groupby(changed.index.to_period('M')).sum().to_numpy()
        a=baseline_predictions(series,values,self.months,cutoff)
        b=baseline_predictions(changed,changed_values,self.months,cutoff)
        for model in a:
            np.testing.assert_array_equal(a[model],b[model])

    def test_metrics_compare_matching_baseline_and_do_not_divide_by_zero(self):
        records=[]
        for name,pred in [('random_forest',9.),('seasonal_naive',6.)]:
            records.append(dict(scope='fund',sector='Whole fund',target='subscriptions',
                model=name,lead_month=1,origin='2020-01',actual=10.,prediction=pred,
                lower=5.,upper=12.))
        summary=summarise(pd.DataFrame(records))
        forest=summary[summary.model=='random_forest'].iloc[0]
        self.assertEqual(forest.mae_gbp,1.)
        self.assertEqual(forest.skill_vs_seasonal,.75)
        self.assertEqual(forest.interval_coverage,1.)


if __name__=='__main__':
    unittest.main()
