"""Temporal and aggregation invariants for forecast helpers."""
import unittest

import numpy as np
import pandas as pd

from fund_analytics.forecast import Spec, add_net, bounds, evaluation_months, monthly_totals, predict, predict_forest


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2020-01-01', '2025-12-31')
        self.series = pd.Series(np.arange(len(self.dates)) % 7 + 10.0, index=self.dates)

    def test_future_changes_cannot_affect_fit(self):
        cutoff = pd.Timestamp('2024-12-31')
        dates = pd.date_range('2025-01-01', '2025-01-31')
        changed = self.series.copy()
        changed.loc[changed.index > cutoff] = 1e12
        for spec in (Spec('mean',12), Spec('weekday',12), Spec('ridge',12)):
            a, _ = predict(self.series, dates, cutoff, spec)
            b, _ = predict(changed, dates, cutoff, spec)
            np.testing.assert_allclose(a,b)
            self.assertTrue((a >= 0).all())

    def test_zero_history_is_explicit(self):
        pred, status = predict(self.series * 0, pd.date_range('2026-01-01','2026-01-31'), self.dates[-1], Spec('ridge',12))
        self.assertEqual(status,'all_zero_history')
        self.assertTrue((pred == 0).all())

    def test_daily_forest_is_cutoff_safe_and_nonnegative(self):
        cutoff=pd.Timestamp('2024-12-31')
        dates=pd.date_range('2025-01-01','2025-01-31')
        changed=self.series.copy()
        changed.loc[changed.index>cutoff]=1e12
        a=predict_forest(self.series,dates,cutoff,12,3,6)
        b=predict_forest(changed,dates,cutoff,12,3,6)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a>=0).all())

    def test_partial_last_month_excluded(self):
        months = evaluation_months(pd.Timestamp('2020-01-01'),pd.Timestamp('2025-12-30'))
        self.assertEqual(len(months),36)
        self.assertEqual(str(months[-1]),'2025-11')

    def test_monthly_uncertainty_uses_monthly_errors(self):
        cal = pd.DataFrame({'month':['2020-01','2020-01','2020-02','2020-02'],
                            'actual':[20,0,20,0], 'prediction':[10,10,10,10]})
        low, high = bounds(monthly_totals(cal),np.array([20.]),True)
        np.testing.assert_equal(low,[20.])
        np.testing.assert_equal(high,[20.])
        low_daily, _ = bounds(cal,np.array([10.]),True)
        self.assertLess(low_daily[0] * 2,low[0])

    def test_net_uses_aligned_actual_and_predicted_differences(self):
        f = pd.DataFrame({'date':pd.to_datetime(['2020-01-01']*2),
                          'scope':['fund']*2,'sector':['Whole fund']*2,'phase':['evaluation']*2,
                          'target':['subscriptions','redemptions'], 'prediction':[100.,40.],
                          'actual':[80.,60.], 'method':['mean']*2})
        net = add_net(f).query("target == 'net_flow'")
        self.assertEqual(net.prediction.iloc[0],60.)
        self.assertEqual(net.actual.iloc[0],20.)


if __name__ == '__main__':
    unittest.main()
