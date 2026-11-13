"""Selected outlooks must preserve objective identity and aggregate uncertainty."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fund_analytics.selected_forecasts import FAMILIES, add_net, compute, select_daily_models


class SelectedTests(unittest.TestCase):
    def test_net_preserves_negative_amounts_and_rejects_misalignment(self):
        frame = pd.DataFrame(dict(date=['2020-01-01']*2,
            target=['subscriptions','redemptions'],prediction=[10.,30.],actual=[5.,40.]))
        result = add_net(frame,['date'])
        net = result[result.target=='net_flow'].iloc[0]
        self.assertEqual(net.prediction,-20.)
        self.assertEqual(net.actual,-35.)
        frame.loc[1,'date']='2020-01-02'
        with self.assertRaises(ValueError):
            add_net(frame,['date'])

    def test_family_selection_net_ranges_and_configuration_identity(self):
        dates = pd.date_range('2006-11-01','2016-10-30')
        history = pd.DataFrame(dict(date=dates,subscriptions=5.,redemptions=8.))
        months = list(map(str,pd.period_range('2013-11','2016-09',freq='M')))
        rows=[];models=[];selection=[]
        for model in FAMILIES:
            for target in ('subscriptions','redemptions'):
                approach = 'expanding' if model=='random_forest' else 'rolling'
                selection.append(dict(period='overall',model=model,target=target,approach=approach,
                                      daily_mae_gbp=1.,monthly_total_mae_gbp=2.))
                models.append(dict(target=target,model=model,approach=approach,alpha=.1,
                    depth=5,leaf=12 if target=='subscriptions' else 24,
                    half_life=1. if approach=='weighted' else np.nan))
                for month in months:
                    for date in pd.date_range(pd.Period(month).start_time,pd.Period(month).end_time.normalize()):
                        rows.append(dict(date=date,month=month,target=target,model=model,approach=approach,
                            phase='evaluation',prediction=2. if target=='subscriptions' else 3.,
                            actual=3. if target=='subscriptions' else 5.))
        meta=dict(coverage_end='2016-10-30',evaluation_months=months,source_fingerprint='test')
        def forest(series,dates,cutoff,window,depth,leaf,half_life):
            self.assertEqual(window,None)
            self.assertEqual(depth,5)
            self.assertEqual(leaf,12 if series.name=='subscriptions' else 24)
            return np.full(len(dates),series.mean())
        def ridge(series,dates,cutoff,spec,half_life):
            self.assertEqual(spec.alpha,.1)
            self.assertEqual(spec.window,60)
            self.assertIsNone(half_life)
            return np.full(len(dates),series.mean()*2),'ok'
        with patch('fund_analytics.selected_forecasts.predict_forest',side_effect=forest), patch(
                'fund_analytics.selected_forecasts.predict',side_effect=ridge):
            result=compute(history,pd.DataFrame(models),pd.DataFrame(rows),meta,pd.DataFrame(selection))
        self.assertEqual(result['metadata']['forecast_month'],'2016-11')
        self.assertEqual(result['metadata']['version'],'selected-history-v2')
        self.assertEqual(len(result['daily']),180)
        self.assertNotIn('monthly',result)
        self.assertEqual(set(result['daily'].model),set(FAMILIES))
        self.assertEqual(result['daily'].query("target=='subscriptions' and model=='ridge'").prediction.sum(),300.)
        self.assertEqual(result['daily'].query("target=='subscriptions' and model=='random_forest'").prediction.sum(),150.)
        self.assertTrue((result['daily'].query("target=='net_flow'").lower<0).all())
        metrics=result['metrics']
        self.assertEqual(len(metrics),6)
        self.assertEqual(set(metrics.model),set(FAMILIES))
        self.assertTrue((metrics.query("objective=='daily'").calibration_cases==730).all())
        with self.assertRaises(ValueError):
            compute(history,pd.DataFrame(models),pd.DataFrame(rows),dict(meta,coverage_end='2016-10-29'),pd.DataFrame(selection))

    def test_daily_selection_resolves_ties_without_mixing_families(self):
        rows=[]
        for family in FAMILIES:
            for target in ('subscriptions','redemptions'):
                for history, daily, monthly in [('rolling',1.,3.),('expanding',1.,2.),('weighted',1.,2.)]:
                    rows.append(dict(period='overall',model=family,target=target,approach=history,
                                     daily_mae_gbp=daily,monthly_total_mae_gbp=monthly))
        selected=select_daily_models(pd.DataFrame(rows))
        self.assertEqual(selected,{family:{target:'expanding' for target in ('subscriptions','redemptions')}
                                   for family in FAMILIES})
        with self.assertRaises(ValueError):
            select_daily_models(pd.DataFrame(rows).query("target != 'subscriptions'"))
