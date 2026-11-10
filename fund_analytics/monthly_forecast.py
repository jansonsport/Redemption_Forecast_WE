"""Direct monthly forecasts with chronological model selection and calibration.

All horizons are learned together by a constrained forest, with lead month as
a feature. Examples sharing an origin are never split randomly. Only examples
whose target outcome is observed by the fitting cutoff enter training.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .db import connect
from .forecast import ROOT, Spec, predict

OUTPUT = ROOT / 'data/clean/monthly_forecasts'
VERSION = 'monthly-rf-v1'
SEED = 42
GRID = [(3, 6), (3, 12), (5, 6), (5, 12)]
FEATURES = ['lead', 'month_sin', 'month_cos', 'previous', 'previous_2',
            'mean_3', 'mean_12', 'same_month_last_year']
BASELINES = {'mean_12': Spec('mean', 12), 'weekday_12': Spec('weekday', 12),
             'ridge_36': Spec('ridge', 36, 1.0)}


def monthly_history(history):
    """Validate a complete daily spine and omit partial boundary months."""
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    if history.duplicated(['date', 'client_sector']).any():
        raise ValueError('Duplicate daily sector records')
    expected = pd.date_range(history.date.min(), history.date.max())
    for _, group in history.groupby('client_sector'):
        if not pd.DatetimeIndex(group.sort_values('date').date).equals(expected):
            raise ValueError('Missing daily coverage; do not silently fill missing data')
    history['month'] = history.date.dt.to_period('M')
    months = pd.period_range(expected[0], expected[-1], freq='M')
    months = months[(months.start_time >= expected[0]) &
                    (months.end_time.normalize() <= expected[-1])]
    frame = history[history.month.isin(months)].groupby(
        ['month', 'client_sector'], as_index=False)[['subscriptions', 'redemptions']].sum()
    return frame, months


def feature_rows(values, months, origin):
    """Use only values at/before origin, including the known seasonal reference."""
    rows = []
    for lead in range(1, 13):
        month = months[origin] + lead
        seasonal_index = origin + lead - 12  # never greater than origin
        rows.append([lead, np.sin(2*np.pi*month.month/12),
                     np.cos(2*np.pi*month.month/12), values[origin],
                     values[origin-1], np.mean(values[origin-2:origin+1]),
                     np.mean(values[origin-11:origin+1]), values[seasonal_index]])
    return np.asarray(rows, dtype=float)


def training_examples(values, months, cutoff, window=120):
    """Each example's target month must be observed by the fitting cutoff."""
    xs, ys = [], []
    for origin in range(max(11, cutoff-window), cutoff):
        features = feature_rows(values, months, origin)
        available = min(12, cutoff-origin)
        xs.extend(features[:available])
        ys.extend(values[origin+1:origin+available+1])
    if not ys:
        raise ValueError('Insufficient monthly training history')
    return np.asarray(xs), np.asarray(ys)


def forest_prediction(values, months, cutoff, setting, prepared=None):
    x, y = prepared if prepared is not None else training_examples(values, months, cutoff)
    depth, leaf = setting
    forest = RandomForestRegressor(n_estimators=200, max_depth=depth,
        min_samples_leaf=leaf, random_state=SEED, n_jobs=1)
    forest.fit(x, y)
    return forest.predict(feature_rows(values, months, cutoff))


def split_origins(count, test_count=24):
    """All 12-month validation labels precede calibration; likewise test."""
    last = count-13
    first = last-test_count+1
    validation = list(range(first-46, first-34))
    calibration = list(range(first-23, first-11))
    evaluation = list(range(first, last+1))
    if validation[0] < 60:
        raise ValueError('Insufficient history for development, calibration and testing')
    assert validation[-1]+12 <= calibration[0]
    assert calibration[-1]+12 <= evaluation[0]
    return validation, calibration, evaluation


def baseline_predictions(series, values, months, origin):
    cutoff = months[origin].end_time.normalize()
    dates = pd.date_range((months[origin]+1).start_time,
                          (months[origin]+12).end_time.normalize())
    result = {'seasonal_naive': np.asarray([values[origin+h-12] for h in range(1,13)])}
    for name, spec in BASELINES.items():
        daily, _ = predict(series, dates, cutoff, spec)
        totals = pd.Series(daily, index=dates).groupby(dates.to_period('M')).sum()
        result[name] = totals.to_numpy()
    return result


def add_derived(frame):
    """Coherent fund predictions and net flows, with identity preserved."""
    keys = ['phase', 'origin', 'target_month', 'lead_month', 'model', 'target']
    fund = frame[frame.scope == 'sector'].groupby(keys, as_index=False)[['actual', 'prediction']].sum(min_count=1)
    fund['scope'], fund['sector'] = 'fund', 'Whole fund'
    frame = pd.concat([frame, fund], ignore_index=True)
    keys = ['scope','sector','phase','origin','target_month','lead_month','model']
    subs = frame[frame.target == 'subscriptions'].set_index(keys)
    reds = frame[frame.target == 'redemptions'].set_index(keys)
    net = subs[['actual','prediction']] - reds[['actual','prediction']]
    net['target'] = 'net_flow'
    return pd.concat([frame, net.reset_index()], ignore_index=True)


def summarise(frame):
    records = []
    for key, g in frame.groupby(['scope','sector','target','model','lead_month']):
        errors = g.prediction-g.actual
        actual_scale = g.actual.abs().mean()
        records.append(dict(zip(['scope','sector','target','model','lead_month'], key),
            origins=g.origin.nunique(), mae_gbp=errors.abs().mean(),
            bias_gbp=errors.mean(), mean_actual_gbp=g.actual.mean(),
            normalised_mae=errors.abs().mean()/actual_scale if actual_scale else np.nan,
            interval_coverage=((g.actual>=g.lower)&(g.actual<=g.upper)).mean(),
            mean_interval_width=(g.upper-g.lower).mean()))
    result = pd.DataFrame(records)
    seasonal = result[result.model=='seasonal_naive'][['scope','sector','target','lead_month','mae_gbp']]
    result = result.merge(seasonal.rename(columns={'mae_gbp':'seasonal_mae_gbp'}),
                          on=['scope','sector','target','lead_month'])
    result['skill_vs_seasonal'] = 1-result.mae_gbp/result.seasonal_mae_gbp.replace(0,np.nan)
    return result


def compute(history, fingerprint):
    started = time.perf_counter()
    monthly, months = monthly_history(history)
    validation, calibration, evaluation = split_origins(len(months))
    scopes = [('sector', s, history[history.client_sector==s])
              for s in sorted(history.client_sector.unique())]
    fund = history.groupby('date',as_index=False)[['subscriptions','redemptions']].sum()
    scopes.append(('fund_direct','Whole fund',fund))
    rows, selections = [], []
    for scope, sector, data in scopes:
        for target in ('subscriptions','redemptions'):
            print(f'Monthly comparison: {sector} / {target}', flush=True)
            series = data.set_index('date')[target].astype(float).sort_index()
            values = series.groupby(series.index.to_period('M')).sum().reindex(months).to_numpy()
            prepared = {}

            def forecast_at(origin, setting):
                if origin not in prepared:
                    prepared[origin] = training_examples(values, months, origin)
                return forest_prediction(values, months, origin, setting, prepared[origin])

            scores = []
            for setting in GRID:
                errors = []
                for origin in validation:
                    predicted = forecast_at(origin, setting)
                    errors.extend(np.abs(predicted-values[origin+1:origin+13]))
                scores.append((float(np.mean(errors)), setting))
            score, setting = min(scores)
            selections.append(dict(scope=scope,sector=sector,target=target,
                max_depth=setting[0],min_samples_leaf=setting[1],validation_mae=score,
                n_estimators=200,random_seed=SEED,training_window_months=120))
            phases = [('calibration',calibration),('evaluation',evaluation),('outlook',[len(months)-1])]
            for phase, origins in phases:
                for origin in origins:
                    predictions = baseline_predictions(series,values,months,origin)
                    predictions['random_forest'] = forecast_at(origin,setting)
                    for model, path in predictions.items():
                        for lead, prediction in enumerate(path,1):
                            rows.append(dict(scope=scope,sector=sector,target=target,
                                phase=phase,origin=str(months[origin]),lead_month=lead,
                                target_month=str(months[origin]+lead),model=model,
                                prediction=float(prediction),actual=float(values[origin+lead])
                                if origin+lead<len(values) else np.nan))
    detail = add_derived(pd.DataFrame(rows))
    # Calibration is after model selection and complete before first test origin.
    keys = ['scope','sector','target','model','lead_month']
    calibration_frame = detail[detail.phase=='calibration'].copy()
    calibration_frame['error'] = calibration_frame.actual-calibration_frame.prediction
    quantiles = calibration_frame.groupby(keys).error.quantile([.1,.9]).unstack().reset_index()
    quantiles = quantiles.rename(columns={.1:'error_lower',.9:'error_upper'})
    detail = detail.merge(quantiles,on=keys,validate='many_to_one')
    detail['lower'] = detail.prediction+detail.error_lower
    detail['upper'] = detail.prediction+detail.error_upper
    amount = detail.target != 'net_flow'
    detail.loc[amount,['lower','upper']] = detail.loc[amount,['lower','upper']].clip(lower=0)
    metrics = summarise(detail[detail.phase=='evaluation'])
    # Preserve actual aggregate errors; do not sum absolute errors or bounds.
    totals = []
    for horizon in (6,12):
        selected = detail[detail.lead_month<=horizon]
        grouped = selected.groupby(['scope','sector','target','model','phase','origin'])
        sums = grouped[['actual','prediction']].sum(min_count=1)
        counts = grouped.size()
        sums = sums[counts==horizon].reset_index()
        sums['lead_month'] = horizon
        totals.append(sums)
    cumulative = pd.concat(totals,ignore_index=True)
    cal = cumulative[cumulative.phase=='calibration'].copy()
    cal['error'] = cal.actual-cal.prediction
    quantiles = cal.groupby(keys).error.quantile([.1,.9]).unstack().reset_index().rename(columns={.1:'error_lower',.9:'error_upper'})
    cumulative = cumulative.merge(quantiles,on=keys,validate='many_to_one')
    cumulative['lower'] = cumulative.prediction+cumulative.error_lower
    cumulative['upper'] = cumulative.prediction+cumulative.error_upper
    mask = cumulative.target!='net_flow'
    cumulative.loc[mask,['lower','upper']] = cumulative.loc[mask,['lower','upper']].clip(lower=0)
    cumulative_metrics = summarise(cumulative[cumulative.phase=='evaluation'])
    metadata = dict(version=VERSION,source_fingerprint=fingerprint,features=FEATURES,
        grid=GRID,seed=SEED,training_window_months=120,
        complete_coverage_start=str(months[0]),complete_coverage_end=str(months[-1]),
        observation_cutoff=str(pd.to_datetime(history.date).max().date()),
        forecast_origin=str(months[-1]),
        validation_origins=[str(months[i]) for i in validation],
        calibration_origins=[str(months[i]) for i in calibration],
        evaluation_origins=[str(months[i]) for i in evaluation],
        interval_level=.8,duration_seconds=time.perf_counter()-started,
        notes=['Retrospective research; historical results were reviewed previously.',
               'Overlapping horizons share target outcomes and are not independent.',
               'Calibration has 12 origins per lead; empirical interval estimates are imprecise.',
               'Daily current-month observations are excluded from monthly models until month is complete.',
               'Fixed baseline specifications: mean12, weekday12, ridge36 alpha1; no exhaustive baseline tuning.',
               'Forest hyperparameters selected by mean monthly absolute error across leads 1–12.',
               'Training examples share outcomes across leads; row count exceeds independent monthly observations.'])
    return dict(detail=detail,metrics=metrics,cumulative=cumulative,
                cumulative_metrics=cumulative_metrics,models=pd.DataFrame(selections),metadata=metadata)


def run(output=OUTPUT):
    with connect() as con:
        history = con.execute('SELECT date,client_sector,subscriptions,redemptions FROM daily_sector ORDER BY date,client_sector').df()
        fingerprint = con.execute('SELECT fingerprint FROM ingestion_runs ORDER BY imported_at DESC LIMIT 1').fetchone()[0]
    result = compute(history,fingerprint)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='monthly-',dir=output.parent))
    backup = output.with_name(output.name+'.backup-'+uuid.uuid4().hex)
    try:
        for name, value in result.items():
            if name=='metadata':
                (stage/'metadata.json').write_text(json.dumps(value,indent=2)+'\n')
            else:
                value.to_csv(stage/f'{name}.csv',index=False)
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result


if __name__=='__main__':
    run()
