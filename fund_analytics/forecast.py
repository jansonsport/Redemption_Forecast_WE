"""Reproducible, temporally evaluated cash-flow forecasts.

Run with ``python -m fund_analytics.forecast`` after importing the database.
Artifacts are published as one directory only after successful computation.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from . import analytics, db

OUTPUT = db.CLEAN_DIR / 'forecasts'
VERSION = '1'
TARGETS = ('subscriptions', 'redemptions')


@dataclass(frozen=True)
class Spec:
    method: str
    window: int
    alpha: float = 1.0


def candidates():
    return [Spec(m, w, a) for w in (12, 36, 60)
            for m in ('mean', 'weekday', 'ridge')
            for a in ((0.1, 1.0, 10.0) if m == 'ridge' else (1.0,))]


def features(dates, origin, trend_mean=0.0, trend_std=1.0):
    dates = pd.DatetimeIndex(dates)
    weekday = np.eye(7)[dates.day_of_week]
    month = np.eye(12)[dates.month - 1]
    bucket = np.eye(4)[np.minimum((dates.day - 1) // 7, 3)]
    trend = np.asarray((dates - origin).days, dtype=float)
    return np.column_stack([weekday, month, bucket, (trend - trend_mean) / trend_std])


def predict(series, dates, cutoff, spec):
    """Fit only observations on/before cutoff; return prediction and fit status."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=spec.window)
    train = series.loc[(series.index >= start) & (series.index <= cutoff)]
    if train.empty:
        raise ValueError('No observations before forecast cutoff')
    status = 'ok' if train.index.min() <= cutoff - pd.DateOffset(months=12) + pd.Timedelta(days=1) else 'insufficient_history'
    if not np.any(train.to_numpy()):
        return np.zeros(len(dates)), 'all_zero_history'
    weekday_means = train.groupby(train.index.day_of_week).mean()
    fallback = np.array([weekday_means.get(d.day_of_week, train.mean()) for d in dates])
    if spec.method == 'mean':
        pred = np.full(len(dates), train.mean())
    elif spec.method == 'weekday':
        pred = fallback
    elif spec.method == 'ridge':
        origin = train.index.min()
        trend = np.asarray((train.index - origin).days, dtype=float)
        mean, std = trend.mean(), max(trend.std(), 1.0)
        try:
            model = Ridge(alpha=spec.alpha)
            model.fit(features(train.index, origin, mean, std), train.to_numpy())
            pred = model.predict(features(dates, origin, mean, std))
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite model output')
        except (ValueError, np.linalg.LinAlgError):
            pred, status = fallback, 'weekday_fallback'
    else:
        raise ValueError(f'Unknown model: {spec.method}')
    return np.maximum(pred, 0.0), status


def predict_forest(series, dates, cutoff, window, depth, leaf):
    """Constrained calendar-feature forest for the daily whole-fund test."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=window)
    train = series.loc[(series.index >= start) & (series.index <= cutoff)]
    if train.empty:
        raise ValueError('No observations before forecast cutoff')
    if not np.any(train.to_numpy()):
        return np.zeros(len(dates))
    origin = train.index.min()
    trend = np.asarray((train.index-origin).days,dtype=float)
    mean,std = trend.mean(),max(trend.std(),1.)
    model = RandomForestRegressor(n_estimators=200,max_depth=depth,
        min_samples_leaf=leaf,random_state=42,n_jobs=1)
    model.fit(features(train.index,origin,mean,std),train.to_numpy())
    return np.maximum(model.predict(features(dates,origin,mean,std)),0.)


def run_daily_forest_backtest(output=db.CLEAN_DIR/'daily_forest_backtest'):
    """Compare daily fund RF forecasts with existing methods on identical cutoffs."""
    frame=analytics.sector_history(by_sector=False)
    saved_models=pd.read_csv(OUTPUT/'models.csv')
    frame['date']=pd.to_datetime(frame.date)
    start,end=frame.date.min(),frame.date.max()
    months=evaluation_months(start,end)
    selection,holdout=months[:24],months[24:]
    records=[]; choices=[]
    forest_specs=[(w,d,l) for w in (12,36,60) for d,l in ((3,6),(3,12),(5,12),(5,24))]
    for target in TARGETS:
        series=frame.set_index('date')[target].astype(float).sort_index()
        ranked=[]
        for window,depth,leaf in forest_specs:
            folds=[]
            for month in selection:
                dates=pd.date_range(month.start_time,month.end_time.normalize())
                cutoff=month.start_time-pd.Timedelta(days=1)
                pred=predict_forest(series,dates,cutoff,window,depth,leaf)
                folds.append(pd.DataFrame({'date':dates,'month':str(month),'actual':series.reindex(dates).values,'prediction':pred}))
            val=pd.concat(folds,ignore_index=True)
            daily_mae=float((val.actual-val.prediction).abs().mean())
            monthly=val.groupby('month')[['actual','prediction']].sum()
            monthly_mae=float((monthly.actual-monthly.prediction).abs().mean())
            ranked.append(((daily_mae,monthly_mae,window,depth,leaf),(window,depth,leaf),val))
        _,setting,validation=min(ranked,key=lambda x:x[0])
        choices.append(dict(target=target,window_months=setting[0],max_depth=setting[1],
            min_samples_leaf=setting[2],n_estimators=200,random_seed=42,
            validation_daily_mae=min(ranked,key=lambda x:x[0])[0][0],
            validation_monthly_mae=min(ranked,key=lambda x:x[0])[0][1]))
        selected=saved_models[(saved_models.scope=='fund_direct')&(saved_models.target==target)].iloc[0]
        comparisons=[('mean_12',Spec('mean',12)),('weekday_12',Spec('weekday',12)),
                     ('ridge_12',Spec('ridge',12)),('ridge_60',Spec('ridge',60)),
                     ('current_selected',Spec(str(selected.method),int(selected.window),float(selected.alpha)))]
        for model,spec in comparisons:
            test=evaluate(series,holdout,spec)
            test['model']=model;test['target']=target;test['phase']='evaluation'
            records.append(test)
        testrows=[]
        for month in holdout:
            dates=pd.date_range(month.start_time,month.end_time.normalize())
            cutoff=month.start_time-pd.Timedelta(days=1)
            pred=predict_forest(series,dates,cutoff,*setting)
            testrows.append(pd.DataFrame({'date':dates,'month':str(month),'cutoff':cutoff,
                'actual':series.reindex(dates).values,'prediction':pred,'status':'ok'}))
        test=pd.concat(testrows,ignore_index=True);test['model']='random_forest';test['target']=target;test['phase']='evaluation'
        records.append(test)
        validation['model']='random_forest';validation['target']=target;validation['phase']='validation'
        records.append(validation)
    detail=pd.concat(records,ignore_index=True)
    detail['error']=detail.prediction-detail.actual
    detail['abs_error']=detail.error.abs()
    summary=detail[detail.phase=='evaluation'].groupby(['target','model'],as_index=False).agg(
        days=('date','count'),daily_mae_gbp=('abs_error','mean'),daily_bias_gbp=('error','mean'))
    monthly=detail[detail.phase=='evaluation'].groupby(['target','model','month'],as_index=False)[['actual','prediction']].sum()
    monthly['abs_error']=(monthly.prediction-monthly.actual).abs()
    monthly_summary=monthly.groupby(['target','model'],as_index=False).agg(
        months=('month','count'),monthly_total_mae_gbp=('abs_error','mean'))
    summary=summary.merge(monthly_summary,on=['target','model'])
    summary.to_csv(Path(output).with_suffix('.csv'),index=False)
    detail.to_csv(Path(output).with_name(Path(output).name+'_detail.csv'),index=False)
    pd.DataFrame(choices).to_csv(Path(output).with_name(Path(output).name+'_models.csv'),index=False)
    return summary,detail,pd.DataFrame(choices)


def evaluation_months(start, end):
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq='M')
    complete = [p for p in months if p.start_time >= start and p.end_time.normalize() <= end]
    if len(complete) < 48:
        raise ValueError('Need at least 48 complete months for 12-month training and 36-month evaluation')
    return complete[-36:]


def evaluate(series, months, spec):
    records = []
    for month in months:
        dates = pd.date_range(month.start_time, month.end_time.normalize())
        cutoff = month.start_time - pd.Timedelta(days=1)
        prediction, status = predict(series, dates, cutoff, spec)
        records.append(pd.DataFrame({'date': dates, 'month': str(month),
                                     'cutoff': cutoff, 'actual': series.reindex(dates).to_numpy(),
                                     'prediction': prediction, 'status': status}))
    return pd.concat(records, ignore_index=True)


def choose(series, months):
    ranked = []
    for spec in candidates():
        first_cutoff = months[0].start_time - pd.Timedelta(days=1)
        if series.index.min() > first_cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=spec.window):
            continue
        frame = evaluate(series, months, spec)
        errors = frame.actual - frame.prediction
        monthly = frame.assign(error=errors).groupby('month').error.sum()
        rank = (float(errors.abs().mean()), float(monthly.abs().mean()),
                ('mean', 'weekday', 'ridge').index(spec.method), spec.window, spec.alpha)
        ranked.append((rank, spec, frame))
    if not ranked:
        spec = Spec('weekday', 12)
        return spec, evaluate(series, months, spec)
    _, spec, frame = min(ranked, key=lambda x: x[0])
    return spec, frame


def bounds(calibration, prediction, nonnegative):
    errors = calibration.actual - calibration.prediction
    qlo, qhi = np.quantile(errors.to_numpy(), [0.1, 0.9])
    lower, upper = prediction + qlo, prediction + qhi
    if nonnegative:
        lower, upper = np.maximum(lower, 0.0), np.maximum(upper, 0.0)
    return lower, upper


def monthly_totals(frame):
    return frame.groupby('month', as_index=False)[['actual', 'prediction']].sum()


def add_net(frame):
    parts = [frame]
    for (scope, sector, phase), group in frame.groupby(['scope', 'sector', 'phase'], sort=False):
        sub = group[group.target == 'subscriptions'].set_index('date')
        red = group[group.target == 'redemptions'].set_index('date')
        net = sub.copy()
        net['prediction'] = sub.prediction - red.prediction
        net['actual'] = sub.actual - red.actual
        net['target'] = 'net_flow'
        net['method'] = 'difference'
        parts.append(net.reset_index())
    return pd.concat(parts, ignore_index=True)


def enrich(calibration, heldout, current):
    """Calibrate aggregate residuals rather than summing marginal bounds."""
    daily_parts, month_parts, history_parts, metrics = [], [], [], []
    keys = ['scope', 'sector', 'target']
    for key, cal in calibration.groupby(keys, sort=False):
        mask = lambda df: np.logical_and.reduce([df[k].eq(v) for k, v in zip(keys, key)])
        test = heldout[mask(heldout)].copy()
        now = current[mask(current)].copy()
        nonnegative = key[2] != 'net_flow'
        now['lower'], now['upper'] = bounds(cal, now.prediction, nonnegative)
        now['calibration_n'] = len(cal)
        test['lower'], test['upper'] = bounds(cal, test.prediction, nonnegative)
        cal_m = monthly_totals(cal)
        now_m = monthly_totals(now)
        test_m = monthly_totals(test)
        now_m['lower'], now_m['upper'] = bounds(cal_m, now_m.prediction, nonnegative)
        now_m['calibration_n'] = len(cal_m)
        test_m['lower'], test_m['upper'] = bounds(cal_m, test_m.prediction, nonnegative)
        for k, v in zip(keys, key):
            now_m[k] = v
        errors = test.prediction - test.actual
        metrics.append(dict(zip(keys, key), daily_mae=float(errors.abs().mean()),
                            monthly_mae=float((test_m.prediction - test_m.actual).abs().mean()),
                            bias=float(errors.mean()),
                            daily_coverage=float(((test.actual >= test.lower) & (test.actual <= test.upper)).mean()),
                            monthly_coverage=float(((test_m.actual >= test_m.lower) & (test_m.actual <= test_m.upper)).mean()),
                            calibration_days=len(cal), calibration_months=len(cal_m),
                            evaluation_days=len(test), evaluation_months=len(test_m)))
        daily_parts.append(now)
        month_parts.append(now_m)
        history_parts.append(test)
    return pd.concat(daily_parts), pd.concat(month_parts), pd.concat(history_parts), pd.DataFrame(metrics)


def compute(history, fingerprint):
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    start, cutoff = history.date.min(), history.date.max()
    months = evaluation_months(start, cutoff)
    selection, holdout = months[:24], months[24:]
    next_month = cutoff.to_period('M') + 1
    future = pd.date_range(next_month.start_time, next_month.end_time.normalize())
    frames, model_rows = [], []
    sectors = sorted(history.client_sector.unique())
    scopes = [('sector', s, history[history.client_sector == s].set_index('date')) for s in sectors]
    scopes.append(('fund_direct', 'Whole fund', history.groupby('date')[list(TARGETS)].sum()))
    for scope, sector, data in scopes:
        print(f'Forecasting {sector} ({scope})', flush=True)
        for target in TARGETS:
            series = data[target].astype(float).sort_index()
            if series.index.duplicated().any() or series.isna().any() or not series.index.equals(pd.date_range(start, cutoff)):
                raise ValueError(f'Expected complete, unique daily history for {sector}/{target}')
            spec, cal = choose(series, selection)
            test = evaluate(series, holdout, spec)
            baseline = evaluate(series, holdout, Spec('mean', 12))
            pred, status = predict(series, future, cutoff, spec)
            now = pd.DataFrame({'date': future, 'month': str(next_month), 'cutoff': cutoff,
                                'prediction': pred, 'actual': np.nan, 'status': status})
            for phase, frame in [('calibration', cal), ('evaluation', test), ('outlook', now)]:
                frame['scope'], frame['sector'], frame['target'], frame['phase'] = scope, sector, target, phase
                frame['method'], frame['window'], frame['alpha'] = spec.method, spec.window, spec.alpha
                frames.append(frame)
            model_rows.append(dict(scope=scope, sector=sector, target=target, **asdict(spec), status=status,
                                   training_start=str(max(start, cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=spec.window)).date()),
                                   training_end=str(cutoff.date()),
                                   baseline_daily_mae=float((baseline.actual - baseline.prediction).abs().mean()),
                                   baseline_monthly_mae=float((monthly_totals(baseline).actual - monthly_totals(baseline).prediction).abs().mean())))
    all_frames = pd.concat(frames, ignore_index=True)
    sector_frames = all_frames[all_frames.scope == 'sector']
    fund = sector_frames.groupby(['date', 'month', 'cutoff', 'target', 'phase'], as_index=False)[['actual', 'prediction']].sum(min_count=1)
    fund['scope'], fund['sector'], fund['method'], fund['status'] = 'fund', 'Whole fund', 'sector_sum', 'ok'
    combined = add_net(pd.concat([all_frames, fund], ignore_index=True))
    daily, monthly, backtest, metrics = enrich(combined[combined.phase == 'calibration'],
                                              combined[combined.phase == 'evaluation'],
                                              combined[combined.phase == 'outlook'])
    models = pd.DataFrame(model_rows)
    metrics = metrics.merge(models[['scope', 'sector', 'target', 'baseline_daily_mae', 'baseline_monthly_mae']],
                            on=['scope', 'sector', 'target'], how='left')
    # Coherent baseline errors are evaluated from the same dates, not summed MAEs.
    baseline_frames = []
    for scope, sector, data in scopes:
        for target in TARGETS:
            b = evaluate(data[target].astype(float).sort_index(), holdout, Spec('mean', 12))
            b['scope'], b['sector'], b['target'], b['phase'] = scope, sector, target, 'evaluation'
            b['method'] = 'mean'
            baseline_frames.append(b)
    baseline_all = pd.concat(baseline_frames)
    baseline_fund = baseline_all[baseline_all.scope == 'sector'].groupby(['date','month','cutoff','target','phase'], as_index=False)[['actual','prediction']].sum()
    baseline_fund['scope'], baseline_fund['sector'], baseline_fund['method'] = 'fund','Whole fund','sector_sum'
    baseline_all = add_net(pd.concat([baseline_all, baseline_fund], ignore_index=True))
    for key, b in baseline_all.groupby(['scope','sector','target']):
        ix = (metrics.scope == key[0]) & (metrics.sector == key[1]) & (metrics.target == key[2])
        bm = monthly_totals(b)
        metrics.loc[ix, 'baseline_daily_mae'] = (b.actual-b.prediction).abs().mean()
        metrics.loc[ix, 'baseline_monthly_mae'] = (bm.actual-bm.prediction).abs().mean()
    metadata = dict(version=VERSION, source_fingerprint=fingerprint, cutoff=str(cutoff.date()),
                    forecast_month=str(next_month), coverage_start=str(start.date()),
                    selection_months=[str(p) for p in selection], evaluation_months=[str(p) for p in holdout],
                    interval_level=0.8, sectors=sectors,
                    notes=['Synthetic data; indicative empirical error ranges.',
                           'Forecasts use cash dates and exclude income distributions.',
                           'Sector classifications are static source attributes.',
                           'No observations are invented after the coverage cutoff.'])
    return dict(daily=daily.drop(columns=['actual']), monthly=monthly.drop(columns=['actual']),
                backtest=backtest, metrics=metrics, models=models, metadata=metadata)


def run_forecasts(output=OUTPUT):
    with db.connect() as con:
        history = analytics.sector_history(con)
        fingerprint = db.require_import(con)['fingerprint']
    result = compute(history, fingerprint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='forecast-', dir=output.parent))
    try:
        for name, value in result.items():
            if name == 'metadata':
                (stage / 'metadata.json').write_text(json.dumps(value, indent=2) + '\n')
            else:
                value.to_csv(stage / f'{name}.csv', index=False)
        # Offline generation is required, as for the database import.
        backup = output.with_name(output.name + '.backup-' + uuid.uuid4().hex)
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result


def load_forecasts(output=OUTPUT):
    output = Path(output)
    result = {name: pd.read_csv(output / f'{name}.csv') for name in ('daily','monthly','backtest','metrics','models')}
    for name in ('daily','backtest'):
        result[name]['date'] = pd.to_datetime(result[name].date)
        result[name]['cutoff'] = pd.to_datetime(result[name].cutoff)
    result['metadata'] = json.loads((output / 'metadata.json').read_text())
    return result


def multi_horizon_backtest(history, max_horizon=12, step_months=6):
    """Rolling-origin monthly backtest through max_horizon months.

    At each origin, select a candidate using only the preceding 24 one-month
    validation forecasts. Then compare its multi-step path with the same-month-
    last-year baseline. Origins are spaced to limit heavily overlapping tests.
    """
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    first, last = history.date.min(), history.date.max()
    complete = pd.period_range(first, last, freq='M')
    complete = [m for m in complete if m.start_time >= first and m.end_time.normalize() <= last]
    # Need 60 months for candidate training plus 24 months for validation,
    # followed by the entire forecast horizon in observed data.
    first_origin_index = 84
    last_origin_index = len(complete) - max_horizon - 1
    origin_indices = list(range(first_origin_index, last_origin_index + 1, step_months))
    if len(origin_indices) < 4:
        raise ValueError('Insufficient complete history for a useful multi-horizon backtest')

    sectors = sorted(history.client_sector.unique())
    scopes = [('sector', s, history[history.client_sector == s]) for s in sectors]
    scopes.append(('fund', 'Whole fund', history.groupby('date', as_index=False)[list(TARGETS)].sum()))
    records = []
    # Keep the inner selection compact: the point of this backtest is the
    # horizon curve, not a second exhaustive tuning exercise.
    specs = [Spec('mean', 12), Spec('weekday', 12), Spec('ridge', 36, 1.0)]
    for scope, name, frame in scopes:
        data = frame.groupby('date')[list(TARGETS)].sum().sort_index()
        for target in TARGETS:
            series = data[target].astype(float).asfreq('D', fill_value=0)
            for oi in origin_indices:
                origin_month = complete[oi]
                cutoff = origin_month.end_time.normalize()
                # Select using 24 preceding one-month forecasts; validation
                # outcomes occur before this origin and cannot leak forward.
                validation_months = complete[oi-24:oi]
                ranked = []
                for spec in specs:
                    try:
                        val = evaluate(series, validation_months, spec)
                    except ValueError:
                        continue
                    err = val.actual - val.prediction
                    monthly_error = val.assign(err=err).groupby('month').err.sum()
                    ranked.append(((float(err.abs().mean()), float(monthly_error.abs().mean()),
                                    ('mean','weekday','ridge').index(spec.method), spec.window, spec.alpha), spec))
                if not ranked:
                    continue
                spec = min(ranked, key=lambda x:x[0])[1]
                future_months = complete[oi+1:oi+1+max_horizon]
                future_dates = pd.date_range(future_months[0].start_time, future_months[-1].end_time.normalize())
                prediction, _ = predict(series, future_dates, cutoff, spec)
                pred_series = pd.Series(prediction, index=future_dates)
                for lead, month in enumerate(future_months, start=1):
                    actual = float(series.loc[month.start_time:month.end_time].sum())
                    predicted = float(pred_series.loc[month.start_time:month.end_time].sum())
                    prior_year = month - 12
                    baseline = float(series.loc[prior_year.start_time:prior_year.end_time].sum())
                    records.append({'scope':scope,'sector':name,'target':target,
                        'origin':str(cutoff.date()),'lead_month':lead,'target_month':str(month),
                        'actual':actual,'prediction':predicted,'baseline':baseline,
                        'method':spec.method,'window_months':spec.window,'alpha':spec.alpha})
    detail = pd.DataFrame(records)
    if detail.empty:
        raise ValueError('No multi-horizon backtest cases were produced')
    summary = detail.groupby(['scope','sector','target','lead_month'],as_index=False).apply(
        lambda g: pd.Series({'sample_origins':g.origin.nunique(),
            'model_mae_gbp':(g.prediction-g.actual).abs().mean(),
            'baseline_mae_gbp':(g.baseline-g.actual).abs().mean(),
            'model_bias_gbp':(g.prediction-g.actual).mean(),
            'model_skill_vs_baseline':1-(g.prediction-g.actual).abs().mean()/(g.baseline-g.actual).abs().mean() if (g.baseline-g.actual).abs().mean() else np.nan,
            'mean_actual_gbp':g.actual.mean(),'mean_prediction_gbp':g.prediction.mean()})).reset_index(drop=True)
    return detail, summary


def run_multi_horizon_backtest(output=OUTPUT):
    """Create separate, reproducible multi-horizon backtest artifacts."""
    history=analytics.sector_history()
    detail, summary=multi_horizon_backtest(history)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    detail.to_csv(output/'horizon_backtest_detail.csv',index=False)
    summary.to_csv(output/'horizon_backtest_summary.csv',index=False)
    return detail,summary


if __name__ == '__main__':
    result = run_forecasts()
    print(json.dumps(result['metadata'], indent=2))
