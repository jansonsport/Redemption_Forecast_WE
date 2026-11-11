"""Performance-feature experiment for monthly-origin daily forecasts."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from .forecast import ROOT, TARGETS, features, recency_weights, training_slice
from .history_backtest import Configuration, split_months, summarize, evaluate
from .history_intervals import calibrate

OUTPUT = ROOT / 'data/clean/performance_backtest'
RETURN_WINDOWS = (5, 21, 63)
CONFIGS = {
    'subscriptions': (Configuration('random_forest', 'expanding', depth=5, leaf=12),
                      Configuration('ridge', 'rolling', alpha=.1)),
    'redemptions': (Configuration('random_forest', 'expanding', depth=5, leaf=24),
                    Configuration('ridge', 'weighted', alpha=.1, half_life=1.)),
}


def performance_features(prices, dates):
    """One vector per target month using prices available before it began."""
    if (prices.empty or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing):
        raise ValueError('Expected unique ordered valuation dates')
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError('Expected finite positive prices')
    dates = pd.DatetimeIndex(dates)
    cutoffs = dates.to_period('M').start_time - pd.Timedelta(days=1)
    rows = []
    for cutoff in cutoffs.unique():
        loc = int(prices.index.searchsorted(cutoff, side='right')) - 1
        row = dict(feature_cutoff=cutoff, price_date=prices.index[loc] if loc >= 0 else pd.NaT)
        for n in RETURN_WINDOWS:
            row[f'return_{n}'] = float(prices.iloc[loc] / prices.iloc[loc-n] - 1) if loc >= n else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).set_index('feature_cutoff').reindex(cutoffs).copy()
    frame.index = dates
    frame['feature_cutoff'] = cutoffs
    return frame


def fit_month(series, perf, month, config, augmented):
    cutoff = month.start_time - pd.Timedelta(days=1)
    dates = pd.date_range(month.start_time, month.end_time.normalize())
    columns = [f'return_{n}' for n in RETURN_WINDOWS]
    train = training_slice(series, cutoff, config.window)
    train = train[perf.reindex(train.index)[columns].notna().all(axis=1)]
    if train.empty:
        raise ValueError('Insufficient eligible training history')
    future = perf.reindex(dates)
    if future[columns].isna().any().any():
        raise ValueError('Insufficient price history at forecast origin')
    origin = train.index.min()
    trend = np.asarray((train.index-origin).days, dtype=float)
    x = features(train.index, origin, trend.mean(), max(trend.std(), 1.))
    future_x = features(dates, origin, trend.mean(), max(trend.std(), 1.))
    if augmented:
        a = perf.reindex(train.index)[columns].to_numpy()
        b = future[columns].to_numpy()
        if config.model == 'ridge':
            mean, std = a.mean(axis=0), a.std(axis=0)
            std = np.where(std > 0, std, 1.)
            a, b = (a-mean)/std, (b-mean)/std
        x, future_x = np.column_stack([x, a]), np.column_stack([future_x, b])
    if not train.any():
        prediction, status = np.zeros(len(dates)), 'all_zero_history'
    else:
        model = (Ridge(alpha=config.alpha) if config.model == 'ridge' else
                 RandomForestRegressor(n_estimators=200, max_depth=config.depth,
                                       min_samples_leaf=config.leaf, random_state=42, n_jobs=1))
        model.fit(x, train.to_numpy(), sample_weight=recency_weights(train.index, cutoff, config.half_life))
        prediction, status = np.maximum(model.predict(future_x), 0.), 'ok'
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    frame = pd.DataFrame(dict(date=dates, month=str(month), cutoff=cutoff,
        actual=series.reindex(dates).to_numpy(), prediction=prediction, status=status,
        training_start=train.index.min(), training_end=train.index.max(), training_days=len(train)))
    for col in ['feature_cutoff', 'price_date', *columns]:
        frame[col] = future[col].to_numpy()
    return frame


def paired_metrics(metrics):
    rows = []
    models = metrics[metrics.model.isin(['ridge', 'random_forest'])]
    for (period, target, model), g in models.groupby(['period', 'target', 'model']):
        control, added = (g[g.feature_set == f].iloc[0] for f in ('calendar', 'performance'))
        row = dict(period=period, target=target, model=model, history=control.history)
        for metric in ('daily_mae_gbp', 'monthly_total_mae_gbp'):
            row[metric+'_control'], row[metric+'_performance'] = control[metric], added[metric]
            row[metric+'_improvement_pct'] = 100*(control[metric]-added[metric])/control[metric] if control[metric] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute(history, prices, fingerprint):
    started = time.perf_counter()
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    _, validation, months = split_months(history)
    perf = performance_features(prices, history.date)
    frames, configurations = [], []
    for target in TARGETS:
        series = history.set_index('date')[target].astype(float)
        for config in CONFIGS[target]:
            configurations.append(dict(target=target, **asdict(config)))
            for augmented in (False, True):
                arm = 'performance' if augmented else 'calendar'
                print(f'Forecasting {target}: {config.model}/{config.approach}/{arm}', flush=True)
                frame = pd.concat([fit_month(series, perf, m, config, augmented) for m in months], ignore_index=True)
                frame['target'], frame['model'] = target, config.model
                frame['history'], frame['feature_set'] = config.approach, arm
                # Existing metric/calibration grouping includes approach; encode arm there too.
                frame['approach'], frame['phase'] = config.approach+'_'+arm, 'evaluation'
                frames.append(frame)
        for method in ('mean', 'weekday'):
            frame = evaluate(series, months, Configuration(method, 'baseline'))
            frame['target'], frame['model'] = target, method
            frame['history'], frame['feature_set'] = 'baseline', 'calendar'
            frame['approach'], frame['phase'] = 'baseline', 'evaluation'
            frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail['error'] = detail.prediction-detail.actual
    detail['abs_error'] = detail.error.abs()
    monthly = detail.groupby(['phase', 'target', 'model', 'approach', 'history', 'feature_set', 'month'], as_index=False)[['actual', 'prediction']].sum()
    monthly['error'] = monthly.prediction-monthly.actual
    monthly['abs_error'] = monthly.error.abs()
    metadata = dict(source_fingerprint=fingerprint, feature_version='calendar-plus-prior-month-return-v1',
        evaluation_months=list(map(str, months)), validation_months=list(map(str, validation)),
        calibration_months=list(map(str, months[:24])), assessment_months=list(map(str, months[24:])),
        return_windows=list(RETURN_WINDOWS), return_units='fraction, valuation-day lags',
        feature_rule='Frozen before each target month, including training targets',
        training_eligibility='Exclude early months without 63-lag price return from both arms',
        configurations=configurations, forest_trees=200, forest_seed=42, sklearn_version=sklearn.__version__,
        source_coverage_start=str(history.date.min().date()), source_coverage_end=str(history.date.max().date()),
        earliest_eligible_training_date=str(perf.dropna().index.min().date()),
        notes=['Retrospective synthetic-data experiment; configurations chosen from previously reviewed results.',
               'Fixed parameters; training-only scaling for Ridge performance columns.'])
    summaries = []
    lookup = detail[['target', 'model', 'approach', 'history', 'feature_set']].drop_duplicates()
    for name, block in [('full_155_months', detail), ('assessment_131_months', detail[detail.month.isin(metadata['assessment_months'])])]:
        metrics = summarize(block).merge(lookup, on=['target', 'model', 'approach'], validate='many_to_one')
        metrics['comparison_period'] = name
        summaries.append(metrics)
    metrics = pd.concat(summaries, ignore_index=True)
    paired = []
    for name, g in metrics.groupby('comparison_period'):
        comparison = paired_metrics(g)
        comparison['comparison_period'] = name
        paired.append(comparison)
    intervals = calibrate(detail, monthly, metadata)
    metadata['duration_seconds'] = time.perf_counter()-started
    return dict(detail=detail, monthly=monthly, metrics=metrics, comparisons=pd.concat(paired, ignore_index=True),
        features=perf.rename_axis('date').reset_index(), models=pd.DataFrame(configurations), metadata=metadata,
        **{'interval_'+name: frame for name, frame in intervals.items()})


def run(output=OUTPUT):
    from .db import connect
    previous = ROOT/'data/clean/history_window_backtest'
    old_metadata = json.loads((previous/'metadata.json').read_text())
    with connect() as con:
        history = con.execute('SELECT date,sum(subscriptions) subscriptions,sum(redemptions) redemptions FROM daily_sector GROUP BY 1 ORDER BY 1').df()
        price_frame = con.execute('SELECT date,accumulation_unit_price_gbp price FROM fund_daily ORDER BY date').df()
        fingerprint = con.execute('SELECT fingerprint FROM ingestion_runs ORDER BY imported_at DESC LIMIT 1').fetchone()[0]
    if old_metadata['source_fingerprint'] != fingerprint:
        raise ValueError('Previous results have a different source fingerprint')
    prices = pd.Series(price_frame.price.to_numpy(dtype=float), index=pd.DatetimeIndex(price_frame.date))
    result = compute(history, prices, fingerprint)
    old = pd.read_csv(previous/'detail.csv')
    references = []
    for name, block in [('full_155_months', old[old.phase == 'evaluation']),
                        ('assessment_131_months', old[(old.phase == 'evaluation') & old.month.isin(result['metadata']['assessment_months'])])]:
        scored = summarize(block)
        scored['comparison_period'] = name
        references.append(scored)
    result['previous_reference_metrics'] = pd.concat(references, ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='performance-', dir=output.parent))
    try:
        for name, value in result.items():
            if name == 'metadata':
                (stage/'metadata.json').write_text(json.dumps(value, indent=2)+'\n')
            else:
                value.to_csv(stage/f'{name}.csv', index=False)
        backup = output.with_name(output.name+'-backup-'+uuid.uuid4().hex)
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result
