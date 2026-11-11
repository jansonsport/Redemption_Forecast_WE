"""Offline outlooks from the selected calendar-only whole-fund models."""
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd

from .forecast import ROOT, TARGETS, Spec, predict, predict_forest, training_slice
from .history_intervals import make_ranges

OUTPUT = ROOT / 'data/clean/selected_forecasts'
VERSION = 'selected-history-v2'
FAMILIES = ('random_forest', 'ridge')


def select_daily_models(metrics):
    """Choose reviewed daily-error leaders within each family and flow."""
    winners = {}
    approach_order = {'rolling': 0, 'expanding': 1, 'weighted': 2}
    for family in FAMILIES:
        winners[family] = {}
        for target in TARGETS:
            eligible = metrics[(metrics.period == 'overall') & (metrics.model == family) &
                               (metrics.target == target)].copy()
            if eligible.empty:
                raise ValueError(f'Missing daily selection metrics for {family}/{target}')
            eligible['approach_order'] = eligible.approach.map(approach_order)
            if eligible[['daily_mae_gbp', 'monthly_total_mae_gbp', 'approach_order']].isna().any().any():
                raise ValueError('Invalid daily selection metrics')
            selected = eligible.sort_values(['daily_mae_gbp', 'monthly_total_mae_gbp', 'approach_order']).iloc[0]
            winners[family][target] = str(selected.approach)
    return winners



def add_net(frame, keys):
    """Derive net actuals/predictions before calibrating aggregate error ranges."""
    sub = frame[frame.target == 'subscriptions'].set_index(keys)
    red = frame[frame.target == 'redemptions'].set_index(keys)
    if not sub.index.equals(red.index):
        raise ValueError('Flow dates do not align')
    net = sub.copy()
    for col in ('prediction', 'actual'):
        if col in frame:
            net[col] = sub[col] - red[col]
    net['target'] = 'net_flow'
    return pd.concat([frame, net.reset_index()], ignore_index=True)


def compute(history, models, backtest, source_metadata, selection_metrics):
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    dates = pd.DatetimeIndex(history.date)
    if dates.has_duplicates or not dates.equals(pd.date_range(dates.min(), dates.max())):
        raise ValueError('Expected complete ordered daily history')
    if not np.isfinite(history[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite flow history')
    cutoff = dates.max()
    if source_metadata['coverage_end'] != str(cutoff.date()):
        raise ValueError('Backtest history is stale; regenerate the history-window experiment')
    next_month = cutoff.to_period('M') + 1
    future_dates = pd.date_range(next_month.start_time, next_month.end_time.normalize())
    calibration_months = source_metadata['evaluation_months'][:24]
    assessment_months = source_metadata['evaluation_months'][24:]
    if not assessment_months:
        raise ValueError('Missing interval assessment period')
    outputs, metric_rows, model_rows = [], [], []
    winners = select_daily_models(selection_metrics)
    for family, flows in winners.items():
        objective = 'daily'
        current_parts, historical_parts = [], []
        for target, approach in flows.items():
            model = family
            selected = models[(models.target == target) & (models.model == model) & (models.approach == approach)]
            if len(selected) != 1:
                raise ValueError(f'Missing unique configuration for {objective}/{target}')
            config = selected.iloc[0]
            series = history.set_index('date')[target].astype(float)
            window = 60 if approach == 'rolling' else None
            half_life = None if pd.isna(config.half_life) else float(config.half_life)
            train = training_slice(series, cutoff, window)
            if model == 'random_forest':
                prediction = predict_forest(series, future_dates, cutoff, window,
                                            int(config.depth), int(config.leaf), half_life)
                status = 'all_zero_history' if not train.any() else 'ok'
            else:
                prediction, status = predict(series, future_dates, cutoff,
                                              Spec('ridge', window, float(config.alpha)), half_life)
            current_parts.append(pd.DataFrame(dict(date=future_dates, month=str(next_month),
                target=target, prediction=prediction, model=model, approach=approach)))
            past = backtest[(backtest.phase == 'evaluation') & (backtest.target == target) &
                            (backtest.model == model) & (backtest.approach == approach)].copy()
            if set(past.month) != set(source_metadata['evaluation_months']):
                raise ValueError('Incomplete selected-model evaluation history')
            historical_parts.append(past[['date', 'month', 'target', 'actual', 'prediction']])
            model_rows.append(dict(objective=objective, target=target, model=model, history=approach,
                alpha=float(config.alpha) if model == 'ridge' else None,
                max_depth=int(config.depth) if model == 'random_forest' else None,
                min_samples_leaf=int(config.leaf) if model == 'random_forest' else None,
                trees=200 if model == 'random_forest' else None, random_seed=42 if model == 'random_forest' else None,
                half_life_years=half_life, training_start=str(train.index.min().date()),
                training_end=str(train.index.max().date()), status=status))
        current = add_net(pd.concat(current_parts, ignore_index=True), ['date', 'month'])
        historical = add_net(pd.concat(historical_parts, ignore_index=True), ['date', 'month'])
        ranged_parts = []
        for target, g in historical.groupby('target'):
            cal = g[g.month.isin(calibration_months)]
            test = g[g.month.isin(assessment_months)]
            # Net-flow bounds retain negative values; amount bounds are nonnegative.
            scale = 'daily_amount' if target != 'net_flow' else 'net_flow'
            now = current[current.target == target].copy()
            ranged_parts.append(make_ranges(cal, now, scale))
            assessed = make_ranges(cal, test, scale)
            metric_rows.append(dict(objective=objective, model=family, target=target,
                full_period_mae_gbp=float((g.actual-g.prediction).abs().mean()),
                assessment_mae_gbp=float((test.actual-test.prediction).abs().mean()),
                assessment_bias_gbp=float((test.prediction-test.actual).mean()),
                coverage=float(((assessed.actual>=assessed.lower)&(assessed.actual<=assessed.upper)).mean()),
                mean_width_gbp=float((assessed.upper-assessed.lower).mean()),
                assessment_cases=len(test), calibration_cases=len(cal), nominal_coverage=.8))
        outputs.append(pd.concat(ranged_parts, ignore_index=True))
    metadata = dict(version=VERSION, source_fingerprint=source_metadata['source_fingerprint'], cutoff=str(cutoff.date()),
        forecast_month=str(next_month), feature_version='calendar-v1', scope='Whole fund direct',
        evaluation_months=source_metadata['evaluation_months'], calibration_months=calibration_months,
        assessment_months=assessment_months, daily_family_winners=winners,
        selection_rule='Reviewed overall daily MAE, monthly-total MAE, rolling/expanding/weighted order',
        notes=['Calendar features only; no fund-performance inputs.',
               'Daily Random Forest and Ridge use independently selected histories per flow.',
               'Historical synthetic-data results were reviewed before model adoption.',
               'Empirical ranges have coverage below 80%; display as diagnostics only.',
               'No observations invented between cutoff and forecast month.'])
    return dict(daily=pd.concat(outputs, ignore_index=True), metrics=pd.DataFrame(metric_rows),
                models=pd.DataFrame(model_rows), metadata=metadata)


def run(output=OUTPUT):
    from .db import connect
    source = ROOT / 'data/clean/history_window_backtest'
    metadata = json.loads((source/'metadata.json').read_text())
    with connect() as con:
        fingerprint = con.execute('SELECT fingerprint FROM ingestion_runs ORDER BY imported_at DESC LIMIT 1').fetchone()[0]
        history = con.execute('SELECT date,sum(subscriptions) subscriptions,sum(redemptions) redemptions FROM daily_sector GROUP BY 1 ORDER BY 1').df()
    if metadata['source_fingerprint'] != fingerprint:
        raise ValueError('Backtest source differs from the current import')
    result = compute(history, pd.read_csv(source/'models.csv'),
                     pd.read_csv(source/'detail.csv'), metadata, pd.read_csv(source/'metrics.csv'))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='selected-forecasts-', dir=output.parent))
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
