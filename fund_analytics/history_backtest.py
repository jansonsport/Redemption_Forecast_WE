"""Retrospective daily forecasts comparing training-history strategies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import sklearn

from . import analytics, db
from .forecast import TARGETS, Spec, predict, predict_forest, training_slice

OUTPUT = db.CLEAN_DIR / 'history_window_backtest'
APPROACHES = ('rolling', 'expanding', 'weighted')


@dataclass(frozen=True)
class Configuration:
    model: str
    approach: str
    alpha: float = 1.0
    depth: int = 3
    leaf: int = 6
    half_life: float | None = None

    @property
    def window(self):
        return 60 if self.approach == 'rolling' else None


def split_months(history):
    """Require a unique complete calendar, then reserve 60 warmup/24 validation months."""
    dates = pd.DatetimeIndex(history.date)
    if dates.empty or dates.has_duplicates or not dates.equals(pd.date_range(dates.min(), dates.max())):
        raise ValueError('Expected complete, unique, ordered daily history')
    values = history[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative flow amounts')
    complete = pd.period_range(dates.min(), dates.max(), freq='M')
    complete = [m for m in complete if m.start_time >= dates.min() and m.end_time.normalize() <= dates.max()]
    if len(complete) < 85:
        raise ValueError('Need 60 warmup, 24 validation and at least one evaluation month')
    return complete[:60], complete[60:84], complete[84:]


def candidates(model, approach):
    half_lives = (1., 3., 5.) if approach == 'weighted' else (None,)
    for half_life in half_lives:
        if model == 'ridge':
            for alpha in (.1, 1., 10.):
                yield Configuration(model, approach, alpha=alpha, half_life=half_life)
        else:
            for depth, leaf in ((3, 6), (3, 12), (5, 12), (5, 24)):
                yield Configuration(model, approach, depth=depth, leaf=leaf, half_life=half_life)


def evaluate(series, months, config):
    rows = []
    for month in months:
        cutoff = month.start_time - pd.Timedelta(days=1)
        dates = pd.date_range(month.start_time, month.end_time.normalize())
        train = training_slice(series, cutoff, config.window)
        if config.model == 'random_forest':
            prediction = predict_forest(series, dates, cutoff, config.window, config.depth,
                                        config.leaf, config.half_life)
            status = 'all_zero_history' if not train.any() else 'ok'
        else:
            window = 12 if config.model in ('mean', 'weekday') else config.window
            train = training_slice(series, cutoff, window)
            prediction, status = predict(series, dates, cutoff,
                                         Spec(config.model, window, config.alpha), config.half_life)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite prediction')
        rows.append(pd.DataFrame(dict(date=dates, month=str(month), cutoff=cutoff,
            training_start=train.index.min(), training_end=train.index.max(), training_days=len(train),
            actual=series.reindex(dates).to_numpy(), prediction=prediction, status=status)))
    return pd.concat(rows, ignore_index=True)


def errors(frame):
    daily = float((frame.prediction - frame.actual).abs().mean())
    monthly = frame.groupby('month')[['actual', 'prediction']].sum()
    return daily, float((monthly.prediction - monthly.actual).abs().mean())


def select(series, months, model, approach):
    best = None
    for order, config in enumerate(candidates(model, approach)):
        frame = evaluate(series, months, config)
        rank = (*errors(frame), order)
        if best is None or rank < best[0]:
            best = rank, config, frame
    return best[1], best[2]


def summarize(detail):
    detail = detail.copy()
    detail['year'] = pd.to_datetime(detail.date).dt.year
    rows = []
    for period, block in [('overall', detail), *[(str(y), g) for y, g in detail.groupby('year')]]:
        for (target, model, approach), g in block.groupby(['target', 'model', 'approach']):
            daily, monthly = errors(g)
            mean = float(g.actual.mean())
            rows.append(dict(period=period, target=target, model=model, approach=approach,
                days=len(g), months=g.month.nunique(), daily_mae_gbp=daily,
                monthly_total_mae_gbp=monthly, daily_bias_gbp=float((g.prediction-g.actual).mean()),
                average_actual_daily_gbp=mean, daily_mae_pct_of_mean_actual=100*daily/mean if mean else np.nan))
    metrics = pd.DataFrame(rows)
    for baseline in ('mean', 'weekday'):
        base = metrics[metrics.model == baseline].set_index(['period', 'target'])
        for metric in ('daily_mae_gbp', 'monthly_total_mae_gbp'):
            denominators = pd.MultiIndex.from_frame(metrics[['period', 'target']]).map(base[metric])
            den = np.asarray(denominators, dtype=float)
            metrics[f'{metric}_improvement_pct_vs_{baseline}'] = np.divide(
                100*(den-metrics[metric]), den, out=np.full(len(metrics), np.nan), where=den != 0)
    return metrics


def paired_comparisons(metrics):
    rows = []
    models = metrics[metrics.model.isin(['ridge', 'random_forest'])]
    for (period, target, model), g in models.groupby(['period', 'target', 'model']):
        g = g.set_index('approach')
        for left, right in (('expanding', 'rolling'), ('weighted', 'rolling'), ('weighted', 'expanding')):
            row = dict(period=period, target=target, model=model, approach=left, compared_with=right)
            for metric in ('daily_mae_gbp', 'monthly_total_mae_gbp'):
                reference = g.loc[right, metric]
                row[f'{metric}_difference'] = g.loc[left, metric] - reference
                row[f'{metric}_improvement_pct'] = 100*(reference-g.loc[left, metric])/reference if reference else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compute(history, fingerprint):
    started = time.perf_counter()
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    warmup, validation, evaluation = split_months(history)
    frames, choices = [], []
    for target in TARGETS:
        series = history.set_index('date')[target].astype(float)
        for model in ('ridge', 'random_forest'):
            for approach in APPROACHES:
                print(f'Selecting {target}: {model}/{approach}', flush=True)
                config, selected = select(series, validation, model, approach)
                daily, monthly = errors(selected)
                choices.append(dict(target=target, **asdict(config), validation_daily_mae=daily,
                                    validation_monthly_total_mae=monthly))
                test = evaluate(series, evaluation, config)
                for phase, frame in (('validation', selected), ('evaluation', test)):
                    frame['target'], frame['model'], frame['approach'], frame['phase'] = target, model, approach, phase
                    frames.append(frame)
        for model in ('mean', 'weekday'):
            frame = evaluate(series, evaluation, Configuration(model, 'baseline'))
            frame['target'], frame['model'], frame['approach'], frame['phase'] = target, model, 'baseline', 'evaluation'
            frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail['error'] = detail.prediction - detail.actual
    detail['abs_error'] = detail.error.abs()
    test = detail[detail.phase == 'evaluation']
    metrics = summarize(test)
    monthly = detail.groupby(['phase', 'target', 'model', 'approach', 'month'], as_index=False)[['actual', 'prediction']].sum()
    monthly['error'] = monthly.prediction - monthly.actual
    monthly['abs_error'] = monthly.error.abs()
    metadata = dict(source_fingerprint=fingerprint, feature_version='calendar-v1',
        coverage_start=str(history.date.min().date()), coverage_end=str(history.date.max().date()),
        warmup_months=list(map(str, warmup)), validation_months=list(map(str, validation)),
        evaluation_months=list(map(str, evaluation)), trees=200, random_seed=42,
        weight_formula='2 ** (-age_days / (365.25 * half_life_years)), normalized to mean 1',
        half_life_candidates=[1,3,5], alpha_candidates=[.1,1,10],
        forest_candidates=[[3,6],[3,12],[5,12],[5,24]],
        selection='daily MAE, monthly-total MAE, listed candidate order',
        sklearn_version=sklearn.__version__, duration_seconds=time.perf_counter()-started,
        notes=['Retrospective synthetic-data experiment; later historical outcomes have previously been reviewed.',
               'Whole-fund cash flows; complete calendar months; partial final month excluded.',
               'Point forecasts only; existing dashboard forecasts are unchanged.'])
    return dict(detail=detail, monthly=monthly, metrics=metrics, models=pd.DataFrame(choices),
                comparisons=paired_comparisons(metrics), metadata=metadata)


def run(output=OUTPUT):
    with db.connect() as con:
        history = analytics.sector_history(con, by_sector=False)
        fingerprint = db.require_import(con)['fingerprint']
    result = compute(history, fingerprint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='history-backtest-', dir=output.parent))
    try:
        for name, frame in result.items():
            if name == 'metadata':
                (stage/'metadata.json').write_text(json.dumps(frame, indent=2)+'\n')
            else:
                frame.to_csv(stage/f'{name}.csv', index=False)
        backup = output.with_name(output.name + '-backup-' + uuid.uuid4().hex)
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
