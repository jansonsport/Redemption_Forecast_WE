"""Direct monthly forecasts with chronological model selection and calibration.

All horizons are learned together by a constrained forest, with lead month as
a feature. Examples sharing an origin are never split randomly. Only examples
whose target outcome is observed by the fitting cutoff enter training.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
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
VERSION = 'monthly-history-v2'
SEED = 42
GRID = [(3, 6), (3, 12), (5, 6), (5, 12)]
FEATURES = ['lead', 'month_sin', 'month_cos', 'previous', 'previous_2',
            'mean_3', 'mean_12', 'same_month_last_year']
BASELINES = {'mean_12': Spec('mean', 12), 'weekday_12': Spec('weekday', 12)}


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
    extra = ['horizon'] if 'horizon' in frame else []
    keys = extra + ['phase', 'origin', 'target_month', 'lead_month', 'model', 'target']
    fund = frame[frame.scope == 'sector'].groupby(keys, as_index=False)[['actual', 'prediction']].sum(min_count=1)
    fund['scope'], fund['sector'] = 'fund', 'Whole fund'
    frame = pd.concat([frame, fund], ignore_index=True)
    keys = extra + ['scope','sector','phase','origin','target_month','lead_month','model']
    subs = frame[frame.target == 'subscriptions'].set_index(keys)
    reds = frame[frame.target == 'redemptions'].set_index(keys)
    net = subs[['actual','prediction']] - reds[['actual','prediction']]
    net['target'] = 'net_flow'
    return pd.concat([frame, net.reset_index()], ignore_index=True)


def summarise(frame):
    records = []
    keys = (['horizon'] if 'horizon' in frame else []) + ['scope','sector','target','model','lead_month']
    for key, g in frame.groupby(keys):
        errors = g.prediction-g.actual
        actual_scale = g.actual.abs().mean()
        records.append(dict(zip(keys, key),
            origins=g.origin.nunique(), mae_gbp=errors.abs().mean(),
            bias_gbp=errors.mean(), mean_actual_gbp=g.actual.mean(),
            normalised_mae=errors.abs().mean()/actual_scale if actual_scale else np.nan,
            interval_coverage=((g.actual>=g.lower)&(g.actual<=g.upper)).mean(),
            mean_interval_width=(g.upper-g.lower).mean()))
    result = pd.DataFrame(records)
    join_keys = [k for k in keys if k != 'model']
    seasonal = result[result.model=='seasonal_naive'][join_keys+['mae_gbp']]
    result = result.merge(seasonal.rename(columns={'mae_gbp':'seasonal_mae_gbp'}),
                          on=join_keys)
    result['skill_vs_seasonal'] = 1-result.mae_gbp/result.seasonal_mae_gbp.replace(0,np.nan)
    return result


@dataclass(frozen=True)
class MonthlySpec:
    model: str
    history: str
    alpha: float = 1.0
    depth: int = 3
    leaf: int = 6
    half_life: float | None = None

    @property
    def window(self):
        return 60 if self.history == 'rolling' else None


def model_candidates(model):
    for history in ('rolling', 'expanding', 'weighted'):
        for half_life in ((1., 3., 5.) if history == 'weighted' else (None,)):
            if model == 'ridge':
                for alpha in (.1, 1., 10.):
                    yield MonthlySpec(model, history, alpha=alpha, half_life=half_life)
            else:
                for depth, leaf in GRID:
                    yield MonthlySpec(model, history, depth=depth, leaf=leaf, half_life=half_life)


def history_examples(values, months, cutoff, window=None):
    """Training targets lie within the window; feature lookbacks may precede it."""
    xs, ys, ages = [], [], []
    first_target = max(12, cutoff-window+1) if window is not None else 12
    for origin in range(max(11, first_target-12), cutoff):
        x = feature_rows(values, months, origin)
        for lead in range(1, min(12, cutoff-origin)+1):
            target = origin+lead
            if target >= first_target:
                xs.append(x[lead-1]); ys.append(values[target]); ages.append(cutoff-target)
    if not ys:
        raise ValueError('Insufficient eligible monthly examples')
    return np.asarray(xs), np.asarray(ys), np.asarray(ages)


def monthly_weights(ages, half_life):
    if half_life is None:
        return None
    weights = np.exp2(-np.asarray(ages)/(12*half_life))
    return weights/weights.mean()


def history_prediction(series, values, months, origin, config, prepared=None):
    if config.model == 'ridge':
        dates = pd.date_range((months[origin]+1).start_time, (months[origin]+12).end_time.normalize())
        daily, _ = predict(series, dates, months[origin].end_time.normalize(),
                           Spec('ridge', config.window, config.alpha), config.half_life)
        return pd.Series(daily, index=dates).groupby(dates.to_period('M')).sum().to_numpy()
    x, y, ages = prepared if prepared is not None else history_examples(values, months, origin, config.window)
    forest = RandomForestRegressor(n_estimators=200, max_depth=config.depth,
        min_samples_leaf=config.leaf, random_state=SEED, n_jobs=1)
    forest.fit(x, y, sample_weight=monthly_weights(ages, config.half_life))
    return np.maximum(forest.predict(feature_rows(values, months, origin)), 0.)


def selection_rank(paths, actual, horizon, order):
    errors = np.asarray(paths)[:, :horizon]-np.asarray(actual)[:, :horizon]
    return float(np.abs(errors.sum(axis=1)).mean()), float(np.abs(errors).mean()), order


def _scope_comparison(args):
    # Keep numerical kernels bounded while two independent scopes run concurrently.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _scope_comparison_inner(args)


def _scope_comparison_inner(args):
    scope, sector, target, data, months, validation, calibration, evaluation = args
    print(f'Monthly history comparison: {scope}/{sector}/{target}', flush=True)
    series = data.set_index('date')[target].astype(float).sort_index()
    series.index = pd.to_datetime(series.index)
    values = series.groupby(series.index.to_period('M')).sum().reindex(months).to_numpy()
    prepared, cached = {}, {}
    def path(origin, config):
        key = (origin, config)
        if key not in cached:
            examples = None
            if config.model == 'random_forest':
                example_key = (origin, config.window)
                if example_key not in prepared:
                    prepared[example_key] = history_examples(values, months, origin, config.window)
                examples = prepared[example_key]
            cached[key] = history_prediction(series, values, months, origin, config, examples)
        return cached[key]
    selected, candidates = {}, []
    actual = np.asarray([values[o+1:o+13] for o in validation])
    for family in ('random_forest', 'ridge'):
        rankings = {h: [] for h in (1, 6, 12)}
        for order, config in enumerate(model_candidates(family)):
            paths = [path(origin, config) for origin in validation]
            for horizon in rankings:
                rank = selection_rank(paths, actual, horizon, order)
                rankings[horizon].append((rank, config))
                candidates.append(dict(scope=scope, sector=sector, target=target, horizon=horizon,
                    **asdict(config), validation_total_mae=rank[0], validation_monthly_mae=rank[1], candidate_order=order))
        for horizon, scores in rankings.items():
            selected[(family, horizon)] = min(scores, key=lambda x:x[0])
    rows, selections = [], []
    for (family, horizon), (rank, config) in selected.items():
        selections.append(dict(scope=scope, sector=sector, target=target, horizon=horizon,
            **asdict(config), window_months=config.window, trees=200 if family=='random_forest' else None,
            seed=SEED, validation_total_mae=rank[0], validation_monthly_mae=rank[1]))
    for phase, origins in [('calibration', calibration), ('evaluation', evaluation), ('outlook', [len(months)-1])]:
        for origin in origins:
            baselines = baseline_predictions(series, values, months, origin)
            # Retain the old fixed Ridge only as a historical reference artifact.
            baselines.pop('ridge_36', None)
            for horizon in (1, 6, 12):
                predictions = {name:p for name,p in baselines.items()}
                for family in ('random_forest', 'ridge'):
                    predictions[family] = path(origin, selected[(family, horizon)][1])
                for model, predicted in predictions.items():
                    for lead in range(1, horizon+1):
                        rows.append(dict(scope=scope, sector=sector, target=target, horizon=horizon,
                            phase=phase, origin=str(months[origin]), lead_month=lead,
                            target_month=str(months[origin]+lead), model=model,
                            prediction=float(predicted[lead-1]), actual=float(values[origin+lead])
                            if origin+lead < len(values) else np.nan))
    return rows, selections, candidates


def compute(history, fingerprint, workers=2):
    started = time.perf_counter()
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    _, months = monthly_history(history)
    validation, calibration, evaluation = split_origins(len(months))
    scopes = [('sector', s, history[history.client_sector==s]) for s in sorted(history.client_sector.unique())]
    scopes.append(('fund_direct', 'Whole fund', history.groupby('date', as_index=False)[['subscriptions','redemptions']].sum()))
    jobs = [(scope, sector, target, data, months, validation, calibration, evaluation)
            for scope, sector, data in scopes for target in ('subscriptions','redemptions')]
    if workers == 1:
        results = list(map(_scope_comparison, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_scope_comparison, jobs))
    rows = [r for result in results for r in result[0]]
    selections = [r for result in results for r in result[1]]
    candidates = [r for result in results for r in result[2]]
    detail = add_derived(pd.DataFrame(rows))
    keys = ['horizon','scope','sector','target','model','lead_month']
    def calibrate_ranges(frame):
        cal = frame[frame.phase=='calibration'].copy()
        cal['error'] = cal.actual-cal.prediction
        quantiles = cal.groupby(keys).error.quantile([.1,.9]).unstack().reset_index().rename(columns={.1:'error_lower',.9:'error_upper'})
        result = frame.merge(quantiles, on=keys, validate='many_to_one')
        result['lower'] = result.prediction+result.error_lower
        result['upper'] = result.prediction+result.error_upper
        mask = result.target!='net_flow'
        result.loc[mask,['lower','upper']] = result.loc[mask,['lower','upper']].clip(lower=0)
        return result
    detail = calibrate_ranges(detail)
    keys_total = ['horizon','scope','sector','target','model','phase','origin']
    cumulative = detail.groupby(keys_total, as_index=False)[['actual','prediction']].sum(min_count=1)
    cumulative['lead_month'] = cumulative.horizon
    cumulative = calibrate_ranges(cumulative)
    metadata = dict(version=VERSION, source_fingerprint=fingerprint, features=FEATURES,
        grid=GRID, seed=SEED, history_candidates=['rolling60','expanding','weighted_expanding'],
        half_life_candidates=[1,3,5], ridge_alpha_candidates=[.1,1,10], horizons=[1,6,12],
        complete_coverage_start=str(months[0]), complete_coverage_end=str(months[-1]),
        observation_cutoff=str(history.date.max().date()), forecast_origin=str(months[-1]),
        validation_origins=[str(months[i]) for i in validation],
        calibration_origins=[str(months[i]) for i in calibration],
        evaluation_origins=[str(months[i]) for i in evaluation], interval_level=.8,
        duration_seconds=time.perf_counter()-started,
        selection='validation cumulative-total MAE, mean monthly MAE, deterministic candidate order',
        notes=['Retrospective synthetic-data research; previous outcomes were reviewed.',
               'Fund paths sum sector predictions; direct fund fits are separate comparisons.',
               'Each horizon uses its own selected configuration within each family.',
               'Only training targets available at cutoff are eligible; forests weight target-month age.',
               'Calibration has 12 origins per lead; intervals are imprecise and not guaranteed.',
               'Overlapping evaluation paths are dependent.',
               'Fund performance features are excluded.'])
    return dict(detail=detail, metrics=summarise(detail[detail.phase=='evaluation']),
        cumulative=cumulative, cumulative_metrics=summarise(cumulative[cumulative.phase=='evaluation']),
        models=pd.DataFrame(selections), candidate_scores=pd.DataFrame(candidates), metadata=metadata)


def run(output=OUTPUT):
    with connect() as con:
        history = con.execute('SELECT date,client_sector,subscriptions,redemptions FROM daily_sector ORDER BY date,client_sector').df()
        fingerprint = con.execute('SELECT fingerprint FROM ingestion_runs ORDER BY imported_at DESC LIMIT 1').fetchone()[0]
    references = {}
    existing = Path(output)
    if (existing/'metadata.json').exists():
        previous_meta = json.loads((existing/'metadata.json').read_text())
        if previous_meta['source_fingerprint'] != fingerprint:
            raise ValueError('Previous monthly comparison uses a different source import')
        prefix = '' if previous_meta.get('version') != VERSION else 'previous_'
        for name in ('metrics', 'cumulative_metrics', 'models'):
            if (existing/f'{prefix}{name}.csv').exists():
                references['previous_'+name] = pd.read_csv(existing/f'{prefix}{name}.csv')
        if prefix:
            if (existing/'previous_metadata.json').exists():
                references['previous_metadata'] = json.loads((existing/'previous_metadata.json').read_text())
        else:
            references['previous_metadata'] = previous_meta
    result = compute(history,fingerprint)
    result.update(references)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='monthly-',dir=output.parent))
    backup = output.with_name(output.name+'.backup-'+uuid.uuid4().hex)
    try:
        for name, value in result.items():
            if isinstance(value, dict):
                (stage/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
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
