"""Calibrate and evaluate indicative 80% ranges for history-window forecasts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd

from .history_backtest import OUTPUT
from . import analytics, db

INTERVAL_OUTPUT = OUTPUT / 'intervals'
LEVEL = .8
CALIBRATION_MONTHS = 24


def make_ranges(calibration, evaluation, scale):
    """Apply central residual quantiles, calibrated independently per scale."""
    errors = calibration.actual.to_numpy() - calibration.prediction.to_numpy()
    lower_error, upper_error = np.quantile(errors, [(1-LEVEL)/2, 1-(1-LEVEL)/2])
    result = evaluation.copy()
    result['lower'] = result.prediction + lower_error
    result['upper'] = result.prediction + upper_error
    if scale in ('daily_amount', 'monthly_total'):
        result[['lower','upper']] = result[['lower','upper']].clip(lower=0.)
    result['calibration_error_lower'] = lower_error
    result['calibration_error_upper'] = upper_error
    result['calibration_n'] = len(errors)
    result['interval_level'] = LEVEL
    return result


def score(frame, scale):
    covered = (frame.actual >= frame.lower) & (frame.actual <= frame.upper)
    result = frame.assign(covered=covered, width=frame.upper-frame.lower)
    metrics = result.groupby(['target','model','approach'],as_index=False).agg(
        observations=('actual','size'), coverage=('covered','mean'),
        mean_width_gbp=('width','mean'), median_width_gbp=('width','median'),
        calibration_n=('calibration_n','first'),
        calibration_error_lower=('calibration_error_lower','first'),
        calibration_error_upper=('calibration_error_upper','first'))
    metrics['scale'] = scale
    metrics['nominal_coverage'] = LEVEL
    metrics['coverage_gap_pp'] = 100*(metrics.coverage-LEVEL)
    result['year'] = pd.to_datetime(result.date if 'date' in result else result.month + '-01').dt.year
    yearly = result.groupby(['year','target','model','approach'],as_index=False).agg(
        observations=('actual','size'),coverage=('covered','mean'),mean_width_gbp=('width','mean'))
    yearly['scale'] = scale
    yearly['nominal_coverage'] = LEVEL
    return metrics, yearly, result


def calibrate(detail, monthly, metadata):
    periods = [pd.Period(p,'M') for p in metadata['evaluation_months']]
    if len(periods) <= CALIBRATION_MONTHS:
        raise ValueError('Not enough evaluation months after the calibration period')
    calibration_months = set(map(str, periods[:CALIBRATION_MONTHS]))
    assessment_months = set(map(str, periods[CALIBRATION_MONTHS:]))
    outputs={'daily_amount':[], 'monthly_total':[]}; metrics=[]; yearly=[]
    for scale, source, date_key in (('daily_amount',detail,'date'),('monthly_total',monthly,'month')):
        source = source[(source.phase=='evaluation') & source.model.isin(['ridge','random_forest'])].copy()
        for (target,model,approach), group in source.groupby(['target','model','approach'],sort=True):
            cal=group[group.month.astype(str).isin(calibration_months)]
            test=group[group.month.astype(str).isin(assessment_months)]
            if set(cal.month.astype(str).unique()) != calibration_months or set(test.month.astype(str).unique()) != assessment_months:
                raise ValueError(f'Incomplete calibration/evaluation coverage: {scale}/{target}/{model}/{approach}')
            ranged=make_ranges(cal,test,scale)
            m,y,scored=score(ranged,scale)
            outputs[scale].append(scored); metrics.append(m); yearly.append(y)
    return {'daily_ranges':pd.concat(outputs['daily_amount'],ignore_index=True),
            'monthly_ranges':pd.concat(outputs['monthly_total'],ignore_index=True),
            'metrics':pd.concat(metrics,ignore_index=True),
            'yearly_metrics':pd.concat(yearly,ignore_index=True)}


def run(source=OUTPUT, output=INTERVAL_OUTPUT):
    analytics.require_feature('forecast')
    source=Path(source); output=Path(output)
    detail=pd.read_csv(source/'detail.csv',parse_dates=['date'])
    monthly=pd.read_csv(source/'monthly.csv')
    metadata=json.loads((source/'metadata.json').read_text())
    if metadata.get('source_fingerprint') != db.require_import()['fingerprint']:
        raise ValueError('Backtest artifacts belong to another data snapshot; regenerate them first')
    result=calibrate(detail,monthly,metadata)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='history-intervals-',dir=output.parent))
    try:
        for name,frame in result.items():
            frame.to_csv(stage/f'{name}.csv',index=False)
        calmonths=list(map(str,[pd.Period(p,'M') for p in metadata['evaluation_months'][:CALIBRATION_MONTHS]]))
        testmonths=list(map(str,[pd.Period(p,'M') for p in metadata['evaluation_months'][CALIBRATION_MONTHS:]]))
        interval_metadata=dict(source_fingerprint=metadata['source_fingerprint'],nominal_coverage=LEVEL,
            quantiles=[.1,.9],calibration_months=calmonths,evaluation_months=testmonths,
            calibration_method='central empirical signed-residual quantiles; daily and monthly totals calibrated separately',
            monthly_sample_size=len(calmonths),notes=['Synthetic data; retrospective coverage diagnostic.',
            'Model settings were selected before the calibration period.',
            'Evaluation months have previously been reviewed in prior experiments.',
            'Monthly calibration uses only 24 residuals and is imprecise.'])
        (stage/'metadata.json').write_text(json.dumps(interval_metadata,indent=2)+'\n')
        backup=output.with_name(output.name+'-backup-'+uuid.uuid4().hex)
        if output.exists(): output.rename(backup)
        try: stage.rename(output)
        except OSError:
            if backup.exists(): backup.rename(output)
            raise
        if backup.exists(): shutil.rmtree(backup)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return result
