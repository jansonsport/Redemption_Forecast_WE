"""Live stress engine: recomputes every stress-dependent result for a user-defined stress definition.

Python port of handoff/segments_stress/reference_ui/stress-engine.js. It reuses the pipeline's own
functions wherever the stress flags are the only input that changes, so the default definition
reproduces the offline results exactly (tests/test_segments.py checks parity):

- stress days: price at least `dd` below its trailing 252-day high (min 20 days) OR inside any active period
- segment flow correlation: daily/weekly x raw/deseasonalised x all/calm/stress
- segment redemption rate on stress vs calm days (raw and month-matched)
- worst 5/20-day net outflow inside each period, applied to current AUM
- segment-tail scenario combined with the stress-period correlation matrix

Descriptive "what if" analysis of past behaviour, not a forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pipeline as S

KEYS = [(f, b, r) for f in ('daily', 'weekly') for b in ('raw', 'deseasonalised') for r in ('all', 'calm', 'stress')]


@dataclass
class Period:
    name: str
    start: str  # ISO date, inclusive
    end: str    # ISO date, inclusive
    on: bool = True


def default_definition(episodes: dict) -> dict:
    return dict(dd=0.10, periods=[Period(n, a, b, True) for n, (a, b) in episodes.items()])


class StressEngine:
    def __init__(self, bundle: dict):
        o = bundle['out']
        self.val, self.net, self.red = o['seg_aum'], o['seg_net'], o['seg_red']
        self.daily = bundle['daily']
        self.dates = self.daily.index
        self.iso = self.dates.strftime('%Y-%m-%d').to_numpy()
        p = self.daily['price']
        self.drawdown = p / p.rolling(252, min_periods=20).max() - 1
        # Rates and their calendar adjustment do not depend on the stress definition: compute once,
        # exactly as pipeline.flow_correlation does.
        prev = self.val.shift(1)
        rate = (self.net / prev.where(prev > 1e5)).iloc[1:]
        self.rate = rate.clip(rate.quantile(.001), rate.quantile(.999), axis=1)
        self.ds = S.deseasonalise(self.rate.fillna(0)).where(self.rate.notna())
        self.weekly = {b: r.resample('W-FRI').sum(min_count=1) for b, r in (('raw', self.rate), ('deseasonalised', self.ds))}
        self.tails = o['tails']
        self.aum = float(o['base']['aum'])

    # ------------------------------------------------------------------ flags
    def period_mask(self, periods) -> np.ndarray:
        mask = np.zeros(len(self.iso), dtype=bool)
        for p in periods:
            if p.on and p.start and p.end and p.start <= p.end:
                mask |= (self.iso >= p.start) & (self.iso <= p.end)
        return mask

    def price_mask(self, dd: float) -> np.ndarray:
        if not dd:
            return np.zeros(len(self.iso), dtype=bool)
        return (self.drawdown <= -dd).fillna(False).to_numpy()

    def flags(self, dd: float, periods) -> pd.Series:
        return pd.Series(self.price_mask(dd) | self.period_mask(periods), index=self.dates, name='stress')

    # ------------------------------------------------------------ correlation
    def correlations(self, stress: pd.Series):
        stress = stress.reindex(self.rate.index).fillna(False).astype(bool)
        sw = stress.resample('W-FRI').sum() >= 3  # a stress week has 3+ stress days
        res, rows = {}, []
        for freq, basis, regime in KEYS:
            x = self.rate if basis == 'raw' else self.ds
            flag = stress
            if freq == 'weekly':
                x, flag = self.weekly[basis], sw
            if regime != 'all':
                x = x.loc[~flag if regime == 'calm' else flag]
            cm = x.corr(min_periods=20)
            off = cm.to_numpy()[~np.eye(cm.shape[0], dtype=bool)]
            finite = off[np.isfinite(off)]
            res[(freq, basis, regime)] = cm
            rows.append(dict(frequency=freq, basis=basis, regime=regime, periods=int(len(x)),
                             mean_pairwise_r=float(finite.mean()) if len(finite) else None,
                             min_r=float(finite.min()) if len(finite) else None,
                             max_r=float(finite.max()) if len(finite) else None))
        return res, pd.DataFrame(rows)

    # ------------------------------------------------------------------- lift
    def lift(self, stress: pd.Series) -> pd.DataFrame:
        if not stress.any():
            return pd.DataFrame({'segment': list(self.val.columns), 'stress_bp_per_day': np.nan, 'calm_bp_per_day': np.nan,
                                 'value_lift': np.nan, 'month_matched_value_lift': np.nan})
        out = S.value_stress_lift(self.val, self.red, pd.DataFrame({'stress': stress}))
        return out.replace([np.inf, -np.inf], np.nan)

    # ----------------------------------------------------------------- replay
    def replay(self, periods, aum: float | None = None) -> pd.DataFrame:
        aum = self.aum if aum is None else aum
        d = self.daily; netd = d['fund_sub'] - d['fund_red']; prev = d['fund_open']; rows = []
        for p in periods:
            if not (p.on and p.start and p.end and p.start <= p.end):
                continue
            inside = (self.iso >= p.start) & (self.iso <= p.end)
            for h in (5, 20):
                r = (netd.rolling(h).sum() / prev.shift(h - 1))[inside].dropna()
                if len(r):
                    worst = float(r.min())
                    rows.append(dict(family='Stress period', scenario=f'{p.name}: worst {h}-day net outflow', horizon_days=h,
                                     ended=r.idxmin().date().isoformat(), outflow_share=max(0.0, -worst),
                                     outflow_gbp=max(0.0, -worst) * aum, payables_included_gbp=0.0))
        return pd.DataFrame(rows, columns=['family', 'scenario', 'horizon_days', 'ended', 'outflow_share', 'outflow_gbp', 'payables_included_gbp'])

    # ------------------------------------------------------------------ tails
    def tail_hist(self, corr_stress: pd.DataFrame) -> dict:
        out = {}
        for h in (1, 5):
            t = self.tails[self.tails.horizon_days == h].set_index('segment').tail_outflow_gbp
            c = corr_stress.reindex(index=t.index, columns=t.index).fillna(0).to_numpy(copy=True)
            np.fill_diagonal(c, 1)
            out[h] = float(np.sqrt(max(0.0, t.values @ c @ t.values)))
        return out

    # -------------------------------------------------------------- all at once
    def run(self, dd: float, periods) -> dict:
        stress = self.flags(dd, periods)
        corr, summary = self.correlations(stress)
        return dict(stress=stress, days=int(stress.sum()), corr=corr, corr_summary=summary, lift=self.lift(stress),
                    replay=self.replay(periods), tail=self.tail_hist(corr[('daily', 'raw', 'stress')]))
