"""Pure (Streamlit-free) logic behind the Segments & stress page, ported from reference_ui/segments.js.

Kept separate from the page so it can be tested directly. Wording is descriptive: it reports past
behaviour and "if ... then" exposures, never forecasts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RARE = 'Rare traders'
RARE_COLOUR = '#8795a8'
# Validated categorical palette (lightness band, chroma floor, CVD and normal-vision separation, 3:1 contrast
# on the app background). Chosen not to reuse the app's subscription teal, redemption orange or net-flow blue.
SEGMENT_PALETTE = ['#6250d6', '#b8860b', '#d55181', '#2e8b3a', '#1c8fd6', '#9c4f2e', '#a050c8']
COVERED, SHORTFALL = '#16877c', '#d26b45'

SECTIONS = ['Overview', '1 · Data', '2 · Finding segments', '3 · The segments', '4 · Segments in stress',
            '5 · Concentration & stress tests', '6 · Segment clients']

FEATURE_LABELS = {'log_holding': 'Holding size', 'log_trades_per_year': 'Trading frequency', 'redemption_share': 'Redemption share',
                  'log_trade_rel': 'Trade size vs holding', 'log_largest_red_rel': 'Largest redemption vs holding',
                  'log_stress_lift': 'Stress response', 'red_march_share': 'March redemptions',
                  'red_december_share': 'December redemptions', 'month_end_trade_share': 'Month-end trading'}

# Query-tool fields: label, format, plain description.
QUERY_FIELDS = {
    'current_holding_gbp': ('Holding at 30 Oct 2026', 'gbp', 'value held now'),
    'mean_holding_gbp': ('Average holding', 'gbp', 'average balance over its history'),
    'trades': ('Trades (all time)', 'int', 'number of subscriptions and redemptions'),
    'years_active': ('Years active', 'num2', 'years with an open account in 2006–2026'),
    'trades_per_year': ('Trades per year', 'num2', 'how often it trades'),
    'redemption_share': ('Redemption share', 'pct', 'share of traded value that was redemptions'),
    'median_trade_rel_holding': ('Typical trade ÷ holding', 'pct', 'typical trade as a share of the balance'),
    'largest_redemption_rel': ('Largest redemption ÷ holding', 'pct', 'biggest single redemption as a share of the average balance'),
    'stress_lift': ('Stress lift', 'num2', 'redemption rate on stress days ÷ calm days (1 = no change)'),
    'red_march_share': ('March redemptions', 'pct', 'share of redemptions made in March'),
    'red_december_share': ('December redemptions', 'pct', 'share of redemptions made in December'),
    'month_end_trade_share': ('Month-end trades', 'pct', 'share of trades in the last 3 business days of a month'),
}
PRESETS = {
    'Big rare traders': dict(segs=[RARE], conds=[('current_holding_gbp', '>=', 5e6)]),
    'Redeem more in stress': dict(segs=[], conds=[('stress_lift', '>=', 1.5), ('trades', '>=', 20)]),
    'Lumpy redeemers': dict(segs=[], conds=[('largest_redemption_rel', '>=', 0.5), ('current_holding_gbp', '>=', 1e6)]),
    'December-heavy, large': dict(segs=[], conds=[('red_december_share', '>=', 0.4), ('current_holding_gbp', '>=', 1e6)]),
}
AUM_BANDS = ['Under £10k', '£10k–£100k', '£100k–£1m', '£1m–£10m', '£10m and over']

PROFILE_METRICS = {'aum_share': ('Share of fund AUM', 'pct'), 'clients': ('Clients', 'int'), 'median_holding': ('Median average holding', 'gbp'),
                   'trades_per_year': ('Trades per year (median)', 'num2'), 'redemption_share': ('Redemption share (median)', 'pct'),
                   'trade_rel_holding': ('Typical trade ÷ holding (median)', 'pct'), 'largest_red_rel': ('Largest redemption ÷ holding (median)', 'pct'),
                   'stress_lift': ('Stress lift, median client', 'num2'), 'march_red_share': ('Share of redemptions in March', 'pct'),
                   'december_red_share': ('Share of redemptions in December', 'pct'), 'month_end_share': ('Share of trades at month-end', 'pct')}


def finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def pct(v, d=1) -> str:
    return f'{100 * float(v):.{d}f}%' if finite(v) else '—'


def num2(v) -> str:
    return f'{float(v):.2f}' if finite(v) else '—'


def gbp(v) -> str:
    if not finite(v):
        return '—'
    v = float(v); a = abs(v); s = '−' if v < 0 else ''
    if a >= 1e9:
        return f'{s}£{a / 1e9:.2f}bn'
    if a >= 1e6:
        return f'{s}£{a / 1e6:.1f}m'
    if a >= 1e3:
        return f'{s}£{a / 1e3:.0f}k'
    return f'{s}£{a:.0f}'


def fmt(v, kind) -> str:
    if not finite(v):
        return '—'
    return {'pct': pct, 'gbp': gbp, 'int': lambda x: f'{round(float(x)):,}', 'num2': num2}[kind](v)


def colours(order: list[str]) -> dict[str, str]:
    """One fixed colour per segment, in the bundle's fixed order (largest AUM first)."""
    named = [s for s in order if s != RARE]
    out = {s: SEGMENT_PALETTE[i % len(SEGMENT_PALETTE)] for i, s in enumerate(named)}
    out[RARE] = RARE_COLOUR
    return out


# ---------------------------------------------------------------- verdicts and descriptions
def verdict(segment: str, cohesion: pd.DataFrame, random_pair_r: float) -> tuple[str, str]:
    """How alike a segment is: Tight group / Loose group / Label only / Not clustered, with a reason."""
    row = cohesion.set_index('segment').loc[segment] if segment in set(cohesion.segment) else None
    if segment == RARE or row is None or not finite(row.get('silhouette')):
        return 'Not clustered', 'Set aside for too little history; the label says little about behaviour.'
    flow = float(row['flow_r_within']) if finite(row.get('flow_r_within')) else 0.0
    ratio = flow / max(0.01, random_pair_r)
    share = float(row['well_fitted_share'])
    if share >= .25 and ratio >= 2.5:
        return 'Tight group', f'{pct(share, 0)} clearly belong; members have moved money together {ratio:.1f}× as much as random clients.'
    if ratio >= 1.5 or share >= .15:
        return 'Loose group', f'{pct(share, 0)} clearly belong; members have moved money together {ratio:.1f}× as much as random clients.'
    return 'Label only', (f'Only {pct(share, 0)} clearly belong and members have moved money together about as much as random clients '
                          f'({ratio:.1f}×). Review clients individually.')


PLAIN = {'trades_per_year': ('trades per year', 'num2'), 'median_holding': ('average holding', 'gbp'),
         'redemption_share': ('of traded value is redemptions', 'pct'), 'trade_rel_holding': ('typical trade as a share of holding', 'pct'),
         'largest_red_rel': ('largest redemption as a share of holding', 'pct'), 'march_red_share': ('of redemptions in March', 'pct'),
         'december_red_share': ('of redemptions in December', 'pct'), 'month_end_share': ('of trades at month-end', 'pct')}


def describe(segment: str, profiles: pd.DataFrame, typical: dict, eligibility: dict, top10_rare: int) -> dict:
    """Top three differences from a typical client and a plain 'for liquidity' reading of past behaviour."""
    p = profiles.set_index('segment').loc[segment]
    if segment == RARE:
        return dict(lines=[f"Fewer than {eligibility['min_trades']} trades, or less than {eligibility['min_years']:g} year of history, "
                           'so behaviour cannot be measured reliably.'],
                    implication=f'They hold {pct(p.aum_share)} of AUM, including {top10_rare} of the 10 biggest accounts. '
                                'Review the large ones individually.')
    ratio = {k: (p[k] or 1e-9) / (typical[k] or 1e-9) for k in PLAIN}
    diffs = sorted(ratio, key=lambda k: -abs(math.log(max(ratio[k], 1e-12))))
    diffs = [k for k in diffs if abs(math.log(max(ratio[k], 1e-12))) > math.log(1.25)][:3]
    lines = [f'**{fmt(p[k], PLAIN[k][1])}** {PLAIN[k][0]} (typical {fmt(typical[k], PLAIN[k][1])})' for k in diffs]
    r = lambda k: (p[k] or 0) / (typical[k] or 1e-9)
    imp = []
    if r('december_red_share') > 1.3 or r('march_red_share') > 1.3:
        imp.append('outflows have clustered around financial year-ends (' + ('March' if r('march_red_share') > r('december_red_share') else 'December') + ')')
    if r('largest_red_rel') > 1.8:
        imp.append('lumpy: has taken out a large part of the holding at once')
    if r('month_end_share') > 1.8:
        imp.append('cash moves have clustered in the last days of each month')
    if r('median_holding') > 1.5 and r('trades_per_year') >= 0.9:
        imp.append('has driven much of the routine cash movement by value')
    if r('trades_per_year') < 0.5:
        imp.append('rarely trades, so each trade is hard to anticipate')
    if r('redemption_share') < 0.9 and r('trade_rel_holding') < 0.8:
        imp.append('a steady, small-ticket, mostly additive base')
    if r('redemption_share') > 1.15:
        imp.append('has withdrawn more than it added')
    return dict(lines=lines or ['Close to the typical client on every measure.'],
                implication=_sentence('; '.join(imp)) if imp else 'No standout liquidity behaviour.')


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + '.'


def lower_label(label: str) -> str:
    """Lower-case a habit label for running text, keeping month names capitalised."""
    return label[:1].lower() + label[1:] if not label.split()[0] in ('March', 'December') else label


def together_answer(summary: pd.DataFrame, freq: str) -> dict:
    """Plain-language answer to 'do the segments pull money out at the same time?'."""
    g = lambda b, r: summary.set_index(['frequency', 'basis', 'regime']).loc[(freq, b, r), 'mean_pairwise_r']
    rc, rs, dc, ds = g('raw', 'calm'), g('raw', 'stress'), g('deseasonalised', 'calm'), g('deseasonalised', 'stress')
    if not all(finite(x) for x in (rc, rs, dc, ds)):
        return dict(values=(rc, rs, dc, ds), text='Not enough stress and calm periods under this definition to compare (at least 20 shared periods are needed).')
    word = 'hardly' if rs < .1 else 'a little' if rs < .25 else 'moderately' if rs < .5 else 'strongly'
    change = 'about the same as' if abs(rs - rc) < .03 else 'higher than' if rs > rc else 'lower than'
    text = (f'Segments have moved together **{word}** in stress ({rs:.2f}), {change} in calm periods ({rc:.2f}). '
            f'After removing calendar habits (everyone redeeming at year-end, for example), it drops to {ds:.2f} in stress vs {dc:.2f} calm. '
            + ('So most of the togetherness is shared calendar timing, not panic. ' if rs - ds > 0.03 else 'Calendar timing explains little of it. ')
            + ('**Caution:** togetherness has risen in stress, so spreading across segments has helped less when it mattered.'
               if rs - rc > 0.05 else 'Spreading across segments has held up in past stress periods.'))
    return dict(values=(rc, rs, dc, ds), text=text)


def lift_answer(lift: pd.DataFrame, profiles: pd.DataFrame) -> str:
    l = lift.set_index('segment').month_matched_value_lift
    up = [s for s, v in l.items() if finite(v) and v >= 1.15]
    down = [s for s, v in l.items() if finite(v) and v <= 0.87]
    if not l.map(finite).any():
        return 'No stress days under this definition, so stress and calm cannot be compared.'
    parts = (', '.join(f'{s} redeemed **{l[s]:.2f}×**' for s in up) + ' their usual rate for the same month') if up else 'no segment redeemed clearly more than usual'
    if down:
        parts += '; ' + ', '.join(down) + ' redeemed **less**'
    med = profiles.stress_lift
    return (f'In your stress periods, {parts}. The typical (median) client barely changed ({med.min():.2f}–{med.max():.2f}×), '
            'so these rises come from a few large redemptions.')


# ---------------------------------------------------------------- reverse stress
def reverse_stress(top: pd.DataFrame, available: float) -> dict:
    cum = top.holding_gbp.cumsum().to_numpy()
    n = int((cum <= available).sum())
    nxt = top.iloc[n] if n < len(top) else None
    return dict(n=n, cum=cum, next=nxt, label='100+' if n >= len(top) else str(n))


# ---------------------------------------------------------------- query tool
def aum_band(v: pd.Series) -> pd.Series:
    return pd.cut(v, [-np.inf, 1e4, 1e5, 1e6, 1e7, np.inf], right=False, labels=range(5)).astype(int)


def query_clients(features: pd.DataFrame, clients: pd.DataFrame, segs=(), background=None, conds=(), search='',
                  sort='current_holding_gbp', descending=True) -> pd.DataFrame:
    """Clients matching segment chips (OR), background filters (OR within, AND across), every condition, and search."""
    df = features.join(clients[['client_name', 'client_sector', 'client_type', 'country', 'region', 'unit_type', 'account_close_date', 'client_aum_gbp']])
    keep = pd.Series(True, index=df.index)
    if segs:
        keep &= df.segment.isin(list(segs))
    bg = background or {}
    for key, col in [('sectors', 'client_sector'), ('types', 'client_type'), ('countries', 'country'), ('regions', 'region'), ('units', 'unit_type')]:
        if bg.get(key):
            keep &= df[col].isin(bg[key])
    if bg.get('aum'):
        keep &= aum_band(df.client_aum_gbp).isin([AUM_BANDS.index(b) for b in bg['aum']])
    if bg.get('status'):
        status = np.where(df.account_close_date.notna(), 'Closed', 'Open')
        keep &= pd.Series(status, index=df.index).isin(bg['status'])
    for field, op, value in conds:
        v = df[field]
        keep &= v.notna() & ((v >= value) if op == '>=' else (v <= value))
    if search.strip():
        q = search.strip().lower()
        keep &= (df.index.str.lower() + ' ' + df.client_name.fillna('').str.lower()).str.contains(q, regex=False)
    out = df[keep]
    if sort == 'name':
        return out.sort_values('client_name', ascending=not descending, na_position='last')
    return out.sort_values(sort, ascending=not descending, na_position='last')
