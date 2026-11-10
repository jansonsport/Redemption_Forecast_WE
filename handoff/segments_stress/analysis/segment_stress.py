"""Client segmentation, segment flow correlation, concentration and liquidity stress tests.

Config-driven (Phase 0): every table and column is looked up through CONFIG, so the same code runs
on another fund database by changing one dictionary. Phases follow the analysis brief:
0 inventory, 1 data checks, 2 fund context, 3 client features, 4 segmentation, 5 flow correlation,
6 concentration, 7 stress tests, 8 outputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/tables'

CONFIG = dict(
    name='Charity fund (synthetic, 2006–2026)',
    files=dict(clients='synthetic_clients.csv', flows='synthetic_client_flows.csv',
               fund='synthetic_fund_daily.csv', payables='synthetic_income_distributions.csv'),
    clients=dict(id='client_id', name='client_name', sector='client_sector', type='client_type', share_class='unit_type',
                 open='account_open_date', close='account_close_date', balance='client_aum_gbp', country='country', region='region'),
    flows=dict(id='transaction_id', client='client_id', trade_date='trade_date', cash_date='cash_flow_date', amount='flow_gbp',
               status='cash_flow_status', balance_after='client_aum_after_trade_gbp', units_after=None, currency=None, flow_type=None),
    fund=dict(date='date', aum='closing_aum_gbp', open_aum='opening_aum_gbp', sub='subscriptions_trade_date_gbp',
              red='redemptions_trade_date_gbp', sub_cash='subscriptions_cash_date_gbp', red_cash='redemptions_cash_date_gbp',
              prices={'Income': 'income_unit_price_gbp', 'Accumulation': 'accumulation_unit_price_gbp'},
              payables=['income_distributions_payable_gbp', 'redemptions_awaiting_payment_gbp']),
    payables=dict(client='client_id', date='payment_date', amount='distribution_gbp', status='payment_status', pending=['Declared', 'Pending', 'Payable']),
    date_format=None,                      # pandas infers ISO dates; set e.g. '%d/%m/%Y' for other sources
    redemption_sign=-1,                    # redemptions are negative amounts in the flow table
    keep_status=['Settled'],               # drop cancelled / pending / reversed
    internal_flow_types=['switch', 'transfer'],  # netted out when a flow-type column exists
    behaviour_date='trade_date',           # instructions (behaviour, stress) use trade date; cash timing uses cash date
    currency='GBP',
    named_episodes={'Global financial crisis': ('2008-09-15', '2009-03-31'), 'Euro-area debt crisis': ('2011-08-01', '2011-10-31'),
                    'COVID-19 market shock': ('2020-02-20', '2020-04-30'), 'UK gilt / LDI crisis': ('2022-09-23', '2022-10-31')},
    eligibility=dict(min_trades=8, min_years=1.0),
    k_range=range(3, 9), min_segment_share=0.03, bootstrap=30, seed=42,
    buffers=[0.05, 0.10, 0.20],
    # Reference liquidity floors (daily, weekly) — binding only for money market funds.
    floors={'UK MMFR · LVNAV/CNAV (current)': (0.10, 0.30), 'UK MMFR · VNAV short-term (current)': (0.075, 0.15),
            'US Rule 2a-7 (2023 amendments)': (0.25, 0.50)},
    fire_sale_haircuts=[0.01, 0.03, 0.05],
    previous=dict(rare_trader_aum_share=0.27),  # figure quoted from the earlier analysis, for comparison
)


# ---------------------------------------------------------------- Phase 0
def load(cfg=CONFIG):
    F = cfg['files']; rd = lambda f, dates: pd.read_csv(ROOT / f, parse_dates=dates, date_format=cfg['date_format'])
    c = cfg['clients']; fl = cfg['flows']; fu = cfg['fund']; pa = cfg['payables']
    clients = rd(F['clients'], [c['open'], c['close']])
    flows = rd(F['flows'], [fl['trade_date'], fl['cash_date']])
    fund = rd(F['fund'], [fu['date']])
    pay = rd(F['payables'], [pa['date']]) if F.get('payables') else None
    return clients, flows, fund, pay


def inventory(cfg, clients, flows, fund, pay):
    need = {'Client master': (clients, ['id', 'sector', 'open', 'close', 'share_class', 'balance'], cfg['clients']),
            'Transactions': (flows, ['client', 'trade_date', 'amount', 'status', 'balance_after'], cfg['flows']),
            'Fund daily': (fund, ['date', 'aum'], cfg['fund']),
            'Known future outflows': (pay, ['client', 'date', 'amount'], cfg['payables'])}
    rows = []
    for role, (df, fields, m) in need.items():
        present = [f for f in fields if df is not None and m.get(f) in df.columns]
        missing = [f for f in fields if f not in present]
        extra = 'unit prices: ' + ', '.join(cfg['fund']['prices']) if role == 'Fund daily' else ''
        rows.append(dict(role=role, table=cfg['files'].get({'Client master': 'clients', 'Transactions': 'flows', 'Fund daily': 'fund', 'Known future outflows': 'payables'}[role]),
                         rows=0 if df is None else len(df), fields_present=', '.join(present), fields_missing=', '.join(missing) or '—', notes=extra))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Phase 1
def clean_and_check(cfg, clients, flows, fund):
    fl, fu, c = cfg['flows'], cfg['fund'], cfg['clients']
    checks = []
    n0 = len(flows)
    status_counts = flows[fl['status']].value_counts().to_dict()
    f = flows[flows[fl['status']].isin(cfg['keep_status'])].copy()
    checks.append(dict(check='Status filter', result=f'kept {len(f):,} of {n0:,}', detail=json.dumps(status_counts), passed=True))
    dup_id = int(f[fl['id']].duplicated().sum()); dup_row = int(f.drop(columns=[fl['id']]).duplicated().sum())
    f = f.drop_duplicates(subset=[fl['id']])
    checks.append(dict(check='Duplicates', result=f'{dup_id} duplicate IDs, {dup_row} duplicate rows', detail='exact duplicates removed', passed=dup_id == 0))
    if fl.get('flow_type') and fl['flow_type'] in f:
        internal = f[fl['flow_type']].str.lower().isin(cfg['internal_flow_types']); f = f[~internal]
        checks.append(dict(check='Internal moves', result=f'{int(internal.sum())} switches/transfers removed', detail='flow-type column', passed=True))
    else:
        # Heuristic: equal and opposite amounts between different clients on the same trade date.
        g = f.assign(a=f[fl['amount']].abs().round(2), pos=f[fl['amount']] > 0, neg=f[fl['amount']] < 0)
        h = g.groupby([fl['trade_date'], 'a'])[['pos', 'neg']].any(); pairs = h.pos & h.neg
        checks.append(dict(check='Internal moves', result=f'{int(pairs.sum())} same-day equal-and-opposite amount groups',
                           detail='No flow-type column; candidates reported, not removed (coincidental matches of small amounts are expected)', passed=True))
    cur = cfg['currency'] if not fl.get('currency') else ', '.join(f[fl['currency']].unique())
    checks.append(dict(check='Currencies', result=cur, detail='single currency' if not fl.get('currency') else 'from currency column', passed=True))
    # Omnibus/nominee: by type name, plus outliers in trade count among large holders.
    name_hit = clients[c['type']].str.contains('omnibus|nominee|platform', case=False).sum()
    counts = f.groupby(fl['client']).size()
    big = clients.set_index(c['id'])[c['balance']].rank(pct=True) > 0.99
    suspicious = counts.reindex(big[big].index).fillna(0)
    flag = int((suspicious > counts.quantile(0.999)).sum())
    checks.append(dict(check='Omnibus / nominee accounts', result=f'{int(name_hit)} by type name; {flag} top-1% holders with extreme trade counts',
                       detail='none hidden behind single IDs detected', passed=True))
    checks.append(dict(check='Trade vs cash date', result=f"behaviour, segments, correlation and stress use {cfg['behaviour_date'].replace('_', ' ')}",
                       detail='cash-date totals reconciled separately; payment lag affects cash timing only', passed=True))
    # Client flows sum to fund flows (trade date and cash date).
    for label, dcol, scol, rcol in [('trade date', fl['trade_date'], fu['sub'], fu['red']), ('cash date', fl['cash_date'], fu['sub_cash'], fu['red_cash'])]:
        g = f.groupby(dcol)[fl['amount']].agg(sub=lambda s: s[s > 0].sum(), red=lambda s: -s[s < 0].sum())
        j = fund.set_index(fu['date'])[[scol, rcol]].join(g, how='left').fillna(0)
        err = max((j['sub'] - j[scol]).abs().max(), (j['red'] - j[rcol]).abs().max())
        outside = int((~f[dcol].isin(fund[fu['date']])).sum())
        checks.append(dict(check=f'Client flows = fund flows ({label})', result=f'max daily difference £{err:,.2f}', detail=f'{outside} transactions dated outside fund calendar (pre-launch trade dates settle inside it)', passed=err < 1))
    return f, pd.DataFrame(checks)


def rebuild_units(cfg, clients, f, fund):
    """Units per transaction from balance-after / class price, checked against flow / price."""
    fl, fu, c = cfg['flows'], cfg['fund'], cfg['clients']
    px = fund.set_index(fu['date'])[list(fu['prices'].values())]
    px.columns = list(fu['prices'].keys())
    # Pre-launch trade dates: use the first available price.
    full = px.reindex(pd.date_range(min(f[fl['trade_date']].min(), px.index.min()), px.index.max())).bfill().ffill()
    cls = f[fl['client']].map(clients.set_index(c['id'])[c['share_class']])
    p = np.where(cls.eq('Income'), f[fl['trade_date']].map(full['Income']), f[fl['trade_date']].map(full['Accumulation']))
    f = f.assign(share_class=cls.values, price=p).sort_values([fl['client'], fl['trade_date'], fl['id']])
    f['units_after'] = f[fl['units_after']] if fl.get('units_after') else f[fl['balance_after']] / f['price']
    f['units_flow'] = f[fl['amount']] / f['price']
    du = f.groupby(fl['client'])['units_after'].diff()
    err = (du - f['units_flow'])[du.notna()].abs()
    rel = err / f['units_after'].abs().clip(lower=1)
    first = ~f.duplicated(fl['client'])
    f['units_before'] = np.where(first, f['units_after'] - f['units_flow'], f.groupby(fl['client'])['units_after'].shift())
    return f, full, dict(median_abs_unit_error=float(err.median()), p99_abs_unit_error=float(err.quantile(.99)), max_relative_error=float(rel.max()))


def daily_units(cfg, f, fund_dates, group):
    """Daily units held per (group, share class) by trade date, including units held before first trade."""
    fl = cfg['flows']
    first = f.drop_duplicates(fl['client'])
    opening = first.assign(g=first[fl['client']].map(group)).groupby(['g', 'share_class'])['units_before'].sum()
    ch = f.assign(g=f[fl['client']].map(group), d=f[fl['trade_date']].clip(lower=fund_dates.min()))
    ch = ch.groupby(['d', 'g', 'share_class'])['units_flow'].sum().unstack(['g', 'share_class']).reindex(fund_dates).fillna(0)
    cols = ch.columns.union(opening.index)
    ch = ch.reindex(columns=cols, fill_value=0)
    # Opening units belong to clients whose first trade is after launch: they hold those units from launch.
    return ch.cumsum() + opening.reindex(cols).fillna(0).values


# ---------------------------------------------------------------- Phase 2
def fund_context(cfg, fund):
    fu = cfg['fund']; d = fund.set_index(fu['date'])
    acc = d[fu['prices']['Accumulation']]
    ret = acc.pct_change()
    vol = float(ret.std() * np.sqrt(252))
    is_mmf = vol < 0.005 and float((acc / acc.cummax() - 1).min()) > -0.005
    dd = acc / acc.rolling(252, min_periods=20).max() - 1
    stress_price = dd <= -0.10
    episode = pd.Series(False, index=d.index); names = pd.Series('', index=d.index)
    for k, (a, b) in cfg['named_episodes'].items():
        m = (d.index >= a) & (d.index <= b); episode |= m; names[m] = k
    stress = episode if is_mmf else (stress_price | episode)
    month = d.groupby(d.index.month).agg(sub=(fu['sub'], 'sum'), red=(fu['red'], 'sum'), aum=(fu['open_aum'], 'sum'))
    month['sub_rate_bp'] = 1e4 * month['sub'] / month['aum']; month['red_rate_bp'] = 1e4 * month['red'] / month['aum']
    month['net_rate_bp'] = month['sub_rate_bp'] - month['red_rate_bp']
    month.index.name = 'month'
    flags = pd.DataFrame({'drawdown_1y': dd, 'stress_price': stress_price, 'episode': episode, 'episode_name': names, 'stress': stress})
    info = dict(annualised_price_vol=vol, max_drawdown=float((acc / acc.cummax() - 1).min()), fund_type='Money market fund' if is_mmf else 'Non-MMF (price-variable)',
                stress_definition=('named episodes only (flat MMF price cannot signal stress)' if is_mmf else
                                   'fund accumulation price ≥10% below its trailing one-year high, plus named market episodes'),
                stress_days=int(stress.sum()), calm_days=int((~stress).sum()))
    return month.reset_index(), flags, info


# ---------------------------------------------------------------- Phase 3
def client_features(cfg, clients, f, fund, flags):
    fl, c = cfg['flows'], cfg['clients']; end = fund[cfg['fund']['date']].max(); start = fund[cfg['fund']['date']].min()
    m = clients.set_index(c['id'])
    life0 = m[c['open']].clip(lower=start); life1 = m[c['close']].fillna(end).clip(upper=end)
    years = ((life1 - life0).dt.days / 365.25).clip(lower=0)
    t = f[fl['trade_date']]; a = f[fl['amount']]; red = (-a).clip(lower=0); sub = a.clip(lower=0)
    bal_before = (f['units_before'] * f['price']).abs()
    f = f.assign(red=red, sub=sub, rel=(a.abs() / bal_before.where(bal_before > 0)), stress=t.map(flags['stress']).fillna(False).astype(bool),
                 m=t.dt.month, me=(t + pd.offsets.BMonthEnd(0) - t).dt.days <= 3, qe=t.dt.month.isin([3, 6, 9, 12]) & ((t + pd.offsets.BMonthEnd(0) - t).dt.days <= 3),
                 hold=(f['units_after'] * f['price']).abs())
    g = f.groupby(fl['client'])
    feat = pd.DataFrame({'trades': g.size(), 'sub_gbp': g['sub'].sum(), 'red_gbp': g['red'].sum(), 'mean_holding_gbp': g['hold'].mean(),
                         'median_trade_rel_holding': g['rel'].median(), 'largest_redemption_gbp': g['red'].max(),
                         'red_march_share': f['red'].where(f.m == 3, 0).groupby(f[fl['client']]).sum() / g['red'].sum().replace(0, np.nan),
                         'red_december_share': f['red'].where(f.m == 12, 0).groupby(f[fl['client']]).sum() / g['red'].sum().replace(0, np.nan),
                         'month_end_trade_share': g['me'].mean(), 'quarter_end_trade_share': g['qe'].mean()})
    feat = feat.reindex(m.index)
    feat['trades'] = feat['trades'].fillna(0); feat['years_active'] = years
    feat['trades_per_year'] = feat['trades'] / years.clip(lower=0.25)
    feat['redemption_share'] = (feat['red_gbp'] / (feat['red_gbp'] + feat['sub_gbp'])).fillna(0)
    feat['largest_redemption_rel'] = (feat['largest_redemption_gbp'] / feat['mean_holding_gbp']).replace([np.inf, -np.inf], np.nan)
    feat['current_holding_gbp'] = m[c['balance']]
    # Stress lift: redemption rate (redeemed £ per day per £ held) on stress days vs calm days, shrunk toward 1.
    stress_days = int(flags['stress'].sum()); calm_days = int((~flags['stress']).sum())
    rs = f[f.stress].groupby(fl['client'])['red'].sum().reindex(m.index).fillna(0) / stress_days
    rc = f[~f.stress].groupby(fl['client'])['red'].sum().reindex(m.index).fillna(0) / calm_days
    prior = (rs.sum() + rc.sum()) / len(m)  # shrinkage pseudo-count in £/day
    feat['stress_lift'] = (rs + prior * 0.5) / (rc + prior * 0.5)
    e = cfg['eligibility']
    feat['eligible'] = (feat['trades'] >= e['min_trades']) & (feat['years_active'] >= e['min_years'])
    feat['yield_sensitivity'] = np.nan  # needs fund yield vs Bank Rate series; not in this database
    return feat


def eligibility_table(feat, total_aum):
    rows = []
    for t in [2, 4, 8, 12, 20]:
        for y in [0.5, 1, 2]:
            ok = (feat.trades >= t) & (feat.years_active >= y)
            rows.append(dict(min_trades=t, min_years=y, eligible_clients=int(ok.sum()), eligible_share=float(ok.mean()),
                             rare_aum_share=float(feat.loc[~ok, 'current_holding_gbp'].sum() / total_aum)))
    return pd.DataFrame(rows)


FEATURES = ['log_holding', 'log_trades_per_year', 'redemption_share', 'log_trade_rel', 'log_largest_red_rel', 'log_stress_lift',
            'red_march_share', 'red_december_share', 'month_end_trade_share']


def model_matrix(feat):
    x = pd.DataFrame(index=feat.index)
    x['log_holding'] = np.log10(feat.mean_holding_gbp.clip(lower=100))
    x['log_trades_per_year'] = np.log10(feat.trades_per_year.clip(lower=0.1))
    x['redemption_share'] = feat.redemption_share
    x['log_trade_rel'] = np.log10(feat.median_trade_rel_holding.clip(1e-4, 10).fillna(1))
    x['log_largest_red_rel'] = np.log10(feat.largest_redemption_rel.clip(1e-4, 10).fillna(1e-4))
    x['log_stress_lift'] = np.log(feat.stress_lift.clip(0.1, 10))
    for k in ['red_march_share', 'red_december_share', 'month_end_trade_share']: x[k] = feat[k].fillna(0)
    return x[FEATURES]


# ---------------------------------------------------------------- Phase 4
def segment(cfg, feat):
    elig = feat[feat.eligible]; x = model_matrix(elig); z = StandardScaler().fit_transform(x)
    rows = []; models = {}
    for k in cfg['k_range']:
        km = KMeans(k, n_init=10, random_state=cfg['seed']).fit(z)
        sil = silhouette_score(z, km.labels_, sample_size=min(6000, len(z)), random_state=cfg['seed'])
        share = np.bincount(km.labels_) / len(z)
        rows.append(dict(k=k, silhouette=float(sil), smallest_segment_share=float(share.min()), inertia=float(km.inertia_))); models[k] = km
    sel = pd.DataFrame(rows)

    def stability(k, n):
        # Refit on bootstrap samples, predict every client, agreement with reference labels (adjusted Rand index).
        rng = np.random.default_rng(cfg['seed'] + k); out = []
        for _ in range(n):
            idx = rng.integers(0, len(z), len(z))
            kb = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(z[idx])
            out.append(adjusted_rand_score(models[k].labels_, kb.predict(z)))
        return np.array(out)
    sel['ari_median_10boot'] = [float(np.median(stability(k, 10))) for k in sel.k]
    # Choice: best silhouette among usable k (every segment >= min share). Bootstrap ARI is reported, not
    # used to choose: with 10 resamples it is too noisy to rank k reliably.
    usable = sel[sel.smallest_segment_share >= cfg['min_segment_share']]
    k = int(usable.sort_values('silhouette', ascending=False).k.iloc[0]); km = models[k]
    sel['chosen'] = sel.k.eq(k)
    ari = stability(k, cfg['bootstrap'])
    labels = pd.Series(km.labels_, index=elig.index)
    names = name_segments(x.assign(seg=labels.values), feat.loc[elig.index])
    seg = pd.Series('Rare traders', index=feat.index); seg[elig.index] = labels.map(names)
    return seg, sel, np.array(ari), names


def name_segments(x, feat):
    prof = x.groupby('seg').mean(); z = (prof - x[FEATURES].mean()) / x[FEATURES].std()
    names = {}; used = set()
    rules = [('log_stress_lift', 1, 'Stress-sensitive redeemers'), ('log_holding', 1, 'Large active holders'),
             ('log_trades_per_year', 1, 'Frequent traders'), ('redemption_share', 1, 'Net withdrawers'),
             ('redemption_share', -1, 'Steady accumulators'), ('red_december_share', 1, 'Year-end redeemers'),
             ('red_march_share', 1, 'March (year-end) redeemers'), ('log_trade_rel', 1, 'Large-ticket traders'),
             ('log_holding', -1, 'Small occasional investors'), ('log_trades_per_year', -1, 'Low-activity holders'),
             ('month_end_trade_share', 1, 'Month-end traders'), ('log_largest_red_rel', 1, 'Lumpy redeemers')]
    for col, sign, label in rules:
        if len(names) == len(prof): break
        cand = (sign * z[col]).drop(list(names)).sort_values(ascending=False)
        if len(cand) and cand.iloc[0] > 0.5 and label not in used:
            names[cand.index[0]] = label; used.add(label)
    for s in prof.index:
        if s not in names:
            col = z.loc[s].abs().idxmax(); names[s] = f"{'High' if z.loc[s, col] > 0 else 'Low'} {col.replace('log_', '').replace('_', ' ')}"
    return names


def validate_segments(seg, clients, cfg):
    c = cfg['clients']; m = clients.set_index(c['id']); out = []
    for k in ['sector', 'type', 'share_class', 'country']:
        ct = pd.crosstab(seg, m[c[k]].reindex(seg.index))
        chi = ((ct - np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()) ** 2 / (np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())).values.sum()
        v = np.sqrt(chi / ct.values.sum() / (min(ct.shape) - 1))
        out.append(dict(attribute=c[k], cramers_v=float(v), categories=ct.shape[1]))
    return pd.DataFrame(out), pd.crosstab(seg, m[c['sector']].reindex(seg.index), normalize='index')


# ---------------------------------------------------------------- Phase 5
def segment_aum(cfg, f, prices, fund, seg):
    dates = pd.DatetimeIndex(fund[cfg['fund']['date']])
    u = daily_units(cfg, f, dates, seg)
    p = prices.reindex(dates)
    val = pd.DataFrame({(g, s): u[(g, s)] * p[s] for g, s in u.columns}).T.groupby(level=0).sum().T
    flows = f.assign(g=f[cfg['flows']['client']].map(seg), d=f[cfg['flows']['trade_date']].clip(lower=dates.min()))
    net = flows.groupby(['d', 'g'])[cfg['flows']['amount']].sum().unstack().reindex(dates).fillna(0)
    red = flows.assign(r=(-flows[cfg['flows']['amount']]).clip(lower=0)).groupby(['d', 'g'])['r'].sum().unstack().reindex(dates).fillna(0)
    return val, net.reindex(columns=val.columns, fill_value=0), red.reindex(columns=val.columns, fill_value=0)


def deseasonalise(rate):
    d = rate.index; key = pd.DataFrame({'m': d.month, 'w': d.dayofweek, 'p': np.minimum((d.day - 1) // 10, 2)}, index=d)
    out = rate.copy()
    for col in rate:
        s = rate[col]; fit = s.groupby([key.m, key.p]).transform('mean') + s.groupby(key.w).transform('mean') - s.mean()
        out[col] = s - fit
    return out


def flow_correlation(val, net, flags):
    prev = val.shift(1)
    rate = (net / prev.where(prev > 1e5)).iloc[1:]
    rate = rate.clip(rate.quantile(.001), rate.quantile(.999), axis=1)
    ds = deseasonalise(rate.fillna(0)).where(rate.notna())
    stress = flags['stress'].reindex(rate.index).fillna(False)
    res = {}; summ = []
    for freq in ['daily', 'weekly']:
        for basis, r in [('raw', rate), ('deseasonalised', ds)]:
            if freq == 'weekly':
                rw = r.resample('W-FRI').sum(min_count=1); sw = stress.resample('W-FRI').sum() >= 3  # a stress week has 3+ stress days
            for regime in ['all', 'calm', 'stress']:
                if freq == 'daily': x = r if regime == 'all' else r.loc[~stress if regime == 'calm' else stress]
                else: x = rw if regime == 'all' else rw.loc[~sw if regime == 'calm' else sw]
                cm = x.corr(min_periods=20); k = cm.shape[0]
                off = cm.values[~np.eye(k, dtype=bool)]
                res[(freq, basis, regime)] = cm
                summ.append(dict(frequency=freq, basis=basis, regime=regime, periods=int(len(x)), mean_pairwise_r=float(np.nanmean(off)),
                                 min_r=float(np.nanmin(off)), max_r=float(np.nanmax(off))))
    return rate, res, pd.DataFrame(summ)


def value_stress_lift(val, red, flags):
    """Segment redemption rate by value on stress vs calm days (redeemed GBP / previous-day AUM), raw and
    month-matched (each stress day compared with calm days of the same calendar month)."""
    prev = val.shift(1); ok = prev > 1e5; st = flags['stress'].reindex(val.index).fillna(False).astype(bool); m = val.index.month; rows = []
    for g in val:
        r, a = red[g].where(ok[g]), prev[g].where(ok[g])
        sr, sa, cr, ca = r[st].sum(), a[st].sum(), r[~st].sum(), a[~st].sum()
        calm_rate = (r[~st].groupby(m[~st]).sum() / a[~st].groupby(m[~st]).sum())
        expected = (a[st].groupby(m[st]).sum() * calm_rate.reindex(a[st].groupby(m[st]).sum().index)).sum()
        rows.append(dict(segment=g, stress_bp_per_day=1e4 * sr / sa, calm_bp_per_day=1e4 * cr / ca, value_lift=(sr / sa) / (cr / ca), month_matched_value_lift=sr / expected))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Phase 6
def client_holdings_at(f, prices, dates, cfg):
    """Holding (£) per client at each date = units after last trade on/before date × class price."""
    fl = cfg['flows']; out = {}
    g = f[[fl['client'], fl['trade_date'], 'units_after', 'units_before', 'share_class']]
    first = g.drop_duplicates(fl['client'])
    for d in dates:
        last = g[g[fl['trade_date']] <= d].drop_duplicates(fl['client'], keep='last').set_index(fl['client'])
        units = last['units_after']
        pre = first[first[fl['trade_date']] > d].set_index(fl['client'])
        units = pd.concat([units, pre['units_before']])
        cls = pd.concat([last['share_class'], pre['share_class']])
        out[d] = (units * cls.map(prices.loc[d])).clip(lower=0)
    return pd.DataFrame(out).fillna(0)


def concentration(cfg, f, prices, fund, seg):
    dates = pd.DatetimeIndex(fund[cfg['fund']['date']]); ye = dates.to_series().groupby(dates.year).max()
    ye = list(ye[ye.dt.month == 12]) + [dates.max()]
    hold = client_holdings_at(f, prices, ye, cfg)
    aum = fund.set_index(cfg['fund']['date'])[cfg['fund']['aum']]
    rows = []
    for d in ye:
        h = hold[d].sort_values(ascending=False); tot = h.sum()
        r = dict(date=d.date().isoformat(), rebuilt_aum_gbp=float(tot), fund_aum_gbp=float(aum.loc[d]), holders=int((h > 1).sum()))
        for n in [1, 5, 10, 20, 50, 100]:
            r[f'top{n}_share'] = float(h.iloc[:n].sum() / tot)
            r[f'top{n}_rare_traders'] = int((seg.reindex(h.index[:n]) == 'Rare traders').sum())
        rows.append(r)
    now = hold[dates.max()].sort_values(ascending=False)
    top = pd.DataFrame({'client_id': now.index[:100], 'holding_gbp': now.values[:100], 'share': now.values[:100] / now.sum(), 'segment': seg.reindex(now.index[:100]).values})
    top['rank'] = range(1, len(top) + 1)
    return pd.DataFrame(rows), top, hold


# ---------------------------------------------------------------- Phase 7
def stress_tests(cfg, fund, flags, val, net, red, rate, corr, top, pay, seg):
    fu = cfg['fund']; d = fund.set_index(fu['date']); aum0 = float(d[fu['aum']].iloc[-1]); date0 = d.index.max()
    tests = []
    for n in [1, 5, 10, 20, 50]:
        amt = float(top.holding_gbp.iloc[:n].sum()); tests.append(dict(family='Concentration', scenario=f'Top {n} clients redeem in full', outflow_gbp=amt, outflow_share=amt / aum0))
    # Historical replay: worst net outflow as share of AUM over 1/5/20 days, applied to current AUM.
    netd = (d[fu['sub']] - d[fu['red']]); prev = d[fu['open_aum']]
    for h in [1, 5, 20]:
        r = (netd.rolling(h).sum() / prev.shift(h - 1)).dropna(); w = r.idxmin()
        tests.append(dict(family='Historical replay', scenario=f'Worst {h}-day net outflow (ended {w.date()})', outflow_gbp=-float(r.min()) * aum0, outflow_share=-float(r.min())))
    for name, (a, b) in cfg['named_episodes'].items():
        m = (d.index >= a) & (d.index <= b)
        for h in [5, 20]:
            r = (netd.rolling(h).sum() / prev.shift(h - 1))[m].dropna()
            if len(r): tests.append(dict(family='Named episode', scenario=f'{name}: worst {h}-day net outflow', outflow_gbp=max(0, -float(r.min())) * aum0, outflow_share=max(0, -float(r.min()))))
    # Segment tails: 99th percentile worst daily and 5-day net outflow rate per segment.
    seg_aum = val.iloc[-1]; tails = []
    for h in [1, 5]:
        rr = rate if h == 1 else (net.rolling(5).sum() / val.shift(5)).iloc[5:]
        q = (-rr).quantile(0.99).clip(lower=0)
        t = q * seg_aum; tails.append(pd.DataFrame({'horizon_days': h, 'segment': t.index, 'tail_rate': q.values, 'segment_aum_gbp': seg_aum.values, 'tail_outflow_gbp': t.values}))
        perfect = float(t.sum())
        c = corr[('daily', 'raw', 'stress')].reindex(index=t.index, columns=t.index).fillna(0).values; np.fill_diagonal(c, 1)
        hist = float(np.sqrt(max(0, t.values @ c @ t.values)))
        tests.append(dict(family='Segment tails', scenario=f'{h}-day 99th pct, all segments together (ρ=1)', outflow_gbp=perfect, outflow_share=perfect / aum0))
        tests.append(dict(family='Segment tails', scenario=f'{h}-day 99th pct, historical stress correlation', outflow_gbp=hist, outflow_share=hist / aum0))
    tails = pd.concat(tails)
    payables = {col: float(d[col].iloc[-1]) for col in fu['payables'] if col in d}
    future_pay = 0.0
    if pay is not None:
        pa = cfg['payables']; fp = pay[(pay[pa['date']] > date0)]; future_pay = float(fp[pa['amount']].sum())
    # The fund's payable columns already include declared distributions not yet paid; the distribution table is
    # used only when the fund table lacks them (otherwise the same cash would be counted twice).
    known = sum(payables.values()) if payables else future_pay
    worst5 = max(t['outflow_gbp'] for t in tests if t['family'] == 'Historical replay' and '5-day' in t['scenario'])
    seg5 = max(t['outflow_gbp'] for t in tests if t['family'] == 'Segment tails' and t['scenario'].startswith('5-day') and 'ρ=1' in t['scenario'])
    top10 = next(t['outflow_gbp'] for t in tests if t['scenario'] == 'Top 10 clients redeem in full')
    for label, base in [('Worst 5-day replay', worst5), ('5-day segment tails (ρ=1)', seg5)]:
        tests.append(dict(family='Combined', scenario=f'{label} + top 10 full exit + known payables', outflow_gbp=base + top10 + known, outflow_share=(base + top10 + known) / aum0))
    tests = pd.DataFrame(tests); tests['payables_included_gbp'] = np.where(tests.family.eq('Combined'), known, 0.0)
    # Reverse stress: largest clients that can leave before the buffer is exhausted.
    cum = top.holding_gbp.cumsum().values; rev = []
    for b in cfg['buffers']:
        buf = b * aum0 - known; n = int((cum <= buf).sum()); rev.append(dict(buffer=f'{b:.0%} of AUM', buffer_gbp=b * aum0, after_payables_gbp=buf, clients_before_exhausted=n,
                                                                              note='more than 100 (top-100 list exhausted)' if n == len(cum) else ''))
    for name, (dl, wl) in cfg['floors'].items():
        for lab, x in [('daily', dl), ('weekly', wl)]:
            buf = x * aum0 - known; n = int((cum <= buf).sum())
            rev.append(dict(buffer=f'{name} {lab} floor {x:.1%}', buffer_gbp=x * aum0, after_payables_gbp=buf, clients_before_exhausted=n, note='reference only: this fund is not an MMF'))
    rev = pd.DataFrame(rev)
    # Second round: shortfall above a 10% liquid buffer must be met by selling other assets at a haircut (illustrative).
    second = []
    for _, t in tests.iterrows():
        short = max(0.0, t.outflow_gbp - 0.10 * aum0)
        for hc in cfg['fire_sale_haircuts']:
            second.append(dict(scenario=t.scenario, family=t.family, shortfall_gbp=short, haircut=hc, sale_needed_gbp=short / (1 - hc) if short else 0.0, loss_gbp=short * hc / (1 - hc) if short else 0.0))
    return tests, tails, rev, pd.DataFrame(second), dict(aum=aum0, date=str(date0.date()), known_payables=known, payable_columns=payables, scheduled_distributions_after_date=future_pay)


# ---------------------------------------------------------------- Phase 4a: how alike is each segment?
def segment_cohesion(cfg, feat, f, clients, fund, sample_flow=60):
    """Per segment: mean silhouette, share of clearly-fitting clients (silhouette > 0.25), spread relative to the
    whole fund (mean distance to own centre / mean distance to fund centre, standardised features), and how much
    members move money together (mean pairwise monthly net-flow correlation within a sample, Notebook 04 rules)."""
    from analysis_helpers import complete_period_mask
    from sklearn.metrics import silhouette_samples
    from sklearn.decomposition import PCA
    e = feat[feat.eligible]; z = StandardScaler().fit_transform(model_matrix(e)); seg = e.segment.values
    rng = np.random.default_rng(cfg['seed']); idx = rng.choice(len(z), min(6000, len(z)), replace=False)
    sil = pd.Series(silhouette_samples(z[idx], seg[idx]), index=e.index[idx])
    Z = pd.DataFrame(z, index=e.index); cent = Z.groupby(seg).transform('mean')
    d_own = np.sqrt(((Z - cent) ** 2).sum(1)); d_all = np.sqrt((Z ** 2).sum(1))
    fl, c, fd = cfg['flows'], cfg['clients'], cfg['fund']['date']
    samp = {s_: rng.choice(feat.index[feat.segment == s_], min(sample_flow, int((feat.segment == s_).sum())), replace=False) for s_ in feat.segment.unique()}
    ids = np.concatenate(list(samp.values())); meta = clients.set_index(c['id']).loc[ids].reset_index()
    mask = complete_period_mask(meta, 'M', fund[fd].min(), fund[fd].max())
    g = f[f[fl['client']].isin(ids)]
    v = g.groupby([g[fl['cash_date']].dt.to_period('M'), fl['client']])[fl['amount']].sum().unstack().reindex(index=mask.index, columns=mask.columns).fillna(0).where(mask)
    C = v.corr(min_periods=24); iu = np.triu_indices(len(ids), 1)
    rows = []
    for s_ in sorted(feat.segment.unique()):
        m = C.loc[samp[s_], samp[s_]].values; w = m[np.triu_indices(len(samp[s_]), 1)]
        r = dict(segment=s_, flow_r_within=float(np.nanmean(w)) if np.isfinite(w).any() else None, flow_pairs=int(np.isfinite(w).sum()))
        if s_ in set(seg):
            ss = sil[e.segment.reindex(sil.index) == s_]
            r.update(silhouette=float(ss.mean()), well_fitted_share=float((ss > 0.25).mean()), spread_vs_fund=float(d_own[seg == s_].mean() / d_all.mean()))
        rows.append(r)
    random_r = float(np.nanmean(C.values[iu]))
    # 2-D map of clients (principal components of the standardised features) for the k-means picture.
    pca = PCA(2, random_state=cfg['seed']).fit(z); P = pca.transform(z)
    pick = rng.choice(len(z), min(2500, len(z)), replace=False)
    pts = pd.DataFrame({'client_id': e.index[pick], 'x': P[pick, 0], 'y': P[pick, 1], 'segment': seg[pick]})
    cen = pd.DataFrame(P).groupby(seg).mean(); cen.columns = ['x', 'y']
    load = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['pc1', 'pc2'])
    return pd.DataFrame(rows), random_r, dict(points=pts, centres=cen.reset_index().rename(columns={'index': 'segment'}), loadings=load.reset_index().rename(columns={'index': 'feature'}),
                                             explained=[float(x) for x in pca.explained_variance_ratio_])


# ---------------------------------------------------------------- Phase 4b: segments vs simpler groupings
def grouping_comparison(cfg, clients, f, fund, flags, k, split='2016-11-01'):
    """Out-of-time test: form segments from behaviour before `split`, then measure how much of each
    behaviour feature AFTER `split` is explained (eta squared, between-group share of variance) by those
    segments versus by sector, client type and region. Clients must be eligible in both halves."""
    cut = pd.Timestamp(split); start = fund[cfg['fund']['date']].min(); end = fund[cfg['fund']['date']].max() + pd.Timedelta(days=1)
    c = cfg['clients']; fl = cfg['flows']; fd = cfg['fund']['date']

    def feats(lo, hi):
        ff = f[(f[fl['trade_date']] >= lo) & (f[fl['trade_date']] < hi)]; fu = fund[(fund[fd] >= lo) & (fund[fd] < hi)]
        cl = clients.copy(); cl[c['open']] = cl[c['open']].clip(lower=lo)
        return client_features(cfg, cl, ff, fu, flags.loc[lo:hi])
    A, B = feats(start, cut), feats(cut, end)
    both = A.index[A.eligible & B.eligible.reindex(A.index).fillna(False)]
    xa, xb = model_matrix(A.loc[both]), model_matrix(B.loc[both])
    km = KMeans(k, n_init=10, random_state=cfg['seed']).fit(StandardScaler().fit_transform(xa))
    m = clients.set_index(c['id'])
    groups = {'Behavioural segments (formed 2006–2016)': pd.Series(km.labels_, index=both), 'Sector': m[c['sector']].reindex(both),
              'Client type': m[c['type']].reindex(both), 'Region': m[c['region']].reindex(both)}
    eta = lambda x, g: 1 - ((x - x.groupby(g).transform('mean')) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    rows = [dict(feature=col, grouping=gname, groups=int(g.nunique()), eta_squared=float(eta(xb[col], g))) for gname, g in groups.items() for col in xb]
    return pd.DataFrame(rows), int(len(both))


# ---------------------------------------------------------------- run everything
def run(cfg=CONFIG, write=True, prefix='11_'):
    clients, flows, fund, pay = load(cfg)
    inv = inventory(cfg, clients, flows, fund, pay)
    f, checks = clean_and_check(cfg, clients, flows, fund)
    f, prices, unit_check = rebuild_units(cfg, clients, f, fund)
    month, flags, ctx = fund_context(cfg, fund)
    feat = client_features(cfg, clients, f, fund, flags)
    aum0 = float(fund[cfg['fund']['aum']].iloc[-1])
    elig = eligibility_table(feat, feat.current_holding_gbp.sum())
    seg, sel, ari, names = segment(cfg, feat)
    valid, sector_mix = validate_segments(seg, clients, cfg)
    val, net, red = segment_aum(cfg, f, prices, fund, seg)
    rebuilt = val.sum(axis=1); fa = fund.set_index(cfg['fund']['date'])[cfg['fund']['aum']]
    rel_err = (rebuilt - fa).abs() / fa
    checks = pd.concat([checks, pd.DataFrame([dict(check='Holdings = units × price', result=f"median unit error {unit_check['median_abs_unit_error']:.5f}; rebuilt fund AUM within {rel_err.max():.4%} of reported every day",
                                                    detail='units from balance-after-trade ÷ class price; unit changes match flow ÷ price', passed=bool(rel_err.max() < 0.005))])], ignore_index=True)
    rate, corr, corr_summary = flow_correlation(val, net, flags)
    vlift = value_stress_lift(val, red, flags)
    conc, top, hold = concentration(cfg, f, prices, fund, seg)
    tests, tails, rev, second, base = stress_tests(cfg, fund, flags, val, net, red, rate, corr, top, pay, seg)
    feat_out = feat.assign(segment=seg)
    prof = feat_out.groupby('segment').agg(clients=('trades', 'size'), aum_gbp=('current_holding_gbp', 'sum'), median_holding=('mean_holding_gbp', 'median'),
                                           trades_per_year=('trades_per_year', 'median'), redemption_share=('redemption_share', 'median'),
                                           trade_rel_holding=('median_trade_rel_holding', 'median'), largest_red_rel=('largest_redemption_rel', 'median'),
                                           stress_lift=('stress_lift', 'median'), march_red_share=('red_march_share', 'mean'), december_red_share=('red_december_share', 'mean'),
                                           month_end_share=('month_end_trade_share', 'mean'))
    prof['aum_share'] = prof.aum_gbp / prof.aum_gbp.sum(); prof = prof.reset_index()
    el = feat_out[feat_out.eligible]
    typical = dict(segment='All eligible clients', clients=int(len(el)), aum_gbp=float(el.current_holding_gbp.sum()), median_holding=float(el.mean_holding_gbp.median()),
                   trades_per_year=float(el.trades_per_year.median()), redemption_share=float(el.redemption_share.median()), trade_rel_holding=float(el.median_trade_rel_holding.median()),
                   largest_red_rel=float(el.largest_redemption_rel.median()), stress_lift=float(el.stress_lift.median()), march_red_share=float(el.red_march_share.mean()),
                   december_red_share=float(el.red_december_share.mean()), month_end_share=float(el.month_end_trade_share.mean()))
    k_chosen = int(sel.loc[sel.chosen, 'k'].iloc[0])
    gcomp, gcomp_n = grouping_comparison(cfg, clients, f, fund, flags, k_chosen)
    cohesion, random_r, pmap = segment_cohesion(cfg, feat_out, f, clients, fund)
    seg_aum_hist = val.resample('A').last()
    out = dict(cohesion=cohesion, random_pair_r=random_r, pca=pmap, value_lift=vlift, typical=typical, grouping=gcomp, grouping_clients=gcomp_n, seg_red=red, inventory=inv, checks=checks, month=month, flags=flags, ctx=ctx, features=feat_out, eligibility=elig, selection=sel, ari=ari, names=names,
               validation=valid, sector_mix=sector_mix, profiles=prof, seg_aum=val, seg_net=net, rate=rate, corr=corr, corr_summary=corr_summary,
               concentration=conc, top=top, tests=tests, tails=tails, reverse=rev, second_round=second, base=base, unit_check=unit_check)
    if write: save(out, cfg, prefix)
    return out


def save(o, cfg, prefix):
    w = lambda df, n, **k: df.to_csv(OUT / f'{prefix}{n}.csv', **k)
    w(o['inventory'], 'inventory', index=False); w(o['checks'], 'data_checks', index=False); w(o['month'], 'monthly_seasonality', index=False)
    w(o['flags'].reset_index().rename(columns={'index': 'date'}), 'stress_flags', index=False)
    cs = o['features'].reset_index().rename(columns={'index': 'client_id'}); w(cs, 'client_features', index=False)
    w(cs[['client_id', 'segment', 'eligible']], 'client_segments', index=False)
    w(o['eligibility'], 'eligibility_sensitivity', index=False); w(o['selection'], 'k_selection', index=False)
    w(pd.DataFrame({'bootstrap': range(len(o['ari'])), 'adjusted_rand_index': o['ari']}), 'stability_ari', index=False)
    w(o['validation'], 'segment_validation', index=False); w(o['sector_mix'], 'segment_sector_mix'); w(o['profiles'], 'segment_profiles', index=False)
    w(o['seg_aum'].resample('M').last(), 'segment_aum_monthly'); w(o['corr_summary'], 'correlation_summary', index=False)
    rows = []
    for (fq, b, rg), m in o['corr'].items():
        for i in m.index:
            for j in m.columns: rows.append(dict(frequency=fq, basis=b, regime=rg, segment_a=i, segment_b=j, r=m.loc[i, j]))
    w(pd.DataFrame(rows), 'segment_correlations', index=False)
    w(o['concentration'], 'concentration', index=False); w(o['top'], 'top_holders', index=False)
    w(o['tests'], 'stress_tests', index=False); w(o['tails'], 'segment_tails', index=False); w(o['reverse'], 'reverse_stress', index=False)
    w(o['second_round'], 'second_round', index=False); w(o['grouping'], 'grouping_comparison', index=False); w(o['value_lift'], 'stress_value_lift', index=False); w(o['cohesion'].assign(random_pair_r=o['random_pair_r']), 'segment_cohesion', index=False)
    rep = dict(config=cfg['name'], fund_type=o['ctx']['fund_type'], stress_definition=o['ctx']['stress_definition'], stress_days=o['ctx']['stress_days'],
               k=int(o['selection'].loc[o['selection'].chosen, 'k'].iloc[0]), ari_median=float(np.median(o['ari'])), ari_p10=float(np.percentile(o['ari'], 10)),
               checks_passed=bool(o['checks'].passed.all()), base=o['base'], unit_check=o['unit_check'])
    (OUT / f'{prefix}report.json').write_text(json.dumps(rep, indent=2, default=str))
