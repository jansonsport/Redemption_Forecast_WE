"""The shared data layer: parameterised queries for the dashboard, forecasts and segments.

Client filters: None means all, [] means nothing. Every function takes an optional `con` (the build
commands share one connection); without it a read-only connection is opened and closed.
"""
import numpy as np
import pandas as pd

from . import db

QUERY_VERSION = '2'
AMOUNTS = {'subscriptions': 'greatest(flow_gbp,0)', 'redemptions': 'greatest(-flow_gbp,0)', 'net_flow': 'flow_gbp'}
GROUPS = {'sector': 'client_sector', 'type': 'client_type'}
PATTERNS = ('weekday', 'month_position', 'month')


def _query(sql, params=None, con=None):
    with db.using(con) as c:
        return c.execute(sql, list(params or [])).df()


def _clients(sectors=None, client_types=None, client_ids=None):
    """WHERE terms on clients aliased `c`."""
    terms, params = [], []
    for column, values in [('c.client_sector', sectors), ('c.client_type', client_types), ('c.client_id', client_ids)]:
        if values is not None:
            if not values:
                terms.append('FALSE')
            else:
                terms.append(f'{column} IN ({",".join("?" for _ in values)})'); params.extend(values)
    return ' AND '.join(terms) or 'TRUE', params


def _filters(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash'):
    if basis not in ('cash', 'trade'):
        raise ValueError('basis must be cash or trade')
    date = 'f.cash_flow_date' if basis == 'cash' else 'f.trade_date'
    terms, params = [], []
    for op, value in [('>=', start), ('<=', end)]:
        if value is not None:
            terms.append(f'{date}{op}?::DATE'); params.append(value)
    where, more = _clients(sectors, client_types, client_ids)
    return date, ' AND '.join(terms + [where]), params + more


def _window(start, end, con):
    if start is None or end is None:
        lo, hi = coverage(con)
        start, end = start or lo, end or hi
    return start, end


def coverage(con=None):
    """First and last fund observation dates."""
    with db.using(con) as c:
        return c.execute('SELECT min(date), max(date) FROM fund_daily').fetchone()


def distinct_values(column, sectors=None, con=None):
    """Sorted sectors, or client types (optionally within sectors)."""
    if column not in GROUPS.values():
        raise ValueError('column must be client_sector or client_type')
    where, params = _clients(sectors)
    return _query(f'SELECT DISTINCT {column} FROM clients c WHERE {where} ORDER BY 1', params, con).iloc[:, 0].tolist()


def client_list(sectors=None, client_types=None, con=None):
    where, params = _clients(sectors, client_types)
    return _query(f'SELECT client_id, client_name, client_sector FROM clients c WHERE {where} ORDER BY client_name, client_id', params, con)


def daily_flows(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """One row per calendar day, zero-filled."""
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    start, end = _window(start, end, con)
    return _query(f'''WITH a AS (SELECT {date} date, sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, sum(flow_gbp) net_flow, count(*) transaction_count FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1)
    SELECT cal.date, coalesce(a.subscriptions,0) subscriptions, coalesce(a.redemptions,0) redemptions, coalesce(a.net_flow,0) net_flow, coalesce(a.transaction_count,0) transaction_count FROM calendar cal LEFT JOIN a USING(date) WHERE cal.date BETWEEN ?::DATE AND ?::DATE ORDER BY cal.date''', params + [start, end], con)


def group_summary(group, start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """Flows by sector or client type, largest redemptions first."""
    if group not in GROUPS:
        raise ValueError('group must be sector or type')
    column = GROUPS[group]
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    return _query(f'''SELECT c.{column}, sum(greatest(flow_gbp,0)) AS subscriptions, sum(greatest(-flow_gbp,0)) AS redemptions, sum(flow_gbp) AS net_flow,
    count(*) AS transaction_count, count(DISTINCT f.client_id) AS transacting_clients, avg(abs(flow_gbp)) AS average_transaction_gbp
    FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1 ORDER BY redemptions DESC, 1''', params, con)


def client_concentration(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """Transacting clients, largest gross flow first."""
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    return _query(f'''SELECT f.client_id, c.client_name, count(*) AS transactions, sum(abs(flow_gbp)) AS gross_flow_gbp, avg(abs(flow_gbp)) AS average_transaction_gbp
    FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1,2 ORDER BY gross_flow_gbp DESC, 1''', params, con)


def active_clients(start=None, end=None, sectors=None, client_types=None, client_ids=None, con=None):
    """Accounts open on each day, respecting historical opening and closure."""
    where, params = _clients(sectors, client_types, client_ids)
    start, end = _window(start, end, con)
    return _query(f'''WITH days AS (SELECT CAST(day AS DATE) AS date FROM generate_series(?::DATE, ?::DATE, INTERVAL 1 DAY) t(day))
    SELECT d.date, count(c.account_open_date) AS active_clients FROM days d LEFT JOIN (SELECT account_open_date, account_close_date FROM clients c WHERE {where}) c
    ON c.account_open_date <= d.date AND (c.account_close_date IS NULL OR c.account_close_date >= d.date) GROUP BY 1 ORDER BY 1''', [start, end] + params, con)


def timing_summary(daily, active, pattern='weekday'):
    """Calendar patterns over all observed days (zero-flow days included) with per-client-day rates."""
    if pattern not in PATTERNS:
        raise ValueError(f'pattern must be one of {PATTERNS}')
    daily = daily.merge(active, on='date', how='left')
    if pattern == 'weekday':
        daily['period'] = daily.date.dt.day_name()
        ordering = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif pattern == 'month_position':
        ordering = ['1–7', '8–14', '15–21', '22–end']
        daily['period'] = pd.cut(daily.date.dt.day, [0, 7, 14, 21, 31], labels=ordering)
    else:
        daily['period'] = daily.date.dt.month_name()
        ordering = pd.date_range('2024-01-01', periods=12, freq='MS').month_name().tolist()
    summary = daily.groupby('period', observed=True).agg(
        subscriptions=('subscriptions', 'sum'), redemptions=('redemptions', 'sum'),
        observations=('date', 'size'), transaction_count=('transaction_count', 'sum'),
        client_days=('active_clients', 'sum')).reindex(ordering).dropna(subset=['observations']).reset_index()
    summary['subscriptions_per_day'] = summary.subscriptions / summary.observations
    summary['redemptions_per_day'] = summary.redemptions / summary.observations
    summary['transactions_per_1000_client_days'] = summary.transaction_count / summary.client_days.replace(0, np.nan) * 1000
    summary['average_transaction_gbp'] = (summary.subscriptions + summary.redemptions) / summary.transaction_count.replace(0, np.nan)
    return summary


def client_histories(client_ids, unit='month', target='net_flow', start=None, end=None, basis='cash', con=None):
    """Per-client period totals: zero while the account is open, absent outside its lifetime.

    Cash that settles after the account closed is still counted in its period."""
    if unit not in ('month', 'week') or target not in AMOUNTS or basis not in ('cash', 'trade'):
        raise ValueError('Unsupported unit, target or basis')
    date = 'cash_flow_date' if basis == 'cash' else 'trade_date'
    start, end = _window(start, end, con)
    ids = list(client_ids)
    marks = ','.join('?' for _ in ids) or 'NULL'
    return _query(f'''WITH selected AS (SELECT * FROM clients WHERE client_id IN ({marks})),
    periods AS (SELECT CAST(p AS DATE) AS period FROM generate_series(date_trunc('{unit}',?::DATE),date_trunc('{unit}',?::DATE),INTERVAL 1 {unit}) t(p)),
    flows AS (SELECT client_id, date_trunc('{unit}',{date}) AS period, sum({AMOUNTS[target]}) AS amount
      FROM client_flows WHERE client_id IN ({marks}) AND {date} BETWEEN ? AND ? GROUP BY 1,2)
    SELECT p.period,c.client_id,coalesce(f.amount,0) AS amount FROM periods p CROSS JOIN selected c
    LEFT JOIN flows f ON f.client_id=c.client_id AND f.period=p.period
    WHERE c.account_open_date <= least(?::DATE,p.period+INTERVAL 1 {unit}-INTERVAL 1 DAY)
      AND coalesce(c.account_close_date,?::DATE) >= greatest(?::DATE,p.period)
    ORDER BY 1,2''', ids + [start, end] + ids + [start, end, end, end, start], con)


def correlations(client_ids, unit='month', target='net_flow', start=None, end=None, basis='cash', con=None):
    """Pearson correlation (min 12 months / 26 weeks shared), shared-period counts and the histories."""
    if not client_ids or len(set(client_ids)) > 50:
        raise ValueError('Select between 1 and 50 distinct clients')
    rows = client_histories(client_ids, unit, target, start, end, basis, con)
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), rows
    matrix = rows.pivot(index='period', columns='client_id', values='amount')
    present = matrix.notna().astype(int)
    return matrix.corr(min_periods=12 if unit == 'month' else 26), present.T @ present, rows


def fund_aum(start=None, end=None, con=None):
    """Whole-fund closing AUM; client filters do not apply."""
    start, end = _window(start, end, con)
    return _query('SELECT date, closing_aum_gbp FROM fund_daily WHERE date BETWEEN ? AND ? ORDER BY date', [start, end], con)


def scheduled_distributions(after, sector=None, con=None):
    """Declared income payments due after `after`, whole fund or one sector."""
    return _query('''SELECT payment_date, payment_status, sum(distribution_gbp) AS scheduled_distribution_gbp
    FROM income_distributions d JOIN clients c USING(client_id)
    WHERE payment_status = 'Declared' AND payment_date > ? AND (?::VARCHAR IS NULL OR client_sector = ?)
    GROUP BY 1,2 ORDER BY 1''', [after, sector, sector], con)


def table_counts(con=None):
    return _query("SELECT 'Clients' AS dataset,count(*) AS rows FROM clients UNION ALL SELECT 'Client flows',count(*) FROM client_flows "
                  "UNION ALL SELECT 'Income distributions',count(*) FROM income_distributions UNION ALL SELECT 'Daily fund',count(*) FROM fund_daily", con=con)


def sector_history(con=None, by_sector=True):
    """Daily cash-date subscriptions and redemptions per sector, or summed across sectors."""
    if by_sector:
        return _query('SELECT date, client_sector, subscriptions, redemptions FROM daily_sector ORDER BY date, client_sector', con=con)
    return _query('SELECT date, sum(subscriptions) subscriptions, sum(redemptions) redemptions FROM daily_sector GROUP BY 1 ORDER BY 1', con=con)


def load_tables(names, con=None):
    """Whole typed source tables, ordered by primary key."""
    unknown = set(names) - set(db.KEYS)
    if unknown:
        raise ValueError(f'unknown tables {sorted(unknown)}')
    with db.using(con) as c:
        return {n: c.execute(f'SELECT * FROM {n} ORDER BY {db.KEYS[n]}').df() for n in names}
