"""The shared data layer: parameterised queries for the dashboard, forecasts and segments.

Client filters: None means all, [] means nothing. Every function takes an optional `con` (the build
commands share one connection); without it a read-only connection is opened and closed.
"""
import numpy as np
import pandas as pd

from . import db

QUERY_VERSION = '3'
AMOUNTS = {'subscriptions': 'greatest(flow_gbp,0)', 'redemptions': 'greatest(-flow_gbp,0)', 'net_flow': 'flow_gbp'}
GROUPS = {'sector': 'client_sector', 'type': 'client_type'}
PATTERNS = ('weekday', 'month_position', 'month')



def inventory(con=None):
    """Usable columns from the import manifest; adapt legacy complete imports."""
    with db.using(con) as c:
        run = db.latest_import(c) or {}
        recorded = run.get('validation', {}).get('inventory')
        if recorded is not None:
            return recorded
        result = {}
        for table in db.KEYS:
            columns = c.execute("SELECT column_name FROM information_schema.columns WHERE table_name=?", [table]).fetchall()
            count = c.execute(f'SELECT count(*) FROM {table}').fetchone()[0] if columns else 0
            result[table] = dict(supplied_columns=[x[0] for x in columns],
                                 usable_columns=[x[0] for x in columns] if count else [],
                                 row_count=count, issues=[])
        return result


def capabilities(con=None):
    """One dependency registry for query preflight and dashboard availability."""
    with db.using(con) as c:
        inv = inventory(c)
        run = db.latest_import(c) or {}
        validation = run.get('validation', {})
        usable = {t: set(v.get('usable_columns', [])) if v.get('row_count', 1) else set()
                  for t, v in inv.items()}
        def fields(table, *columns):
            return [(table, column) for column in columns]
        ids = fields('clients', 'client_id')
        linked = ids + fields('client_flows', 'client_id')
        exposure = ids + fields('clients', 'account_open_date', 'account_close_date')
        flow = fields('client_flows', 'flow_gbp')
        deps = {
            'sector': linked + fields('clients', 'client_sector'),
            'type': linked + fields('clients', 'client_type'),
            'clients': linked,
            'concentration': linked + flow,
            'exposure': exposure,
            'timing': [],
            'correlations': linked + flow + exposure,
            'aum': fields('fund_daily', 'date', 'closing_aum_gbp'),
            'distributions': fields('income_distributions', 'payment_date', 'payment_status', 'distribution_gbp'),
            'forecast': [(t, col) for t, columns in db.EXPECTED.items() for col in columns.split(',')],
            'segments': [(t, col) for t, columns in db.EXPECTED.items() for col in columns.split(',')],
        }
        for basis, date in [('cash', 'cash_flow_date'), ('trade', 'trade_date')]:
            deps['flow_'+basis] = flow + fields('client_flows', date)
            deps['counts_'+basis] = fields('client_flows', date, 'transaction_id')
        cov = validation.get('coverage')
        if cov is not None:
            complete = bool(cov.get('complete'))
        elif 'date' in usable.get('fund_daily', set()):
            # Legacy imports materialised a complete calendar across the fund's
            # valuation coverage (including weekends and holidays).
            exists = c.execute("SELECT 1 FROM information_schema.tables WHERE table_name='calendar'").fetchone()
            lo, hi, n = c.execute('SELECT min(date),max(date),count(DISTINCT date) FROM calendar').fetchone() if exists else (None, None, 0)
            complete = lo is not None and n == (hi-lo).days+1
        else:
            complete = False
        result = {}
        for name, requirements in deps.items():
            reasons = []
            for table, column in dict.fromkeys(requirements):
                if column not in usable.get(table, set()):
                    detail = next((str(i) for i in inv.get(table, {}).get('issues', [])
                                   if column in str(i)), None)
                    reasons.append(f'{table}.{column}: {detail or "missing or invalid field"}')
            if name in ('timing', 'exposure', 'correlations', 'forecast', 'segments') and not complete:
                reasons.append('Confirmed complete daily coverage is unavailable')
            if name in ('forecast', 'segments'):
                for check, status in validation.get('reconciliation', {}).items():
                    if status != 'passed':
                        reasons.append(f'Reconciliation {check}: {status}')
            result[name] = dict(available=not reasons, reasons=reasons)
        return result


def require_feature(name, con=None):
    state = capabilities(con)[name]
    if not state['available']:
        raise db.FeatureUnavailable(f'{name} unavailable: ' + '; '.join(state['reasons']))


def _validate_filters(sectors, client_types, client_ids, con):
    for values, feature in [(sectors, 'sector'), (client_types, 'type'), (client_ids, 'clients')]:
        if values is not None:
            require_feature(feature, con)


def _flow_query_parts(start, end, sectors, client_types, client_ids, basis, con):
    _validate_filters(sectors, client_types, client_ids, con)
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    joined = sectors is not None or client_types is not None
    if joined:
        source = 'client_flows f JOIN clients c USING(client_id)'
    else:
        source = 'client_flows f'
        where = where.replace('c.client_id', 'f.client_id')
    return date, where, params, source

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
    """Display bounds from valid observation dates, falling back to event dates."""
    with db.using(con) as c:
        inv = inventory(c)
        for table, columns in [('fund_daily', ['date']), ('client_flows', ['cash_flow_date', 'trade_date']),
                               ('income_distributions', ['payment_date', 'ex_distribution_date'])]:
            for column in columns:
                if column in inv.get(table, {}).get('usable_columns', []):
                    lo, hi = c.execute(f'SELECT min({column}),max({column}) FROM {table}').fetchone()
                    if lo is not None:
                        return lo, hi
    return None, None


def distinct_values(column, sectors=None, con=None):
    """Sorted sectors, or client types (optionally within sectors)."""
    if column not in GROUPS.values():
        raise ValueError('column must be client_sector or client_type')
    require_feature('sector' if column == 'client_sector' else 'type', con)
    _validate_filters(sectors, None, None, con)
    where, params = _clients(sectors)
    return _query(f'SELECT DISTINCT {column} FROM clients c WHERE {where} ORDER BY 1', params, con).iloc[:, 0].tolist()


def client_list(sectors=None, client_types=None, con=None):
    require_feature('clients', con)
    _validate_filters(sectors, client_types, None, con)
    where, params = _clients(sectors, client_types)
    return _query(f'SELECT client_id, coalesce(client_name,client_id) client_name, client_sector FROM clients c WHERE {where} ORDER BY client_name, client_id', params, con)


def daily_flows(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """Daily amounts/counts; zero fill only within confirmed observation coverage."""
    with db.using(con) as c:
        caps = capabilities(c)
        if basis not in ('cash', 'trade'):
            raise ValueError('basis must be cash or trade')
        amounts, counts = caps['flow_'+basis]['available'], caps['counts_'+basis]['available']
        if not amounts and not counts:
            require_feature('flow_'+basis, c)
        date, where, params, source = _flow_query_parts(start, end, sectors, client_types, client_ids, basis, c)
        start, end = _window(start, end, c)
        values = ('sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, '
                  'sum(flow_gbp) net_flow' if amounts else
                  'NULL::DOUBLE subscriptions, NULL::DOUBLE redemptions, NULL::DOUBLE net_flow')
        values += ', ' + ('count(*)' if counts else 'NULL::BIGINT') + ' transaction_count'
        aggregate = f'SELECT {date} date, {values} FROM {source} WHERE {where} GROUP BY 1'
        if caps['timing']['available']:
            selects = ', '.join((f'coalesce(a.{key},0)' if available else 'NULL::DOUBLE') + f' {key}'
                                for key, available in [('subscriptions', amounts), ('redemptions', amounts),
                                                        ('net_flow', amounts), ('transaction_count', counts)])
            return _query(f"WITH a AS ({aggregate}) SELECT cal.date,{selects} FROM calendar cal LEFT JOIN a USING(date) WHERE cal.date BETWEEN ? AND ? ORDER BY cal.date", params+[start,end], c)
        return _query(aggregate+' ORDER BY 1', params, c)


def group_summary(group, start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """Flows by sector or client type, largest redemptions first."""
    if group not in GROUPS:
        raise ValueError('group must be sector or type')
    require_feature(group, con)
    require_feature('flow_'+basis, con)
    _validate_filters(sectors, client_types, client_ids, con)
    column = GROUPS[group]
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    counts = capabilities(con)['counts_'+basis]['available']
    count_sql = 'count(*)' if counts else 'NULL::BIGINT'
    average_sql = 'avg(abs(flow_gbp))' if counts else 'NULL::DOUBLE'
    return _query(f'''SELECT c.{column}, sum(greatest(flow_gbp,0)) AS subscriptions, sum(greatest(-flow_gbp,0)) AS redemptions, sum(flow_gbp) AS net_flow,
    {count_sql} AS transaction_count, count(DISTINCT f.client_id) AS transacting_clients, {average_sql} AS average_transaction_gbp
    FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1 ORDER BY redemptions DESC, 1''', params, con)


def client_concentration(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash', con=None):
    """Transacting clients, largest gross flow first."""
    require_feature('concentration', con)
    require_feature('flow_'+basis, con)
    _validate_filters(sectors, client_types, client_ids, con)
    date, where, params = _filters(start, end, sectors, client_types, client_ids, basis)
    counts = capabilities(con)['counts_'+basis]['available']
    count_sql = 'count(*)' if counts else 'NULL::BIGINT'
    average_sql = 'avg(abs(flow_gbp))' if counts else 'NULL::DOUBLE'
    return _query(f'''SELECT f.client_id, coalesce(c.client_name,c.client_id) client_name, {count_sql} AS transactions, sum(abs(flow_gbp)) AS gross_flow_gbp, {average_sql} AS average_transaction_gbp
    FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1,2 ORDER BY gross_flow_gbp DESC, 1''', params, con)


def active_clients(start=None, end=None, sectors=None, client_types=None, client_ids=None, con=None):
    """Accounts open on each day, respecting historical opening and closure."""
    require_feature('exposure', con)
    _validate_filters(sectors, client_types, client_ids, con)
    where, params = _clients(sectors, client_types, client_ids)
    start, end = _window(start, end, con)
    return _query(f'''WITH days AS (SELECT CAST(day AS DATE) AS date FROM generate_series(?::DATE, ?::DATE, INTERVAL 1 DAY) t(day))
    SELECT d.date, count(c.account_open_date) AS active_clients FROM days d LEFT JOIN (SELECT account_open_date, account_close_date FROM clients c WHERE {where}) c
    ON c.account_open_date <= d.date AND (c.account_close_date IS NULL OR c.account_close_date >= d.date) GROUP BY 1 ORDER BY 1''', [start, end] + params, con)


def timing_summary(daily, active=None, pattern='weekday'):
    """Calendar patterns over all observed days (zero-flow days included) with per-client-day rates."""
    if pattern not in PATTERNS:
        raise ValueError(f'pattern must be one of {PATTERNS}')
    daily = daily.merge(active, on='date', how='left') if active is not None else daily.assign(active_clients=np.nan)
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
        subscriptions=('subscriptions', lambda x: x.sum(min_count=1)), redemptions=('redemptions', lambda x: x.sum(min_count=1)),
        observations=('date', 'size'), transaction_count=('transaction_count', lambda x: x.sum(min_count=1)),
        client_days=('active_clients', lambda x: x.sum(min_count=1))).reindex(ordering).dropna(subset=['observations']).reset_index()
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
    require_feature('correlations', con)
    require_feature('flow_'+basis, con)
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
    require_feature('aum', con)
    start, end = _window(start, end, con)
    return _query('SELECT date, closing_aum_gbp FROM fund_daily WHERE date BETWEEN ? AND ? ORDER BY date', [start, end], con)


def scheduled_distributions(after, sector=None, con=None):
    """Declared distributions do not need a client join unless sector-filtered."""
    require_feature('distributions', con)
    source, extra, params = 'income_distributions d', '', [after]
    if sector is not None:
        require_feature('sector', con)
        if 'client_id' not in inventory(con).get('income_distributions', {}).get('usable_columns', []):
            raise db.FeatureUnavailable('Missing income_distributions.client_id')
        source += ' JOIN clients c USING(client_id)'
        extra = ' AND client_sector = ?'
        params.append(sector)
    return _query(f"SELECT payment_date,payment_status,sum(distribution_gbp) scheduled_distribution_gbp FROM {source} WHERE payment_status='Declared' AND payment_date > ?{extra} GROUP BY 1,2 ORDER BY 1", params, con)


def table_counts(con=None):
    return _query("SELECT 'Clients' AS dataset,count(*) AS rows FROM clients UNION ALL SELECT 'Client flows',count(*) FROM client_flows "
                  "UNION ALL SELECT 'Income distributions',count(*) FROM income_distributions UNION ALL SELECT 'Daily fund',count(*) FROM fund_daily", con=con)


def sector_history(con=None, by_sector=True):
    """Daily cash-date subscriptions and redemptions per sector, or summed across sectors."""
    require_feature('forecast', con)
    if by_sector:
        return _query('SELECT date, client_sector, subscriptions, redemptions FROM daily_sector ORDER BY date, client_sector', con=con)
    return _query('SELECT date, sum(subscriptions) subscriptions, sum(redemptions) redemptions FROM daily_sector GROUP BY 1 ORDER BY 1', con=con)


def unit_prices(con=None):
    """Daily accumulation unit price, the fund-return input for the performance backtest."""
    require_feature('forecast', con)
    return _query('SELECT date, accumulation_unit_price_gbp price FROM fund_daily ORDER BY date', con=con)


def load_tables(names, con=None):
    """Whole typed source tables, ordered by primary key."""
    unknown = set(names) - set(db.KEYS)
    if unknown:
        raise ValueError(f'unknown tables {sorted(unknown)}')
    with db.using(con) as c:
        return {n: c.execute(f'SELECT * FROM {n} ORDER BY {db.KEYS[n]}').df() for n in names}
