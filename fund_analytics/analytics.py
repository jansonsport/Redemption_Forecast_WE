"""Small, parameterised query results for notebooks and the dashboard."""
import json
import pandas as pd
from .db import connect

QUERY_VERSION = '1'

def _query(sql, params=None):
    with connect() as con:
        return con.execute(sql, params or []).df()

def metadata():
    with connect() as con:
        cursor = con.execute('SELECT * FROM ingestion_runs ORDER BY imported_at DESC LIMIT 1')
        row = cursor.fetchone()
        if not row:
            return {}
        result = dict(zip([c[0] for c in cursor.description], row))
    for key in ('source_hashes', 'row_counts', 'validation'):
        result[key] = json.loads(result[key])
    return result

def _filters(start=None, end=None, sectors=None, client_types=None, client_ids=None, basis='cash'):
    if basis not in ('cash', 'trade'):
        raise ValueError('basis must be cash or trade')
    date = 'f.cash_flow_date' if basis == 'cash' else 'f.trade_date'
    terms, params = [], []
    for op, value in [('>=',start), ('<=',end)]:
        if value is not None:
            terms.append(f'{date}{op}?::DATE'); params.append(value)
    for column, values in [('c.client_sector',sectors),('c.client_type',client_types),('c.client_id',client_ids)]:
        if values is not None:
            if not values:
                terms.append('FALSE')
            else:
                terms.append(f'{column} IN ({",".join("?" for _ in values)})'); params.extend(values)
    return date, ' AND '.join(terms) or 'TRUE', params

def daily_flows(start=None,end=None,sectors=None,client_types=None,client_ids=None,basis='cash'):
    date, where, params = _filters(start,end,sectors,client_types,client_ids,basis)
    info = metadata()
    lo, hi = start or info['coverage_start'], end or info['coverage_end']
    return _query(f'''WITH a AS (SELECT {date} date, sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, sum(flow_gbp) net_flow, count(*) transaction_count FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY 1)
    SELECT cal.date, coalesce(a.subscriptions,0) subscriptions, coalesce(a.redemptions,0) redemptions, coalesce(a.net_flow,0) net_flow, coalesce(a.transaction_count,0) transaction_count FROM calendar cal LEFT JOIN a USING(date) WHERE cal.date BETWEEN ?::DATE AND ?::DATE ORDER BY cal.date''', params+[lo,hi])

def client_summary(start=None,end=None,sectors=None,client_types=None,client_ids=None,basis='cash'):
    date, where, params = _filters(start,end,sectors,client_types,client_ids,basis)
    return _query(f'''SELECT c.client_id, any_value(client_name) client_name, any_value(client_sector) client_sector, any_value(client_type) client_type, count(*) transaction_count, sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, sum(flow_gbp) net_flow, avg(abs(flow_gbp)) average_transaction_gbp, min({date}) first_transaction, max({date}) last_transaction FROM client_flows f JOIN clients c USING(client_id) WHERE {where} GROUP BY c.client_id ORDER BY subscriptions+redemptions DESC''', params)

def timing_patterns(start=None,end=None,sectors=None,client_types=None,client_ids=None,basis='cash',group='weekday'):
    """Daily averages and per-open-account exposure, including zero-flow days."""
    expressions = {'weekday':'isodow(date)', 'month':'month(date)', 'month_position': 'least(4,ceil(day(date)/7.0))'}
    if group not in expressions:
        raise ValueError('group must be weekday, month or month_position')
    info = metadata()
    start, end = start or info['coverage_start'], end or info['coverage_end']
    daily = daily_flows(start,end,sectors,client_types,client_ids,basis)
    terms, params = ['TRUE'], []
    for column, values in [('client_sector',sectors),('client_type',client_types),('client_id',client_ids)]:
        if values is not None:
            if not values:
                terms.append('FALSE')
            else:
                terms.append(f'{column} IN ({",".join("?" for _ in values)})'); params.extend(values)
    # Sweep account starts/ends against calendar rather than materialising account x day.
    sql = f'''WITH selected AS (SELECT * FROM clients WHERE {' AND '.join(terms)}), changes AS (
    SELECT greatest(account_open_date,?::DATE) date, 1 delta FROM selected WHERE account_open_date<=?::DATE AND coalesce(account_close_date,?::DATE)>=?::DATE
    UNION ALL SELECT account_close_date+1 date,-1 delta FROM selected WHERE account_close_date BETWEEN ?::DATE AND ?::DATE),
    grouped AS (SELECT date,sum(delta) delta FROM changes GROUP BY date)
    SELECT c.date, sum(coalesce(delta,0)) OVER (ORDER BY c.date) active_accounts FROM calendar c LEFT JOIN grouped USING(date) WHERE c.date BETWEEN ?::DATE AND ?::DATE ORDER BY c.date'''
    exposure = _query(sql, params+[start,end,end,start,start,end,start,end])
    daily = daily.merge(exposure,on='date',how='left')
    with connect() as con:
        con.register('selected_daily', daily)
        return con.execute(f'''SELECT {expressions[group]} period, count(*) observed_days, sum(active_accounts) account_days, sum(transaction_count) transaction_count, avg(subscriptions) mean_daily_subscriptions, avg(redemptions) mean_daily_redemptions, sum(subscriptions) subscriptions, sum(redemptions) redemptions, sum(transaction_count)/nullif(sum(active_accounts),0) transactions_per_account_day, (sum(subscriptions)+sum(redemptions))/nullif(sum(transaction_count),0) average_transaction_gbp FROM selected_daily GROUP BY 1 ORDER BY 1''').df()

def correlations(client_ids,start=None,end=None,frequency='month',target='net_flow',basis='cash'):
    """Return correlation, pairwise overlap counts and zero-filled long histories."""
    if not client_ids or len(set(client_ids)) > 50:
        raise ValueError('Select between 1 and 50 distinct clients')
    if frequency not in ('month','week') or target not in ('net_flow','subscriptions','redemptions'):
        raise ValueError('Unsupported frequency or target')
    info = metadata()
    start, end = start or info['coverage_start'], end or info['coverage_end']
    date, where, params = _filters(start,end,client_ids=client_ids,basis=basis)
    placeholders = ','.join('?' for _ in client_ids)
    sql = f'''WITH periods AS (SELECT DISTINCT date_trunc('{frequency}',date)::DATE period FROM calendar WHERE date BETWEEN ?::DATE AND ?::DATE),
    grid AS (SELECT p.period,c.client_id FROM periods p CROSS JOIN clients c WHERE c.client_id IN ({placeholders}) AND c.account_open_date<=least(p.period+INTERVAL 1 {frequency}-INTERVAL 1 DAY,?::DATE) AND coalesce(c.account_close_date,?::DATE)>=greatest(p.period,?::DATE)),
    a AS (SELECT date_trunc('{frequency}',{date})::DATE period, f.client_id, sum(flow_gbp) net_flow,sum(greatest(flow_gbp,0)) subscriptions,sum(greatest(-flow_gbp,0)) redemptions FROM client_flows f JOIN clients c USING(client_id) WHERE {where} AND {date} BETWEEN c.account_open_date AND coalesce(c.account_close_date,?::DATE) GROUP BY 1,2)
    SELECT g.period,g.client_id,coalesce(a.net_flow,0) net_flow,coalesce(a.subscriptions,0) subscriptions,coalesce(a.redemptions,0) redemptions FROM grid g LEFT JOIN a USING(period,client_id) ORDER BY 1,2'''
    rows = _query(sql,[start,end]+list(client_ids)+[end,end,start]+params+[end])
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), rows
    matrix = rows.pivot(index='period',columns='client_id',values=target)
    observed = matrix.notna().astype(int)
    overlap = observed.T @ observed
    corr = matrix.corr(min_periods=12 if frequency=='month' else 26)
    return corr, overlap, rows
