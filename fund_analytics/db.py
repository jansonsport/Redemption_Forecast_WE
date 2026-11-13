"""The analytics database: location, connections, import history and the typed, atomic import.

`ingest` is the only code that reads the raw CSVs; everything else reads the database.
"""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import os
import time
import tempfile
import uuid
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / 'data/clean'
DB_PATH = Path(os.environ.get('FUND_ANALYTICS_DB') or CLEAN_DIR / 'fund_analytics.duckdb')
RAW_DIR = ROOT / 'data/raw'
SCHEMA_VERSION = '2'
FILES = {'clients': 'synthetic_clients.csv', 'client_flows': 'synthetic_client_flows.csv', 'income_distributions': 'synthetic_income_distributions.csv', 'fund_daily': 'synthetic_fund_daily.csv'}
KEYS = {'clients': 'client_id', 'client_flows': 'transaction_id', 'income_distributions': 'distribution_id', 'fund_daily': 'date'}
EXPECTED = {
'clients': 'client_id,client_name,client_sector,client_type,country,region,unit_type,account_open_date,account_close_date,valuation_date,client_aum_gbp,opening_valuation_date,opening_client_aum_gbp',
'client_flows': 'transaction_id,trade_date,cash_flow_date,cash_flow_status,client_id,flow_gbp,client_aum_after_trade_gbp',
'income_distributions': 'distribution_id,ex_distribution_date,payment_date,client_id,distribution_gbp,payment_status',
'fund_daily': 'date,opening_aum_gbp,subscriptions_trade_date_gbp,redemptions_trade_date_gbp,net_flow_trade_date_gbp,investment_value_change_gbp,income_distributions_declared_gbp,closing_aum_gbp,active_client_count,transaction_count,income_unit_price_gbp,accumulation_unit_price_gbp,subscriptions_cash_date_gbp,redemptions_cash_date_gbp,net_client_flow_cash_date_gbp,income_distributions_paid_gbp,net_investor_cash_flow_gbp,redemptions_awaiting_payment_gbp,income_distributions_payable_gbp'}

class FeatureUnavailable(RuntimeError):
    """A requested analysis lacks valid inputs; safe to display to dashboard users."""


class DatabaseUnavailable(RuntimeError):
    """The database file is missing, locked by another process or unreadable."""


def connect(read_only=True, path=None):
    """Open the database (default `DB_PATH`, looked up at call time)."""
    path = Path(path or DB_PATH)
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.IOException as exc:
        raise DatabaseUnavailable(f'Cannot open the analytics database {path}: {exc}. If the dashboard or another '
                                  'process is using it, stop it and retry; if it does not exist yet, run '
                                  '`venv/bin/python -m fund_analytics ingest`.') from exc


@contextmanager
def using(con=None):
    """Yield `con`, or a read-only connection that is closed afterwards."""
    if con is not None:
        yield con
        return
    with connect() as own:
        yield own


def import_runs(con=None):
    """All imports in insertion order, as dicts with their JSON fields parsed."""
    with using(con) as c:
        if not c.execute("SELECT 1 FROM information_schema.tables WHERE table_name='ingestion_runs'").fetchone():
            return []
        runs = c.execute('SELECT * FROM ingestion_runs ORDER BY rowid').df().to_dict('records')
    for run in runs:
        for key in ('source_hashes', 'row_counts', 'validation'):
            run[key] = json.loads(run[key])
    return runs


def latest_import(con=None):
    """The most recently inserted import, or None."""
    runs = import_runs(con)
    return runs[-1] if runs else None


def require_import(con=None):
    """The latest import; raises when there is no database or no import yet."""
    if con is None and not DB_PATH.exists():
        raise DatabaseUnavailable(f'No analytics database at {DB_PATH}. Run `venv/bin/python -m fund_analytics ingest`.')
    run = latest_import(con)
    if run is None:
        raise LookupError('The analytics database has no import yet. Run `venv/bin/python -m fund_analytics ingest`.')
    return run

def _type(column):
    if column in ('income_unit_price_gbp', 'accumulation_unit_price_gbp'):
        return 'DECIMAL(24,10)'
    if column.endswith('_gbp'):
        return 'DECIMAL(20,2)'
    if column == 'date' or column.endswith('_date'):
        return 'DATE'
    if column in ('transaction_id', 'distribution_id') or column.endswith('_count'):
        return 'BIGINT'
    return 'VARCHAR'

def _require(con, sql, label):
    rows = con.execute(sql).fetchmany(5)
    if rows:
        raise ValueError(f'{label}; example records: {rows}')

def _normal_header(row):
    return [value.strip().lstrip('\ufeff').lower() for value in row]


def _header(source, table, scan_rows, override=None):
    expected = set(EXPECTED[table].split(','))
    candidates = []
    with source.open(newline='', encoding='utf-8-sig') as stream:
        for number, row in enumerate(csv.reader(stream, strict=True), 1):
            if number > (override or scan_rows):
                break
            if override and number != override:
                continue
            normal = _normal_header(row)
            matches = set(normal) & expected
            # One canonical column suffices for an explicitly assigned file. Data rows
            # almost never contain canonical names; multiple candidates are rejected.
            if matches:
                candidates.append((number, normal))
    if len(candidates) != 1:
        raise ValueError(f'{source.name}: expected one header for {table} within {scan_rows} records; '
                         f'found {len(candidates)}. Use --header-row {table}=N.')
    number, header = candidates[0]
    if len(header) != len(set(header)) or any(not c for c in header):
        raise ValueError(f'{source.name}: duplicate or empty header')
    return number, header


def _sources(raw_dir, sources, scan_rows, header_rows):
    assigned = {t: Path(p) for t, p in (sources or {}).items()}
    overridden_defaults = {(raw_dir / FILES[t]).resolve() for t in assigned if t in FILES}
    if set(assigned) - set(FILES) or set(header_rows) - set(FILES):
        raise ValueError('Unknown canonical table in source/header override')
    for table, name in FILES.items():
        if table not in assigned and (raw_dir / name).exists():
            assigned[table] = raw_dir / name
    used = {p.resolve() for p in assigned.values()}
    for source in sorted(raw_dir.glob('*.csv')):
        if source.resolve() in used or source.resolve() in overridden_defaults:
            continue
        scores = {}
        with source.open(newline='', encoding='utf-8-sig') as stream:
            for number, row in enumerate(csv.reader(stream, strict=True), 1):
                if number > scan_rows:
                    break
                names = set(_normal_header(row))
                for table in FILES:
                    # Shared client_id alone does not identify an event file.
                    unique = set(EXPECTED[table].split(',')) - set().union(
                        *(set(v.split(',')) for t, v in EXPECTED.items() if t != table))
                    score = len(names & set(EXPECTED[table].split(',')))
                    if names & unique:
                        scores[table] = max(scores.get(table, 0), score)
        if not scores:
            raise ValueError(f'{source.name}: no recognised table; use --source TABLE=PATH')
        best = max(scores.values())
        roles = [t for t, score in scores.items() if score == best]
        if len(roles) != 1 or roles[0] in assigned:
            raise ValueError(f'{source.name}: ambiguous or duplicate table role; use --source TABLE=PATH')
        assigned[roles[0]] = source
    if set(header_rows) - set(assigned):
        raise ValueError('Header override has no source: ' + ', '.join(sorted(set(header_rows) - set(assigned))))
    if len({p.resolve() for p in assigned.values()}) != len(assigned):
        raise ValueError('Each CSV may supply only one canonical table')
    return {t: (p, *_header(p, t, scan_rows, header_rows.get(t))) for t, p in assigned.items()}


def ingest(raw_dir=RAW_DIR, path=None, *, scan_rows=100, sources=None, header_rows=None):
    """Import one replaceable snapshot; unavailable fields never become invented values."""
    started = time.perf_counter()
    if scan_rows < 1 or any(n < 1 for n in (header_rows or {}).values()):
        raise ValueError('Scan limit and header rows must be positive')
    raw_dir, path = Path(raw_dir), Path(path or DB_PATH)
    selected = _sources(raw_dir, sources, scan_rows, header_rows or {})
    if not selected:
        raise ValueError('No usable canonical CSV files found')
    hashes = {}
    for table, (source, number, header) in selected.items():
        with source.open('rb') as stream:
            hashes[table] = {'file': str(source.resolve()), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
                             'header_row': number, 'columns': header}
    fingerprint = hashlib.sha256(json.dumps([SCHEMA_VERSION, hashes, scan_rows], sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(False, path)
    try:
        con.execute('BEGIN TRANSACTION')
        con.execute('CREATE TABLE IF NOT EXISTS ingestion_runs (run_id VARCHAR PRIMARY KEY, fingerprint VARCHAR, imported_at TIMESTAMP, schema_version VARCHAR, source_hashes JSON, row_counts JSON, validation JSON, coverage_start DATE, coverage_end DATE, duration_seconds DOUBLE)')
        previous = latest_import(con)
        if previous and previous['fingerprint'] == fingerprint:
            con.execute('ROLLBACK')
            return {'status': 'unchanged', 'fingerprint': fingerprint}
        inventory, counts = {}, {}
        for table, names in EXPECTED.items():
            columns = names.split(',')
            item = inventory[table] = dict(source=None, header_row=None, supplied_columns=[], usable_columns=[],
                missing_columns=columns.copy(), extra_columns=[], issues=[], invalid_counts={}, row_count=0)
            con.execute(f'CREATE TEMP TABLE stage_{table} (' + ','.join(f'"{c}" {_type(c)}' for c in columns) + ')')
            if table not in selected:
                item['issues'].append(f'Missing dataset: {table}')
                counts[table] = 0
                continue
            source, number, header = selected[table]
            supplied = [c for c in columns if c in header]
            item.update(source=str(source), header_row=number, supplied_columns=supplied,
                        missing_columns=[c for c in columns if c not in header], extra_columns=[c for c in header if c not in columns])
            # Normalise into a temporary file; original records and source files are preserved.
            with tempfile.TemporaryDirectory(prefix='fund_csv_') as directory:
                normalised = Path(directory) / 'normalised.csv'
                count = 0
                positions = [header.index(c) for c in supplied]
                with source.open(newline='', encoding='utf-8-sig') as stream, normalised.open('w', newline='', encoding='utf-8') as out:
                    reader, writer = csv.reader(stream, strict=True), csv.writer(out)
                    for _ in range(number):
                        next(reader)
                    writer.writerow(supplied)
                    for record, row in enumerate(reader, number + 1):
                        if not row or all(not cell.strip() for cell in row):
                            continue
                        if len(row) != len(header) or _normal_header(row) == header:
                            raise ValueError(f'{source.name}: malformed/repeated header record {record}')
                        writer.writerow([row[i].strip() for i in positions])
                        count += 1
                con.execute(f'CREATE TEMP TABLE raw_{table} (' + ','.join(f'"{c}" VARCHAR' for c in supplied) + ')')
                quoted = str(normalised).replace("'", "''")
                con.execute(f"COPY raw_{table} FROM '{quoted}' (HEADER, DELIMITER ',', NULLSTR '')")
            item['row_count'] = counts[table] = count
            if not count:
                item['issues'].append(f'Empty dataset: {table}')
            usable = []
            for column in supplied:
                nullable = column == 'account_close_date'
                predicate = f'("{column}" IS NOT NULL AND TRY_CAST("{column}" AS {_type(column)}) IS NULL)'
                if _type(column) == 'BIGINT':
                    predicate += f" OR (\"{column}\" IS NOT NULL AND NOT regexp_full_match(\"{column}\", '[+-]?[0-9]+'))"
                if not nullable:
                    predicate += f' OR "{column}" IS NULL'
                invalid = con.execute(f'SELECT count(*) FROM raw_{table} WHERE {predicate}').fetchone()[0]
                if invalid:
                    item['issues'].append(f'Invalid {table}.{column}: {invalid} values')
                    item['invalid_counts'][column] = invalid
                elif count:
                    usable.append(column)
            item['usable_columns'] = usable
            expressions = [f'CAST("{c}" AS {_type(c)})' if c in usable else f'NULL::{_type(c)}' for c in columns]
            con.execute(f'INSERT INTO stage_{table} SELECT ' + ','.join(expressions) + f' FROM raw_{table}')

        def has(table, *columns):
            return set(columns) <= set(inventory[table]['usable_columns'])

        def disable(table, columns, reason):
            for column in columns:
                if column in inventory[table]['usable_columns']:
                    inventory[table]['usable_columns'].remove(column)
                    inventory[table]['issues'].append(f'{table}.{column}: {reason}')
                    con.execute(f'UPDATE stage_{table} SET "{column}"=NULL')

        def invalid(table, columns, predicate, reason):
            if has(table, *columns) and con.execute(f'SELECT 1 FROM stage_{table} WHERE {predicate} LIMIT 1').fetchone():
                disable(table, columns, reason)

        for table, key in KEYS.items():
            if has(table, key) and con.execute(f'SELECT 1 FROM stage_{table} GROUP BY "{key}" HAVING count(*)>1 LIMIT 1').fetchone():
                disable(table, [key] + (['flow_gbp'] if table == 'client_flows' else ['distribution_gbp'] if table == 'income_distributions' else []), 'duplicate key')
        for table in ('client_flows', 'income_distributions'):
            if has(table, 'client_id') and has('clients', 'client_id'):
                if con.execute(f'SELECT 1 FROM stage_{table} f LEFT JOIN stage_clients c USING(client_id) WHERE c.client_id IS NULL LIMIT 1').fetchone():
                    disable(table, ['client_id'], 'unknown client reference')
        invalid('clients', ['account_open_date', 'account_close_date'], 'account_close_date<account_open_date', 'invalid account lifetime')
        if has('client_flows', 'client_id', 'trade_date') and has('clients', 'client_id', 'account_open_date', 'account_close_date'):
            if con.execute('SELECT 1 FROM stage_client_flows f JOIN stage_clients c USING(client_id) WHERE trade_date<account_open_date OR trade_date>account_close_date LIMIT 1').fetchone():
                disable('client_flows', ['trade_date'], 'transaction outside account lifetime')
        invalid('client_flows', ['trade_date', 'cash_flow_date'], 'cash_flow_date<trade_date', 'cash date precedes trade date')
        invalid('client_flows', ['flow_gbp'], 'flow_gbp=0', 'zero transaction amount')
        if 'cash_flow_status' in inventory['client_flows']['supplied_columns'] and not has('client_flows', 'cash_flow_status'):
            disable('client_flows', ['flow_gbp'], 'invalid settlement status column')
        if has('client_flows', 'cash_flow_status') and con.execute("SELECT 1 FROM stage_client_flows WHERE cash_flow_status <> 'Settled' LIMIT 1").fetchone():
            disable('client_flows', ['cash_flow_status', 'flow_gbp'], 'unsupported settlement status')
        invalid('income_distributions', ['payment_date', 'ex_distribution_date'], 'payment_date<ex_distribution_date', 'payment precedes entitlement')
        invalid('income_distributions', ['distribution_gbp'], 'distribution_gbp<0', 'negative distribution')
        invalid('income_distributions', ['payment_status'], "payment_status NOT IN ('Paid','Declared')", 'unsupported payment status')
        if not any(item['usable_columns'] for item in inventory.values()):
            raise ValueError('No usable canonical columns in snapshot')
        reconciliation = {}
        for basis, datecol in [('trade', 'trade_date'), ('cash', 'cash_flow_date')]:
            targets = [f'subscriptions_{basis}_date_gbp', f'redemptions_{basis}_date_gbp',
                       'net_flow_trade_date_gbp' if basis == 'trade' else 'net_client_flow_cash_date_gbp']
            label = f'{basis}_flows'
            if has('client_flows', datecol, 'flow_gbp') and has('fund_daily', 'date', *targets):
                comparisons = ' OR '.join(f'd."{c}"<>coalesce(a.{a},0)' for c, a in zip(targets, ['subs', 'reds', 'net']))
                if basis == 'trade' and has('fund_daily', 'transaction_count'):
                    comparisons += ' OR d.transaction_count<>coalesce(a.n,0)'
                mismatch = con.execute(f'''WITH a AS (SELECT {datecol} date, sum(greatest(flow_gbp,0)) subs, sum(greatest(-flow_gbp,0)) reds, sum(flow_gbp) net,count(*) n FROM stage_client_flows GROUP BY 1)
                    SELECT 1 FROM stage_fund_daily d LEFT JOIN a USING(date) WHERE {comparisons} LIMIT 1''').fetchone()
                reconciliation[label] = 'failed' if mismatch else 'passed'
            else:
                reconciliation[label] = 'unavailable'
        for label, datecol, target, where in [('declared','ex_distribution_date','income_distributions_declared_gbp','TRUE'), ('paid','payment_date','income_distributions_paid_gbp',"payment_status='Paid'")]:
            if has('income_distributions', datecol, 'distribution_gbp', 'payment_status') and has('fund_daily', 'date', target):
                mismatch = con.execute(f'''WITH a AS (SELECT {datecol} date,sum(distribution_gbp) amount FROM stage_income_distributions WHERE {where} GROUP BY 1)
                    SELECT 1 FROM stage_fund_daily d LEFT JOIN a USING(date) WHERE d.{target}<>coalesce(a.amount,0) LIMIT 1''').fetchone()
                reconciliation[f'{label}_distributions'] = 'failed' if mismatch else 'passed'
            else:
                reconciliation[f'{label}_distributions'] = 'unavailable'
        start, end = con.execute('SELECT min(date),max(date) FROM stage_fund_daily').fetchone()
        # Valuation files may omit weekends/holidays. Reconciled cash totals establish
        # coverage across that interval only when no cash events occur on omitted dates.
        continuous = bool(start and counts['fund_daily'] == (end-start).days + 1)
        reconciled = False
        if start and has('fund_daily', 'date', 'subscriptions_cash_date_gbp',
                         'redemptions_cash_date_gbp', 'net_client_flow_cash_date_gbp') and reconciliation['cash_flows'] == 'passed':
            omitted_cash = con.execute('''SELECT 1 FROM stage_client_flows f
                LEFT JOIN stage_fund_daily d ON d.date=f.cash_flow_date
                WHERE f.cash_flow_date BETWEEN ? AND ? AND d.date IS NULL LIMIT 1''', [start, end]).fetchone()
            reconciled = omitted_cash is None
        complete = bool(start and has('fund_daily', 'date') and (continuous or reconciled))
        coverage_source = 'fund_daily' if complete else 'observed_events'
        if not start:
            start, end = con.execute('''SELECT min(d),max(d) FROM (SELECT cash_flow_date d FROM stage_client_flows UNION ALL SELECT trade_date FROM stage_client_flows UNION ALL SELECT payment_date FROM stage_income_distributions)''').fetchone()
        warnings = []
        if has('income_distributions', 'client_id', 'payment_date') and has('clients', 'client_id', 'account_close_date'):
            count = con.execute('''SELECT count(*) FROM stage_income_distributions d JOIN stage_clients c USING(client_id)
                WHERE payment_date>account_close_date''').fetchone()[0]
            if count:
                warnings.append(dict(code='payment_after_account_closure', count=count,
                    explanation='Review entitlement dates; payment after closure is not necessarily invalid.'))
        validation = dict(inventory=inventory, reconciliation=reconciliation, warnings=warnings,
            coverage=dict(complete=complete, source=coverage_source if start else 'none', start=str(start) if start else None, end=str(end) if end else None))
        for view in ('client_weekly', 'client_monthly'):
            con.execute(f'DROP VIEW IF EXISTS {view}')
        for table in ('daily_sector','calendar','client_flows','income_distributions','fund_daily','clients'):
            con.execute(f'DROP TABLE IF EXISTS {table}')
        for table in FILES:
            con.execute(f'CREATE TABLE {table} AS SELECT * FROM stage_{table}')
        if complete:
            con.execute("CREATE TABLE calendar AS SELECT CAST(range AS DATE) date FROM range(?::TIMESTAMP, ?::TIMESTAMP + INTERVAL 1 DAY, INTERVAL 1 DAY)", [start, end])
        else:
            con.execute('CREATE TABLE calendar (date DATE)')
        if complete and has('clients', 'client_id', 'client_sector') and has('client_flows', 'client_id', 'cash_flow_date', 'flow_gbp'):
            con.execute('''CREATE TABLE daily_sector AS WITH a AS (SELECT cash_flow_date date,client_sector,sum(greatest(flow_gbp,0)) subscriptions,sum(greatest(-flow_gbp,0)) redemptions,sum(flow_gbp) net_flow,count(*) transaction_count FROM client_flows JOIN clients USING(client_id) GROUP BY 1,2)
                SELECT cal.date,s.client_sector,coalesce(a.subscriptions,0) subscriptions,coalesce(a.redemptions,0) redemptions,coalesce(a.net_flow,0) net_flow,coalesce(a.transaction_count,0) transaction_count FROM calendar cal CROSS JOIN (SELECT DISTINCT client_sector FROM clients) s LEFT JOIN a ON a.date=cal.date AND a.client_sector=s.client_sector ORDER BY cal.date,s.client_sector''')
        else:
            con.execute('CREATE TABLE daily_sector (date DATE,client_sector VARCHAR,subscriptions DECIMAL(20,2),redemptions DECIMAL(20,2),net_flow DECIMAL(20,2),transaction_count BIGINT)')
        if has('client_flows', 'cash_flow_date', 'client_id', 'flow_gbp'):
            for frequency, name in [('week','client_weekly'),('month','client_monthly')]:
                con.execute(f"CREATE VIEW {name} AS SELECT date_trunc('{frequency}',cash_flow_date)::DATE period,client_id,sum(greatest(flow_gbp,0)) subscriptions,sum(greatest(-flow_gbp,0)) redemptions,sum(flow_gbp) net_flow,count(*) transaction_count FROM client_flows GROUP BY 1,2")
        run_id, elapsed = str(uuid.uuid4()), time.perf_counter()-started
        con.execute('INSERT INTO ingestion_runs VALUES (?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?)', [run_id, fingerprint, SCHEMA_VERSION, json.dumps(hashes),json.dumps(counts),json.dumps(validation),start,end,elapsed])
        con.execute('COMMIT')
        return dict(status='imported',run_id=run_id,fingerprint=fingerprint,row_counts=counts,validation=validation,duration_seconds=elapsed)
    except Exception:
        con.execute('ROLLBACK')
        raise
    finally:
        con.close()
