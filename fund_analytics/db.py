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
import uuid
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / 'data/clean'
DB_PATH = Path(os.environ.get('FUND_ANALYTICS_DB') or CLEAN_DIR / 'fund_analytics.duckdb')
RAW_DIR = ROOT / 'data/raw'
SCHEMA_VERSION = '1'
FILES = {'clients': 'synthetic_clients.csv', 'client_flows': 'synthetic_client_flows.csv', 'income_distributions': 'synthetic_income_distributions.csv', 'fund_daily': 'synthetic_fund_daily.csv'}
KEYS = {'clients': 'client_id', 'client_flows': 'transaction_id', 'income_distributions': 'distribution_id', 'fund_daily': 'date'}
EXPECTED = {
'clients': 'client_id,client_name,client_sector,client_type,country,region,unit_type,account_open_date,account_close_date,valuation_date,client_aum_gbp,opening_valuation_date,opening_client_aum_gbp',
'client_flows': 'transaction_id,trade_date,cash_flow_date,cash_flow_status,client_id,flow_gbp,client_aum_after_trade_gbp',
'income_distributions': 'distribution_id,ex_distribution_date,payment_date,client_id,distribution_gbp,payment_status',
'fund_daily': 'date,opening_aum_gbp,subscriptions_trade_date_gbp,redemptions_trade_date_gbp,net_flow_trade_date_gbp,investment_value_change_gbp,income_distributions_declared_gbp,closing_aum_gbp,active_client_count,transaction_count,income_unit_price_gbp,accumulation_unit_price_gbp,subscriptions_cash_date_gbp,redemptions_cash_date_gbp,net_client_flow_cash_date_gbp,income_distributions_paid_gbp,net_investor_cash_flow_gbp,redemptions_awaiting_payment_gbp,income_distributions_payable_gbp'}

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

def ingest(raw_dir=RAW_DIR, path=None):
    """Import one snapshot transactionally; unchanged content is a no-op. Needs the dashboard stopped."""
    started = time.perf_counter()
    raw_dir, path = Path(raw_dir), Path(path or DB_PATH)
    hashes = {}
    for table, filename in FILES.items():
        source = raw_dir / filename
        with source.open(newline='') as stream:
            header = next(csv.reader(stream))
        if header != EXPECTED[table].split(','):
            raise ValueError(f'{filename}: unexpected columns {header}')
        with source.open('rb') as stream:
            hashes[filename] = hashlib.file_digest(stream, 'sha256').hexdigest()
    fingerprint = hashlib.sha256(json.dumps([SCHEMA_VERSION, hashes], sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(False, path)
    try:
        con.execute('CREATE TABLE IF NOT EXISTS ingestion_runs (run_id VARCHAR PRIMARY KEY, fingerprint VARCHAR, imported_at TIMESTAMP, schema_version VARCHAR, source_hashes JSON, row_counts JSON, validation JSON, coverage_start DATE, coverage_end DATE, duration_seconds DOUBLE)')
        previous = latest_import(con)
        if previous and previous['fingerprint'] == fingerprint:
            return {'status': 'unchanged', 'fingerprint': fingerprint}
        con.execute('BEGIN TRANSACTION')
        counts = {}
        for table, filename in FILES.items():
            columns = EXPECTED[table].split(',')
            definitions = ', '.join(f'"{c}" {_type(c)}' for c in columns)
            con.execute(f'CREATE TEMP TABLE stage_{table} ({definitions})')
            source = str(raw_dir / filename).replace("'", "''")
            con.execute(f"COPY stage_{table} FROM '{source}' (HEADER, DELIMITER ',', NULLSTR '')")
            key = KEYS[table]
            _require(con, f'SELECT {key},count(*) FROM stage_{table} GROUP BY {key} HAVING count(*)>1', f'{filename}: duplicate key')
            required = [c for c in columns if c != 'account_close_date']
            _require(con, f'SELECT {key} FROM stage_{table} WHERE ' + ' OR '.join(f'{c} IS NULL' for c in required), f'{filename}: missing required field')
            counts[table] = con.execute(f'SELECT count(*) FROM stage_{table}').fetchone()[0]
            if not counts[table]:
                raise ValueError(f'{filename}: empty source')
        for table in ('client_flows', 'income_distributions'):
            _require(con, f'SELECT f.{KEYS[table]}, f.client_id FROM stage_{table} f LEFT JOIN stage_clients c USING(client_id) WHERE c.client_id IS NULL', f'{table}: unknown client')
        _require(con, 'SELECT client_id FROM stage_clients WHERE account_close_date < account_open_date', 'clients: invalid lifetime')
        _require(con, 'SELECT transaction_id FROM stage_client_flows f JOIN stage_clients c USING(client_id) WHERE cash_flow_date<trade_date OR trade_date<account_open_date OR trade_date>account_close_date OR flow_gbp=0', 'flows: invalid date or zero amount')
        _require(con, "SELECT distribution_id FROM stage_income_distributions WHERE payment_date<ex_distribution_date OR distribution_gbp<0 OR payment_status NOT IN ('Paid','Declared')", 'distributions: invalid date, amount or status')
        _require(con, "SELECT transaction_id FROM stage_client_flows WHERE cash_flow_status <> 'Settled'", 'flows: unsupported settlement status')
        reconciliation = {}
        for basis, col in [('trade', 'trade_date'), ('cash', 'cash_flow_date')]:
            net_col = 'net_flow_trade_date_gbp' if basis == 'trade' else 'net_client_flow_cash_date_gbp'
            sql = f'''WITH a AS (SELECT {col} date, sum(greatest(flow_gbp,0)) subs, sum(greatest(-flow_gbp,0)) reds, sum(flow_gbp) net, count(*) n FROM stage_client_flows GROUP BY 1)
            SELECT d.date FROM stage_fund_daily d LEFT JOIN a USING(date) WHERE d.subscriptions_{basis}_date_gbp<>coalesce(a.subs,0) OR d.redemptions_{basis}_date_gbp<>coalesce(a.reds,0) OR d.{net_col}<>coalesce(a.net,0)'''
            if basis == 'trade':
                sql += ' OR d.transaction_count<>coalesce(a.n,0)'
            _require(con, sql, f'{basis} flow reconciliation failed')
            reconciliation[f'{basis}_flows'] = 'passed'
        for label, datecol, target, where in [('declared','ex_distribution_date','income_distributions_declared_gbp','TRUE'), ('paid','payment_date','income_distributions_paid_gbp',"payment_status='Paid'")]:
            _require(con, f'''WITH a AS (SELECT {datecol} date, sum(distribution_gbp) amount FROM stage_income_distributions WHERE {where} GROUP BY 1) SELECT d.date FROM stage_fund_daily d LEFT JOIN a USING(date) WHERE d.{target}<>coalesce(a.amount,0)''', f'{label} distributions reconciliation failed')
            reconciliation[f'{label}_distributions'] = 'passed'
        warning_count = con.execute('SELECT count(*) FROM stage_income_distributions d JOIN stage_clients c USING(client_id) WHERE payment_date>account_close_date').fetchone()[0]
        validation = {'reconciliation': reconciliation, 'warnings': [{'code': 'payment_after_account_closure', 'count': warning_count, 'explanation': 'Review entitlement dates; payment after closure is not necessarily invalid.'}] if warning_count else []}
        for view in ('client_weekly', 'client_monthly'):
            con.execute(f'DROP VIEW IF EXISTS {view}')
        for table in ('daily_sector','calendar','client_flows','income_distributions','fund_daily','clients'):
            con.execute(f'DROP TABLE IF EXISTS {table}')
        for table in FILES:
            cols = EXPECTED[table].split(',')
            definitions = ', '.join(f'"{c}" {_type(c)}' + ('' if c == 'account_close_date' else ' NOT NULL') for c in cols)
            constraints = f', PRIMARY KEY ({KEYS[table]})'
            if table in ('client_flows','income_distributions'):
                constraints += ', FOREIGN KEY (client_id) REFERENCES clients(client_id)'
            con.execute(f'CREATE TABLE {table} ({definitions}{constraints})')
            con.execute(f'INSERT INTO {table} SELECT * FROM stage_{table}')
        start, end = con.execute('SELECT min(date), max(date) FROM fund_daily').fetchone()
        con.execute("CREATE TABLE calendar AS SELECT CAST(range AS DATE) date FROM range(?::TIMESTAMP, ?::TIMESTAMP + INTERVAL 1 DAY, INTERVAL 1 DAY)", [start, end])
        con.execute('''CREATE TABLE daily_sector AS WITH a AS (SELECT cash_flow_date date, client_sector, sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, sum(flow_gbp) net_flow, count(*) transaction_count FROM client_flows JOIN clients USING(client_id) GROUP BY 1,2) SELECT cal.date, s.client_sector, coalesce(a.subscriptions,0) subscriptions, coalesce(a.redemptions,0) redemptions, coalesce(a.net_flow,0) net_flow, coalesce(a.transaction_count,0) transaction_count FROM calendar cal CROSS JOIN (SELECT DISTINCT client_sector FROM clients) s LEFT JOIN a ON a.date=cal.date AND a.client_sector=s.client_sector ORDER BY cal.date,s.client_sector''')
        for frequency, name in [('week', 'client_weekly'), ('month', 'client_monthly')]:
            con.execute(f"CREATE VIEW {name} AS SELECT date_trunc('{frequency}', cash_flow_date)::DATE period, client_id, sum(greatest(flow_gbp,0)) subscriptions, sum(greatest(-flow_gbp,0)) redemptions, sum(flow_gbp) net_flow, count(*) transaction_count FROM client_flows GROUP BY 1,2")
        run_id = str(uuid.uuid4())
        elapsed = time.perf_counter() - started
        con.execute('INSERT INTO ingestion_runs VALUES (?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?)', [run_id, fingerprint, SCHEMA_VERSION, json.dumps(hashes), json.dumps(counts), json.dumps(validation), start, end, elapsed])
        con.execute('COMMIT')
        return {'status': 'imported', 'run_id': run_id, 'fingerprint': fingerprint, 'row_counts': counts, 'validation': validation, 'duration_seconds': elapsed}
    except Exception:
        try:
            con.execute('ROLLBACK')
        except duckdb.Error:
            pass
        raise
    finally:
        con.close()
