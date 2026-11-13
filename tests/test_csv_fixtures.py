"""Small deterministic manual-import examples, independent of expensive model fits."""
import csv
import importlib.util
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from fund_analytics import db

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'csv_import'
spec = importlib.util.spec_from_file_location('csv_fixture_generator', EXAMPLES / 'generate.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def read(directory, table):
    with (directory / db.FILES[table]).open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_fixture_exactly_one_missing_column():
    for table in db.FILES:
        complete_header, complete = read(EXAMPLES / 'complete_2025', table)
        missing_header, missing = read(EXAMPLES / 'missing_sector_2025', table)
        assert complete_header == db.EXPECTED[table].split(',')
        if table == 'clients':
            assert missing_header == [c for c in complete_header if c != 'client_sector']
            assert missing == [{k: v for k, v in r.items() if k != 'client_sector'} for r in complete]
        else:
            assert complete == missing
            assert complete_header == missing_header


def test_fixture_regeneration_is_identical(tmp_path):
    generator.generate(tmp_path)
    for original in EXAMPLES.glob('*_2025/*.csv'):
        assert (tmp_path / original.relative_to(EXAMPLES)).read_bytes() == original.read_bytes()


def test_fixture_one_year_reconciles_exactly():
    tables = generator.records()
    assert len(tables['clients']) == 12
    assert len(tables['client_flows']) == 288
    assert len(tables['income_distributions']) == 48
    assert [r['date'] for r in tables['fund_daily']] == [
        (date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    client_ids = {r['client_id'] for r in tables['clients']}
    flows, distributions = defaultdict(list), defaultdict(list)
    for row in tables['client_flows']:
        assert row['client_id'] in client_ids
        assert row['trade_date'] == row['cash_flow_date']
        flows[row['cash_flow_date']].append(Decimal(row['flow_gbp']))
    for row in tables['income_distributions']:
        assert row['client_id'] in client_ids
        assert row['ex_distribution_date'] == row['payment_date']
        distributions[row['payment_date']].append(Decimal(row['distribution_gbp']))
    previous_close = sum(Decimal(r['opening_client_aum_gbp']) for r in tables['clients'])
    for row in tables['fund_daily']:
        day = row['date']
        subs = sum((x for x in flows[day] if x > 0), Decimal(0))
        reds = sum((-x for x in flows[day] if x < 0), Decimal(0))
        paid = sum(distributions[day], Decimal(0))
        assert Decimal(row['opening_aum_gbp']) == previous_close
        assert Decimal(row['closing_aum_gbp']) == previous_close + subs - reds - paid
        assert row['transaction_count'] == len(flows[day])
        for basis in ('cash', 'trade'):
            assert Decimal(row[f'subscriptions_{basis}_date_gbp']) == subs
            assert Decimal(row[f'redemptions_{basis}_date_gbp']) == reds
        assert Decimal(row['net_client_flow_cash_date_gbp']) == subs - reds
        assert Decimal(row['net_flow_trade_date_gbp']) == subs - reds
        assert Decimal(row['income_distributions_declared_gbp']) == paid
        assert Decimal(row['income_distributions_paid_gbp']) == paid
        previous_close = Decimal(row['closing_aum_gbp'])
    assert previous_close == sum(Decimal(r['client_aum_gbp']) for r in tables['clients'])


def test_complete_fixture_passes_database_reconciliation(tmp_path):
    result = db.ingest(raw_dir=EXAMPLES / 'complete_2025', path=tmp_path / 'fixture.duckdb')
    assert result['row_counts'] == dict(clients=12, client_flows=288, income_distributions=48, fund_daily=365)
    assert all(value == 'passed' for value in result['validation']['reconciliation'].values())
