"""Small replacement snapshots exercise discovery and field-level validation."""
import csv
import tempfile
from pathlib import Path

import pytest

from fund_analytics import db


@pytest.fixture
def snapshot():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        yield root, root / 'snapshot.duckdb'


def write(root, name, header, rows, preamble=()):
    with (root / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerows(preamble)
        writer.writerow(header)
        writer.writerows(rows)


def inventory(result, table='client_flows'):
    return result['validation']['inventory'][table]


def test_header_detection_normalisation_extra_and_quoted_values(snapshot):
    root, path = snapshot
    write(root, 'events.csv', [' Flow_GBP ', 'Cash_flow_date', 'transaction_id', 'note'],
          [['123.45', '2025-01-01', '1', 'quoted, note']], [('Report',), ()])
    result = db.ingest(root, path)
    assert inventory(result)['header_row'] == 3
    assert inventory(result)['extra_columns'] == ['note']
    assert set(inventory(result)['usable_columns']) == {'flow_gbp', 'cash_flow_date', 'transaction_id'}
    with db.connect(path=path) as con:
        assert float(con.execute('SELECT flow_gbp FROM client_flows').fetchone()[0]) == 123.45
        assert con.execute('SELECT count(*) FROM clients').fetchone()[0] == 0
        assert con.execute('SELECT count(*) FROM calendar').fetchone()[0] == 0
    assert result['validation']['coverage']['complete'] is False
    assert db.ingest(root, path)['status'] == 'unchanged'


def test_scan_boundary_and_override_fingerprint(snapshot):
    root, path = snapshot
    write(root, db.FILES['client_flows'], ['flow_gbp', 'cash_flow_date'],
          [['100', '2025-01-01']], [('Report',), ()])
    with pytest.raises(ValueError, match='header'):
        db.ingest(root, path, scan_rows=2)
    first = db.ingest(root, path, scan_rows=3)
    second = db.ingest(root, path, scan_rows=2, header_rows={'client_flows': 3})
    assert first['fingerprint'] != second['fingerprint']
    assert second['status'] == 'imported'


def test_ambiguous_header_and_source_override(snapshot):
    root, path = snapshot
    write(root, db.FILES['client_flows'], ['flow_gbp', 'cash_flow_date'],
          [['flow_gbp', 'cash_flow_date'], ['100', '2025-01-01']])
    with pytest.raises(ValueError, match='found 2'):
        db.ingest(root, path)
    result = db.ingest(root, path, header_rows={'client_flows': 2})
    assert inventory(result)['row_count'] == 1


def test_invalid_column_is_null_for_every_row(snapshot):
    root, path = snapshot
    write(root, db.FILES['client_flows'], ['transaction_id', 'flow_gbp', 'cash_flow_date'],
          [[1, '100', '2025-01-01'], [2, 'broken', '2025-01-02']])
    result = db.ingest(root, path)
    assert 'flow_gbp' not in inventory(result)['usable_columns']
    assert inventory(result)['invalid_counts']['flow_gbp'] == 1
    assert 'cash_flow_date' in inventory(result)['usable_columns']
    with db.connect(path=path) as con:
        assert con.execute('SELECT count(*), count(flow_gbp) FROM client_flows').fetchone() == (2, 0)


def test_duplicate_ids_disable_amounts_orphans_disable_join_only(snapshot):
    root, path = snapshot
    write(root, db.FILES['clients'], ['client_id', 'client_type'], [['C1', 'Charity']])
    write(root, db.FILES['client_flows'], ['transaction_id', 'flow_gbp', 'cash_flow_date', 'client_id'],
          [[1, '100', '2025-01-01', 'unknown'], [1, '-20', '2025-01-02', 'unknown']])
    result = db.ingest(root, path)
    item = inventory(result)
    assert set(item['usable_columns']) == {'cash_flow_date'}
    assert any('duplicate key' in reason for reason in item['issues'])
    assert any('unknown client' in reason for reason in item['issues'])


def test_replacement_removes_prior_data_and_failed_import_rolls_back(snapshot):
    root, path = snapshot
    write(root, db.FILES['clients'], ['client_id', 'client_sector'], [['C1', 'Health']])
    write(root, db.FILES['client_flows'], ['transaction_id', 'flow_gbp', 'cash_flow_date'], [[1, '100', '2025-01-01']])
    first = db.ingest(root, path)
    (root / db.FILES['clients']).unlink()
    second = db.ingest(root, path)
    assert first['fingerprint'] != second['fingerprint']
    with db.connect(path=path) as con:
        assert con.execute('SELECT count(*) FROM clients').fetchone()[0] == 0
    with (root / db.FILES['client_flows']).open('a') as stream:
        stream.write('footer\n')
    with pytest.raises(ValueError, match='malformed'):
        db.ingest(root, path)
    with db.connect(path=path) as con:
        assert db.latest_import(con)['fingerprint'] == second['fingerprint']
        assert con.execute('SELECT count(*) FROM client_flows').fetchone()[0] == 1


def test_complete_daily_coverage_and_reconciliation_failure(snapshot):
    root, path = snapshot
    write(root, db.FILES['client_flows'], ['flow_gbp', 'cash_flow_date'], [['100', '2025-01-01']])
    write(root, db.FILES['fund_daily'], ['date', 'subscriptions_cash_date_gbp', 'redemptions_cash_date_gbp', 'net_client_flow_cash_date_gbp'],
          [['2025-01-01', '101', '0', '101'], ['2025-01-02', '0', '0', '0']])
    result = db.ingest(root, path)
    assert result['validation']['coverage']['complete'] is True
    assert result['validation']['reconciliation']['cash_flows'] == 'failed'
    with db.connect(path=path) as con:
        assert con.execute('SELECT count(*) FROM calendar').fetchone()[0] == 2


def test_bom_and_duplicate_normalised_headers(snapshot):
    root, path = snapshot
    (root / db.FILES['clients']).write_text('\ufeffCLIENT_ID,client_name\nC1,"A, Charity"\n', encoding='utf-8')
    result = db.ingest(root, path)
    assert inventory(result, 'clients')['usable_columns'] == ['client_id', 'client_name']
    write(root, db.FILES['clients'], ['client_id', ' CLIENT_ID '], [['C1', 'C1']])
    with pytest.raises(ValueError, match='duplicate'):
        db.ingest(root, path)


def test_integer_values_do_not_round_and_explicit_source_wins(snapshot):
    root, path = snapshot
    write(root, db.FILES['client_flows'], ['transaction_id', 'flow_gbp'], [[1, '999']])
    outside = root / 'chosen.csv'
    write(root, outside.name, ['transaction_id', 'flow_gbp', 'cash_flow_date'], [['1.5', '25', '2025-01-01']])
    result = db.ingest(root, path, sources={'client_flows': outside})
    assert inventory(result)['source'] == str(outside)
    assert 'transaction_id' not in inventory(result)['usable_columns']
    with db.connect(path=path) as con:
        assert float(con.execute('SELECT sum(flow_gbp) FROM client_flows').fetchone()[0]) == 25


def test_cli_flags_import_and_bad_override(snapshot):
    import os
    import subprocess
    import sys
    root, path = snapshot
    write(root, 'events.csv', ['flow_gbp', 'cash_flow_date'], [['10', '2025-01-01']], [('Title',)])
    env = dict(os.environ, FUND_ANALYTICS_DB=str(path))
    command = [sys.executable, '-m', 'fund_analytics', 'ingest', '--raw-dir', str(root),
               '--scan-rows', '1', '--source', f'client_flows={root / "events.csv"}', '--header-row', 'client_flows=2']
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"status": "imported"' in result.stdout
    bad = subprocess.run(command + ['--header-row', 'client_flows=3'], env=env, text=True, capture_output=True)
    assert bad.returncode != 0
    assert 'Duplicate override' in bad.stderr


def test_valuation_weekends_confirmed_only_by_reconciled_cash(snapshot):
    root, path = snapshot
    header = ['date', 'subscriptions_cash_date_gbp', 'redemptions_cash_date_gbp', 'net_client_flow_cash_date_gbp']
    write(root, db.FILES['fund_daily'], header,
          [['2025-01-03', '100', '0', '100'], ['2025-01-06', '50', '0', '50']])
    write(root, db.FILES['client_flows'], ['flow_gbp', 'cash_flow_date'],
          [['100', '2025-01-03'], ['50', '2025-01-06']])
    result = db.ingest(root, path)
    assert result['validation']['coverage']['complete'] is True
    with db.connect(path=path) as con:
        assert con.execute('SELECT count(*) FROM calendar').fetchone()[0] == 4
    write(root, db.FILES['client_flows'], ['flow_gbp', 'cash_flow_date'],
          [['100', '2025-01-03'], ['50', '2025-01-06'], ['10', '2025-01-04']])
    assert db.ingest(root, path)['validation']['coverage']['complete'] is False
    write(root, db.FILES['fund_daily'], ['date', 'closing_aum_gbp'],
          [['2025-01-03', '1000'], ['2025-01-06', '1050']])
    assert db.ingest(root, path)['validation']['coverage']['complete'] is False


def test_payment_after_account_closure_is_warning(snapshot):
    root, path = snapshot
    write(root, db.FILES['clients'], ['client_id', 'account_close_date'], [['C1', '2025-01-01']])
    write(root, db.FILES['income_distributions'], ['distribution_id', 'client_id', 'payment_date', 'distribution_gbp'],
          [[1, 'C1', '2025-01-02', '10']])
    result = db.ingest(root, path)
    assert result['validation']['warnings'][0]['code'] == 'payment_after_account_closure'
    assert result['validation']['warnings'][0]['count'] == 1


def test_header_override_requires_a_source(snapshot):
    root, path = snapshot
    write(root, db.FILES['clients'], ['client_id'], [['C1']])
    with pytest.raises(ValueError, match='no source'):
        db.ingest(root, path, header_rows={'client_flows': 1})
