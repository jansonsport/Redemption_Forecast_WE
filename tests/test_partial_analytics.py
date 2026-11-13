"""Dependency isolation: incomplete imports must not corrupt supported analytics."""
import csv
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from fund_analytics import analytics, db

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'csv_import'


def copy_sources(tmp_path, variant='complete_2025', tables=None):
    directory = tmp_path / 'sources'
    directory.mkdir()
    for table in tables or db.FILES:
        filename = db.FILES[table]
        shutil.copyfile(EXAMPLES / variant / filename, directory / filename)
    return directory


def edit_csv(directory, table, *, remove=(), invalid=None):
    path = directory / db.FILES[table]
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    columns = [column for column in columns if column not in remove]
    if invalid:
        column, value = invalid
        rows[0][column] = value
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def import_sources(directory, tmp_path):
    path = tmp_path / 'partial.duckdb'
    db.ingest(raw_dir=directory, path=path)
    return db.connect(path=path)


def test_missing_sector_preserves_unrelated_analytics(tmp_path):
    complete_path, partial_path = tmp_path / 'complete.duckdb', tmp_path / 'partial.duckdb'
    db.ingest(raw_dir=EXAMPLES / 'complete_2025', path=complete_path)
    db.ingest(raw_dir=EXAMPLES / 'missing_sector_2025', path=partial_path)
    with db.connect(path=complete_path) as complete, db.connect(path=partial_path) as partial:
        pd.testing.assert_frame_equal(analytics.daily_flows(con=complete), analytics.daily_flows(con=partial))
        pd.testing.assert_frame_equal(analytics.group_summary('type', con=complete), analytics.group_summary('type', con=partial))
        pd.testing.assert_frame_equal(analytics.fund_aum(con=complete), analytics.fund_aum(con=partial))
        assert analytics.distinct_values('client_type', con=partial)
        assert len(analytics.client_list(con=partial)) == 12
        assert analytics.capabilities(partial)['type']['available']
        assert not analytics.capabilities(partial)['sector']['available']
        for operation in (lambda: analytics.require_feature('sector', partial),
                          lambda: analytics.distinct_values('client_sector', con=partial),
                          lambda: analytics.group_summary('sector', con=partial),
                          lambda: analytics.daily_flows(sectors=['Education'], con=partial)):
            with pytest.raises(db.FeatureUnavailable, match=r'clients\.client_sector'):
                operation()
        inventory = analytics.inventory(partial)['clients']
        assert 'client_sector' not in inventory['supplied_columns']
        assert 'client_sector' not in inventory['usable_columns']


def test_missing_account_dates_disables_exposure_only(tmp_path):
    directory = copy_sources(tmp_path)
    edit_csv(directory, 'clients', remove=('account_open_date', 'account_close_date'))
    with import_sources(directory, tmp_path) as con:
        caps = analytics.capabilities(con)
        assert not caps['exposure']['available']
        assert caps['flow_cash']['available']
        daily = analytics.daily_flows(con=con)
        assert len(daily) == 365
        assert daily.transaction_count.sum() == 288
        assert daily.subscriptions.sum() > 0
        with pytest.raises(db.FeatureUnavailable, match='account_open_date'):
            analytics.active_clients(con=con)
        summary = analytics.timing_summary(daily)
        assert summary.transactions_per_1000_client_days.isna().all()
        assert summary.average_transaction_gbp.notna().all()


def test_transactions_only_keeps_observed_dates(tmp_path):
    directory = copy_sources(tmp_path, tables=['client_flows'])
    with import_sources(directory, tmp_path) as con:
        daily = analytics.daily_flows(con=con)
        with (directory / db.FILES['client_flows']).open(newline='') as stream:
            observed = {row['cash_flow_date'] for row in csv.DictReader(stream)}
        assert set(daily.date.dt.strftime('%Y-%m-%d')) == observed
        assert len(daily) < 365
        assert (daily.transaction_count > 0).all()
        assert daily.transaction_count.sum() == 288
        for feature in ('timing', 'correlations', 'forecast', 'exposure'):
            assert not analytics.capabilities(con)[feature]['available']
            with pytest.raises(db.FeatureUnavailable):
                analytics.require_feature(feature, con)
        with pytest.raises(db.FeatureUnavailable):
            analytics.correlations(['T001', 'T002'], con=con)


def test_invalid_amount_preserves_counts_without_partial_totals(tmp_path):
    directory = copy_sources(tmp_path)
    edit_csv(directory, 'client_flows', invalid=('flow_gbp', 'not-a-number'))
    with import_sources(directory, tmp_path) as con:
        caps = analytics.capabilities(con)
        assert not caps['flow_cash']['available']
        assert caps['counts_cash']['available']
        daily = analytics.daily_flows(con=con)
        assert daily.transaction_count.sum() == 288
        assert daily[['subscriptions', 'redemptions', 'net_flow']].isna().all().all()
        inventory = analytics.inventory(con)['client_flows']
        assert 'flow_gbp' in inventory['supplied_columns']
        assert 'flow_gbp' not in inventory['usable_columns']
        assert any('flow_gbp' in str(issue) for issue in inventory['issues'])
        assert analytics.fund_aum(con=con).closing_aum_gbp.notna().all()
        with pytest.raises(db.FeatureUnavailable, match='flow_gbp'):
            analytics.group_summary('type', con=con)


def test_fund_only_aum_remains_available(tmp_path):
    directory = copy_sources(tmp_path, tables=['fund_daily'])
    with import_sources(directory, tmp_path) as con:
        assert analytics.capabilities(con)['aum']['available']
        assert not analytics.capabilities(con)['flow_cash']['available']
        frame = analytics.fund_aum(con=con)
        assert len(frame) == 365
        assert frame.closing_aum_gbp.notna().all()
        counts = analytics.table_counts(con=con).set_index('dataset')['rows'].to_dict()
        assert counts['Daily fund'] == 365
        assert counts['Clients'] == 0
        with pytest.raises(db.FeatureUnavailable):
            analytics.daily_flows(con=con)


def test_partial_import_does_not_modify_sources_or_main_database(tmp_path):
    directory = copy_sources(tmp_path, 'missing_sector_2025')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    main_before = hashlib.sha256(db.DB_PATH.read_bytes()).hexdigest() if db.DB_PATH.exists() else None
    with import_sources(directory, tmp_path) as con:
        assert analytics.inventory(con)['clients']['row_count'] == 12
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()} == before
    assert set(p.name for p in directory.iterdir()) == set(db.FILES.values())
    if main_before:
        assert hashlib.sha256(db.DB_PATH.read_bytes()).hexdigest() == main_before


MISSING_COLUMNS = [(table, column) for table, columns in db.EXPECTED.items()
                   for column in columns.split(',')]


@pytest.mark.parametrize('table,column', MISSING_COLUMNS,
                         ids=[f'{table}.{column}' for table, column in MISSING_COLUMNS])
def test_every_original_column_can_be_missing_without_breaking_supported_queries(tmp_path, table, column):
    """Exercise advertised query capabilities for every one-column omission."""
    directory = copy_sources(tmp_path)
    edit_csv(directory, table, remove=(column,))
    with import_sources(directory, tmp_path) as con:
        inv = analytics.inventory(con)
        assert column not in inv[table]['supplied_columns']
        assert column not in inv[table]['usable_columns']
        caps = analytics.capabilities(con)
        assert not caps['forecast']['available']
        assert not caps['segments']['available']
        assert len(analytics.table_counts(con)) == 4
        assert len(analytics.load_tables(list(db.KEYS), con=con)) == 4
        analytics.coverage(con)
        if caps['clients']['available']:
            assert len(analytics.client_list(con=con)) == 12
        if caps['aum']['available']:
            assert len(analytics.fund_aum(con=con)) == 365
        if caps['distributions']['available']:
            analytics.scheduled_distributions(after='2025-01-01', con=con)
        for feature, attribute in [('sector', 'client_sector'), ('type', 'client_type')]:
            if caps[feature]['available']:
                assert analytics.distinct_values(attribute, con=con)
        active = analytics.active_clients(con=con) if caps['exposure']['available'] else None
        for basis in ('cash', 'trade'):
            amounts = caps[f'flow_{basis}']['available']
            counts = caps[f'counts_{basis}']['available']
            if amounts or counts:
                daily = analytics.daily_flows(basis=basis, con=con)
                assert len(daily) > 0
                if amounts:
                    assert daily.subscriptions.notna().all()
                else:
                    assert daily.subscriptions.isna().all()
                if counts:
                    assert daily.transaction_count.sum() == 288
                else:
                    assert daily.transaction_count.isna().all()
                if caps['timing']['available']:
                    analytics.timing_summary(daily, active)
                if caps['clients']['available']:
                    analytics.daily_flows(client_ids=['T001'], basis=basis, con=con)
                for feature in ('sector', 'type'):
                    if caps[feature]['available']:
                        if amounts:
                            analytics.group_summary(feature, basis=basis, con=con)
                        values = analytics.distinct_values('client_sector' if feature == 'sector' else 'client_type', con=con)
                        filters = {'sectors' if feature == 'sector' else 'client_types': values[:1]}
                        analytics.daily_flows(basis=basis, con=con, **filters)
            if amounts and caps['concentration']['available']:
                analytics.client_concentration(basis=basis, con=con)
            if amounts and caps['correlations']['available']:
                analytics.correlations(['T001', 'T002'], basis=basis, con=con)


def test_legacy_complete_manifest_without_inventory_uses_schema(tmp_path):
    """Version-1 complete imports remain readable without regenerating the database."""
    import json

    path = tmp_path / 'legacy.duckdb'
    db.ingest(raw_dir=EXAMPLES / 'complete_2025', path=path)
    with db.connect(read_only=False, path=path) as con:
        expected = analytics.capabilities(con)
        manifest = db.latest_import(con)
        validation = manifest['validation']
        validation.pop('inventory', None)
        validation.pop('coverage', None)
        con.execute('UPDATE ingestion_runs SET schema_version=?, validation=? WHERE run_id=?',
                    ['1', json.dumps(validation), manifest['run_id']])
        assert analytics.capabilities(con) == expected
        for table, columns in db.EXPECTED.items():
            assert analytics.inventory(con)[table]['usable_columns'] == columns.split(',')
        assert len(analytics.daily_flows(con=con)) == 365
        assert len(analytics.group_summary('sector', con=con)) == 3
        assert len(analytics.fund_aum(con=con)) == 365


def test_missing_transaction_id_keeps_concentration_amounts(tmp_path):
    directory = copy_sources(tmp_path)
    edit_csv(directory, 'client_flows', remove=('transaction_id',))
    with import_sources(directory, tmp_path) as con:
        assert analytics.capabilities(con)['concentration']['available']
        result = analytics.client_concentration(con=con)
        assert len(result) == 12
        assert result.gross_flow_gbp.notna().all()
        assert result.transactions.isna().all()
        assert result.average_transaction_gbp.isna().all()
