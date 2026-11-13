"""Dashboard contract: partial imports leave unrelated components usable."""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from fund_analytics import db

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'examples/csv_import/complete_2025'


class PartialDashboardTests(unittest.TestCase):
    def prepare(self, directory, missing=None, only=None, invalid=None):
        raw = Path(directory) / 'raw'
        shutil.copytree(FIXTURE, raw)
        for table, filename in db.FILES.items():
            path = raw / filename
            if only is not None and table not in only:
                path.unlink()
                continue
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
                columns = list(rows[0])
            if missing and missing[0] == table:
                columns.remove(missing[1])
            if invalid and invalid[0] == table:
                rows[0][invalid[1]] = 'not-a-valid-value'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
        database = Path(directory) / 'test.duckdb'
        db.ingest(raw_dir=raw, path=database)
        return database

    def check_pages(self, database, expected=None):
        with patch.object(db, 'DB_PATH', database):
            at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            if expected:
                self.assertTrue(any(expected in e.value for e in at.info), [e.value for e in at.info])
            for page in ['Timing patterns', 'Client behaviour', 'Forecasts', 'Data quality', 'Segments & stress']:
                at.sidebar.radio[0].set_value(page).run()
                self.assertFalse(at.exception, f'{page}: {[e.value for e in at.exception]}')
            return at

    def test_missing_sector_keeps_flows_aum_and_type_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self.prepare(directory, missing=('clients', 'client_sector'))
            with patch.object(db, 'DB_PATH', database):
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(any('client_sector' in e.value for e in at.info))
                labels = [m.label for m in at.metric]
                self.assertIn('Subscriptions', labels)
                self.assertIn('Closing AUM · latest selected date', labels)
                self.assertNotIn('Sectors · all when empty', [m.label for m in at.sidebar.multiselect])
                at.sidebar.radio[0].set_value('Client behaviour').run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertEqual(next(r for r in at.radio if r.label == 'Compare groups').options, ['Client type'])
                self.assertTrue(at.get('plotly_chart'))
            self.check_pages(database)

    def test_partial_sources_and_invalid_amount_render_safely(self):
        cases = [dict(only={'fund_daily'}), dict(only={'client_flows'}),
                 dict(missing=('clients', 'account_open_date')),
                 dict(missing=('clients', 'client_name')),
                 dict(only={'clients', 'client_flows', 'fund_daily'}),
                 dict(invalid=('client_flows', 'flow_gbp'))]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                self.check_pages(self.prepare(directory, **case))

    def test_clients_only_without_observation_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self.prepare(directory, only={'clients'})
            with patch.object(db, 'DB_PATH', database):
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertFalse(at.sidebar.date_input)
                self.assertTrue(any('no valid observation dates' in e.value.lower() for e in at.info))
            self.check_pages(database)

    def test_dataset_switch_clears_selected_sector(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self.prepare(directory)
            with patch.object(db, 'DB_PATH', database):
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
                sector = next(m for m in at.sidebar.multiselect if m.label == 'Sectors · all when empty')
                sector.select(sector.options[0]).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(at.session_state['filter_sectors'])
                original_identity = at.session_state['_dataset_identity']
                db.ingest(raw_dir=ROOT / 'examples/csv_import/missing_sector_2025', path=database)
                at.run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertNotEqual(original_identity, at.session_state['_dataset_identity'])
                self.assertNotIn('filter_sectors', at.session_state)
                self.assertNotIn('Sectors · all when empty', [m.label for m in at.sidebar.multiselect])
                self.assertTrue(any('client_sector' in e.value for e in at.info))
                self.assertIn('Subscriptions', [m.label for m in at.metric])
                at.sidebar.radio[0].set_value('Client behaviour').run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertEqual(next(r for r in at.radio if r.label == 'Compare groups').options, ['Client type'])

    def test_invalid_amount_preserves_transaction_count(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self.prepare(directory, invalid=('client_flows', 'flow_gbp'))
            with patch.object(db, 'DB_PATH', database):
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                metrics = {m.label: m.value for m in at.metric}
                self.assertEqual(metrics['Transactions'], '288')
                for label in ('Subscriptions', 'Redemptions', 'Net flow'):
                    self.assertNotIn(label, metrics)
                self.assertIn('Closing AUM · latest selected date', metrics)
                self.assertTrue(any('flow_gbp' in e.value for e in at.info))

    def test_transaction_only_timing_preserves_event_size(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self.prepare(directory, only={'client_flows'})
            with patch.object(db, 'DB_PATH', database):
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
                at.sidebar.radio[0].set_value('Timing patterns').run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(any('Calendar timing analysis unavailable' in e.value for e in at.info))
                plots = at.get('plotly_chart')
                self.assertEqual(len(plots), 1)
                summary = at.dataframe[0].value
                self.assertTrue(summary.average_transaction_gbp.notna().all())
                for column in ('subscriptions_per_day', 'redemptions_per_day', 'client_days', 'transactions_per_1000_client_days'):
                    self.assertTrue(summary[column].isna().all(), column)
