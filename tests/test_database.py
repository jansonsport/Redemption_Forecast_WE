"""Focused ingestion contract checks using the supplied data."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fund_analytics import db, analytics

class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'test.duckdb'
        cls.result = db.ingest(path=cls.path)
        cls.patch = patch.object(db, 'DB_PATH', cls.path)
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.temp.cleanup()

    def test_counts_and_exact_reconciliation(self):
        self.assertEqual(self.result['row_counts'], {'clients':11103,'client_flows':901088,'income_distributions':462696,'fund_daily':5053})
        self.assertTrue(all(x=='passed' for x in self.result['validation']['reconciliation'].values()))
        with db.connect(path=self.path) as con:
            self.assertEqual(con.execute('SELECT sum(net_flow) FROM daily_sector').fetchone()[0],con.execute('SELECT sum(flow_gbp) FROM client_flows WHERE cash_flow_date BETWEEN (SELECT min(date) FROM calendar) AND (SELECT max(date) FROM calendar)').fetchone()[0])

    def test_import_history(self):
        runs = db.import_runs()
        self.assertEqual(db.latest_import(), runs[-1])
        self.assertEqual(db.require_import()['fingerprint'], self.result['fingerprint'])
        self.assertEqual(runs[-1]['row_counts'], self.result['row_counts'])
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.duckdb'
            with self.assertRaisesRegex(db.DatabaseUnavailable, 'ingest'):
                db.connect(path=missing)
            with patch.object(db, 'DB_PATH', missing), self.assertRaises(db.DatabaseUnavailable):
                db.require_import()
            with db.connect(False, missing) as con:
                self.assertIsNone(db.latest_import(con))
                with self.assertRaises(LookupError):
                    db.require_import(con)

    def test_idempotent(self):
        self.assertEqual(db.ingest(path=self.path)['status'],'unchanged')

    def test_invalid_snapshot_rolls_back_and_valid_change_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for file in db.FILES.values():
                shutil.copyfile(db.RAW_DIR/file,raw/file)
            client_file = raw/db.FILES['clients']
            with client_file.open(newline='') as stream:
                rows = list(csv.reader(stream))
            with client_file.open('w',newline='') as stream:
                csv.writer(stream).writerows(rows+[rows[1]])
            with self.assertRaisesRegex(ValueError,'duplicate key'):
                db.ingest(raw_dir=raw,path=self.path)
            with db.connect(path=self.path) as con:
                self.assertEqual(con.execute('SELECT count(*) FROM clients').fetchone()[0],11103)
            rows[1][1] += ' Updated'
            with client_file.open('w',newline='') as stream:
                csv.writer(stream).writerows(rows)
            changed = db.ingest(raw_dir=raw,path=self.path)
            self.assertEqual(changed['status'],'imported')
            with db.connect(path=self.path) as con:
                self.assertTrue(con.execute('SELECT client_name FROM clients WHERE client_id=?',[rows[1][0]]).fetchone()[0].endswith(' Updated'))
            db.ingest(path=self.path)

    def test_analytics_and_correlation(self):
        frame = analytics.daily_flows(start='2026-01-01',end='2026-01-31')
        self.assertEqual(len(frame),31)
        self.assertTrue((frame['subscriptions']>=0).all())
        empty = analytics.daily_flows(start='2026-01-01',end='2026-01-31',client_ids=[])
        self.assertEqual(empty['transaction_count'].sum(),0)
        active = analytics.active_clients(start='2026-01-01',end='2026-01-31')
        self.assertEqual(str(active['active_clients'].dtype),'int64')
        for pattern in analytics.PATTERNS:
            timing = analytics.timing_summary(frame,active,pattern)
            self.assertEqual(timing['observations'].sum(),31)
        groups = analytics.group_summary('type',start='2026-01-01',end='2026-01-31')
        self.assertEqual(groups['transaction_count'].sum(),frame['transaction_count'].sum())
        with self.assertRaises(ValueError):
            analytics.group_summary('client_name')
        corr, overlap, rows = analytics.correlations(['C000001','C000002'])
        self.assertEqual(corr.shape,(2,2))
        self.assertTrue((overlap.values>=0).all())
        self.assertEqual(list(rows.columns),['period','client_id','amount'])
        with self.assertRaises(ValueError):
            analytics.correlations([])
        tables = analytics.load_tables(['fund_daily'])
        self.assertTrue(tables['fund_daily'].date.is_monotonic_increasing)

if __name__ == '__main__':
    unittest.main()
