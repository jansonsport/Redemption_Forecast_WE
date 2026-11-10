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

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_counts_and_exact_reconciliation(self):
        self.assertEqual(self.result['row_counts'], {'clients':11103,'client_flows':901088,'income_distributions':462696,'fund_daily':5053})
        self.assertTrue(all(x=='passed' for x in self.result['validation']['reconciliation'].values()))
        with db.connect(path=self.path) as con:
            self.assertEqual(con.execute('SELECT sum(net_flow) FROM daily_sector').fetchone()[0],con.execute('SELECT sum(flow_gbp) FROM client_flows WHERE cash_flow_date BETWEEN (SELECT min(date) FROM calendar) AND (SELECT max(date) FROM calendar)').fetchone()[0])

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
        with patch.object(analytics,'connect',lambda: db.connect(path=self.path)):
            frame = analytics.daily_flows(start='2026-01-01',end='2026-01-31')
            self.assertEqual(len(frame),31)
            self.assertTrue((frame['subscriptions']>=0).all())
            empty = analytics.daily_flows(start='2026-01-01',end='2026-01-31',client_ids=[])
            self.assertEqual(empty['transaction_count'].sum(),0)
            timing = analytics.timing_patterns(start='2026-01-01',end='2026-01-31')
            self.assertEqual(timing['observed_days'].sum(),31)
            corr, overlap, rows = analytics.correlations(['C000001','C000002'])
            self.assertEqual(corr.shape,(2,2))
            self.assertTrue((overlap.values>=0).all())
            with self.assertRaises(ValueError):
                analytics.correlations([])

if __name__ == '__main__':
    unittest.main()
