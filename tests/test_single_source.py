"""Guard: only fund_analytics/db.py knows the raw CSVs and the database file; everything else reads the database."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r'data/raw|\bRAW_DIR\b|\bdb\.FILES\b|import[^\n]*\bFILES\b|synthetic_\w+\.csv|\.duckdb\b')


class SingleSourceTests(unittest.TestCase):
    def test_raw_files_and_database_path_only_in_db(self):
        files = sorted((ROOT / 'fund_analytics').rglob('*.py')) + [ROOT / 'app.py', ROOT / 'segments_page.py']
        self.assertGreater(len(files), 10)
        hits = [f'{path.relative_to(ROOT)}:{number}: {line.strip()}'
                for path in files if path != ROOT / 'fund_analytics/db.py'
                for number, line in enumerate(path.read_text().splitlines(), 1) if FORBIDDEN.search(line)]
        self.assertEqual(hits, [])

    def test_shared_queries_only_in_data_layer(self):
        shared = {ROOT / 'fund_analytics' / name for name in ('db.py', 'analytics.py', 'benchmark.py')}
        files = sorted((ROOT / 'fund_analytics').rglob('*.py')) + [ROOT / 'app.py', ROOT / 'segments_page.py']
        hits = [f'{path.relative_to(ROOT)}:{number}: {line.strip()}' for path in files if path not in shared
                for number, line in enumerate(path.read_text().splitlines(), 1)
                if re.search(r'FROM\s+(daily_sector|ingestion_runs)\b|\.execute\(', line)
                and 'benchmark.QUERIES' not in line]  # the timing button runs the fixed benchmark SQL on purpose
        self.assertEqual(hits, [], 'query the database through fund_analytics.analytics / db')

    def test_guard_catches_raw_reads(self):
        for line in ("pd.read_csv(ROOT / 'data/raw/x.csv')", 'from .db import RAW_DIR', 'for f in db.FILES.values():',
                     "duckdb.connect('data/clean/fund_analytics.duckdb')", "pd.read_csv('synthetic_clients.csv')"):
            self.assertRegex(line, FORBIDDEN)
        self.assertNotRegex("pd.read_csv(OUT / '11_top_holders.csv')", FORBIDDEN)


if __name__ == '__main__':
    unittest.main()
