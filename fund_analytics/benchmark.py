"""Repeatable uncached SQL timings; no dashboard cache involved."""
import json
import platform
import statistics
import time
from pathlib import Path
from .db import connect

QUERIES = {
    'sector_totals': 'SELECT client_sector, sum(subscriptions), sum(redemptions) FROM daily_sector GROUP BY 1',
    'monthly_history': "SELECT date_trunc('month', date), sum(subscriptions), sum(redemptions) FROM daily_sector GROUP BY 1 ORDER BY 1",
    'client_history': "SELECT cash_flow_date, sum(flow_gbp) FROM client_flows WHERE client_id = 'C000001' GROUP BY 1 ORDER BY 1",
}


def run():
    results = {}
    with connect() as con:
        for name, sql in QUERIES.items():
            timings = []
            for _ in range(20):
                started = time.perf_counter()
                con.execute(sql).fetchall()
                timings.append(time.perf_counter() - started)
            results[name] = {'median_seconds': statistics.median(timings), 'p95_seconds': sorted(timings)[18], 'sql': sql}
    report = {'machine': platform.platform(), 'processor': platform.processor(), 'python': platform.python_version(), 'runs_per_query': 20, 'cache_note': 'No application cache; OS and database caches may be warm.', 'queries': results}
    db = Path('data/clean/fund_analytics.duckdb')
    report['database_bytes'] = db.stat().st_size
    path = Path('data/clean/benchmark.json')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
