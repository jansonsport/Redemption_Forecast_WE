"""One temporary analytics database per test session, selected through FUND_ANALYTICS_DB."""
import os
import shutil
import tempfile
from pathlib import Path

TEMP = Path(tempfile.mkdtemp(prefix='fund_analytics_tests_'))
os.environ['FUND_ANALYTICS_DB'] = str(TEMP / 'fund_analytics.duckdb')  # before fund_analytics is imported

from fund_analytics import db  # noqa: E402


def pytest_sessionstart(session):
    assert db.DB_PATH == TEMP / 'fund_analytics.duckdb'
    db.ingest()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEMP, ignore_errors=True)
