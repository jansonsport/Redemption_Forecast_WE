# Fund flow analytics

Local analytics for the supplied synthetic charity-fund data. See [the PRD](docs/PRD.md) for scope and modelling decisions.

## Setup

Run from the project root using Python 3.12:

```sh
python3.12 -m venv venv  # only if the environment does not already exist
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m fund_analytics ingest
venv/bin/python -m fund_analytics forecast
venv/bin/python -m fund_analytics daily-forest-backtest
venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Stop the dashboard before rebuilding the database. Import and forecast generation are explicit commands; page navigation does not retrain models or reread raw CSVs.

## Data flow

```text
data/raw/*.csv ──ingest──▶ DuckDB database ──▶ forecast, monthly-forecast, segments ──▶ files under data/clean
                                         └──▶ dashboard (app.py)
```

- `ingest` (`fund_analytics/db.py`) is the only code that reads the raw CSVs. It validates and reconciles
  them and replaces the database tables in one transaction; an unchanged snapshot is a no-op. Each import
  is recorded in `ingestion_runs`, and the latest import (last inserted row) gives the fingerprint that
  every output carries.
- Everything else reads the database through `fund_analytics/analytics.py`: the dashboard's queries, the
  forecast histories and the segments study's source tables. Outputs (forecasts, monthly forecasts,
  segments, benchmark) are still files under `data/clean`; pages refuse outputs built from another import.
- The database is `data/clean/fund_analytics.duckdb` unless `FUND_ANALYTICS_DB` names another file, e.g.
  `FUND_ANALYTICS_DB=/tmp/other.duckdb venv/bin/python -m fund_analytics ingest`. Set it the same way for
  the build commands and `streamlit run app.py`. The tests use their own temporary database.
- `ingest` needs the dashboard stopped: DuckDB lets one process write or several processes read, not
  both. A lock clash is reported as "The database is not ready" (dashboard) or `DatabaseUnavailable`.

## Monthly random forest comparison

```sh
venv/bin/python -m fund_analytics monthly-forecast
```

This offline command fits constrained forests and the existing baseline methods,
evaluates monthly forecasts through twelve months, and publishes artifacts under
`data/clean/monthly_forecasts`. It can take several minutes. Open **Forecasts →
Monthly model comparison** for six/twelve-month outlooks and error comparisons.

The separate `venv/bin/python -m fund_analytics daily-forest-backtest` command
compares constrained random forest with the current whole-fund daily Ridge
choices on the same historical cutoffs. CSV results go to
`data/clean/daily_forest_backtest*`.

Four forest configurations use 200 trees, depths 3/5 and minimum leaves 6/12,
with a fixed seed. One configuration per sector and flow direction is selected
on 12 validation origins, using mean monthly absolute error across leads 1–12.
The model learns all leads together, with lead month as a feature; only targets
known by each fitting cutoff enter training. Its examples cover up to 120 prior
origin months. These rows share outcomes and are not independent monthly samples.

A separate 12-origin calibration period follows model selection; every
12-month validation path finishes before calibration begins, and every
calibration path finishes before the first evaluation origin. The final
24 monthly evaluation origins cover all calendar months. Their forecast
windows overlap, so error observations are dependent. This is retrospective
research, not an untouched prospective trial. Empirical intervals have only
12 calibration errors per lead and should be treated as indicative.

The outlook starts after the latest **complete** month: incomplete current-month
observations are excluded. Existing daily forecasts retain their original cutoff.
The historical same-month-last-year rule is included as a serious benchmark;
random forest is an experimental challenger, not an assumed improvement.

## Segments & stress (client relationships)

```sh
venv/bin/python -m fund_analytics segments
```

Builds the **Segments & stress** page data in about 10 seconds: client segments
(k-means), segment and sector flow correlation (heat maps in section 4),
concentration and liquidity stress tests.
It reads one snapshot of the database tables, writes the `11_*.csv` tables and `bundle.pkl` to
`data/clean/segments`, and copies the study notebook to `static/segments` for the
page's "Open full study notebook" link. Rebuild after every new import: the page
refuses a bundle built from a different import (or when there is no import). Restart the dashboard
after rebuilding.

This is **descriptive analysis of relationships between clients, not a
forecast**. It does not feed the Forecasts page. Segments are assigned using
each client's full history (hindsight). Correlation is co-movement, not
causation, and stress scenarios are illustrative exposures. There are no
portfolio holdings, so cash buffers are illustrative. Segment and client
holdings are reconstructed as units × price and checked against reported fund
AUM; see the PRD correctness rules.

The study is ported from `handoff/segments_stress/` (the original analysis,
notebook and reference UI). Changes for this repository:

- Inputs come from the database tables (`CONFIG['tables']`) instead of the CSVs, outputs go to
  `data/clean/segments`, and the code is updated for pandas 3 (read-only arrays; `'YE'`/`'ME'` resample aliases).
- **k is fixed at 7** (`CONFIG['k_fixed']`). On silhouette, 7 and 8 groups are
  nearly tied, and the scikit-learn version decides which wins. The automatic
  choice is still shown on the page and is 8 in this environment.
- The main k-means fit uses **50 restarts** (`CONFIG['n_init']`, was 10). With
  10, this scikit-learn version stopped at a worse k=7 solution. With 50, results
  are stable (50 and 200 restarts agree), 98% of clients get the same segment as
  in the original study, and all seven segment names match.
- The browser stress engine is now Python (`fund_analytics/segments/engine.py`)
  and matches the pipeline exactly for the default stress definition. A custom
  stress definition lasts for the session; download it as JSON to keep it.

Numbers that differ from the original study because of the clustering changes:
mean daily segment correlation on stress days is 0.13 (study 0.12), and median
bootstrap stability is 0.70 (study 0.72). Everything that does not depend on the
clusters is identical, including stress days, rare traders, concentration and
the reverse stress test.

## Validation and performance

```sh
venv/bin/python -m pytest -q
venv/bin/python -m fund_analytics benchmark
```

Benchmark results are stored in `data/clean/benchmark.json`. They distinguish uncached SQL execution from dashboard caching; OS/database caches may still be warm. Performance targets in the PRD are provisional until measured on your machine.

The raw CSVs are preserved. The generated DuckDB database and forecast artifacts live under `data/clean` and can be recreated. Source fingerprints prevent mixing analysis and forecasts from different snapshots.

## Notebook kernel

```sh
venv/bin/python -m ipykernel install --user --name redemption-forecasting --display-name 'Python (redemption_forecasting)'
```

Select that kernel in Jupyter. The `fund_analytics` package is intended for reuse by notebooks started from the project root.

## Interpretation

Subscriptions and redemption magnitudes are positive; net flow is subscriptions minus redemptions. Cash dates drive forecasting. Distributions are separate obligations, and synthetic patterns do not establish real client behaviour. Historical client AUM is not inferred from present-day snapshots. Forecast error ranges are empirical and indicative, not guarantees.
