# Fund flow analytics

Local analytics for the supplied synthetic charity-fund data. See [the PRD](docs/PRD.md) for scope and modelling decisions.

## Setup

Run from the project root using Python 3.12:

```sh
python3.12 -m venv venv  # only if the environment does not already exist
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m fund_analytics ingest
venv/bin/python -m fund_analytics history-window-backtest
venv/bin/python -m fund_analytics selected-forecast
venv/bin/python -m fund_analytics monthly-forecast
venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Stop the dashboard before rebuilding the database. Import and forecast generation are explicit commands; page navigation does not retrain models or reread raw CSVs.

## Data flow

```text
data/raw/*.csv ──ingest──▶ DuckDB database ──▶ forecast and backtest commands, segments ──▶ files under data/clean
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

## Daily and monthly model comparison

The default daily view compares the newer whole-fund Random Forest and Ridge winners. `history-window-backtest` generates their historical evaluation; `selected-forecast` refits the selected configurations at the latest observed cutoff. The first backtest can take several minutes.

`monthly-forecast` separately selects the best history and settings within Random Forest and Ridge for each sector/direction and horizon 1/6/12, ranking validation cumulative-total error. It compares rolling five-year, expanding and weighted-expanding histories. The current full run takes about fifteen minutes with two bounded workers. The displayed fund monthly forecasts sum sector paths; direct fund comparisons are saved separately.

Open **Forecasts → Monthly model comparison**, then choose **Daily next month** or **Monthly**. Seasonal, mean and weekday benchmarks remain available in the monthly comparison. Ranges are optional diagnostics; their observed coverage and small calibration samples are shown explicitly. Forecast windows overlap and results use synthetic data already reviewed in earlier experiments, so they are retrospective rather than an untouched prospective trial.

The monthly outlook starts after the latest complete month; daily models use the later observation cutoff. See [results and previous-model comparisons](docs/monthly_history_results.md) and the dashboard details below.

Historical experiment commands remain available for reproducibility: `forecast`, then `daily-forest-backtest`, recreate the earlier daily baseline comparison. Their legacy dashboard view has been removed.

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

### Training-history comparison

Run `venv/bin/python -m fund_analytics history-window-backtest` to compare five-year rolling, expanding, and recency-weighted expanding histories for daily next-month Ridge and random forest forecasts. Settings are selected on November 2011–October 2013, then evaluated on November 2013–September 2026. These retrospective whole-fund results supply the newer daily dashboard model choices.

Artifacts: `data/clean/history_window_backtest/`. Interpretation: [history-window results](docs/history_window_results.md).

Calibrate and evaluate indicative 80% ranges from the saved history-window forecasts with `venv/bin/python -m fund_analytics history-window-intervals`. Results: [interval coverage](docs/history_window_intervals.md).

### Fund performance features

Run `venv/bin/python -m fund_analytics performance-backtest` after `history-window-backtest`. This compares four frozen leading configurations with calendar-only inputs versus calendar plus trailing 5/21/63-valuation-day accumulation-price returns known before the target month. Both arms share eligible training dates. Separate artifacts, historical references and indicative-range diagnostics are saved in `data/clean/performance_backtest/`. See [performance-feature results](docs/performance_forecast_results.md).

### Dashboard forecast comparison

Open **Forecasts → Monthly model comparison**, the single forecasting workspace. Choose **Daily next month** or **Monthly** and compare the best **Random Forest** and **Ridge** within each family. Legacy forecast navigation has been removed.

- Generate daily artifacts with `venv/bin/python -m fund_analytics selected-forecast` after the original `history-window-backtest`. Daily whole-fund forecasts use the reviewed daily-error winners: expanding forests (depth 5, 200 trees, seed 42, leaf 12 subscriptions / 24 redemptions), and rolling 60-month Ridge alpha 0.1 for both flows. Daily forecasts include the latest observed partial month; weekly points sum forecast days only, with no weekly ranges.
- Generate monthly artifacts with `venv/bin/python -m fund_analytics monthly-forecast`. It selects rolling five-year, expanding or weighted-expanding history independently within each family, target, sector/direct-fund scope and horizon (1/6/12 months). Validation ranks cumulative cash-total error, then monthly path error. The displayed whole-fund monthly path sums selected sector paths.
- Model settings and history choices are visible. A winner may use rolling rather than expanding history. Seasonal, mean and weekday benchmarks remain visible in monthly comparisons. Net errors and ranges are derived from aggregate net flows.
- Ranges are hidden by default. Enabled diagnostics show historical coverage and the limitations of small calibration samples. Fund-performance features are excluded.

Current daily outlook: November 2026 using observations through 30 October; no October 31 observations are invented. Current monthly outlook begins October 2026 after the last complete month, September. The different origins are shown explicitly.

Daily artifacts use `selected-history-v2` under `data/clean/selected_forecasts/`. Monthly artifacts use `monthly-history-v2` under `data/clean/monthly_forecasts/`; stale versions require regeneration. Monthly generation retains original comparison metrics, settings and metadata as `previous_*` reference files and stores candidate validation scores. See [monthly history-selection results](docs/monthly_history_results.md).

## Flexible CSV datasets

Imports accept subsets of the four canonical datasets and columns. Stop the dashboard before
importing, and use a separate database when experimenting:

```sh
FUND_ANALYTICS_DB=/tmp/fund_missing_sector.duckdb venv/bin/python -m fund_analytics ingest --raw-dir examples/csv_import/missing_sector_2025
FUND_ANALYTICS_DB=/tmp/fund_missing_sector.duckdb venv/bin/python -m streamlit run app.py
```

See [the one-year examples](examples/csv_import/README.md) for a matching complete dataset,
a deterministic fixture generator and expected dashboard behaviour.

The importer scans the first 100 CSV records for a canonical header, allowing introductory
text, blank records, UTF-8 BOMs, reordered columns and header case/whitespace differences.
Use `--scan-rows 200` to expand the scan. Alternative filenames are recognised from their
columns; ambiguous files or headers require explicit overrides:

```sh
venv/bin/python -m fund_analytics ingest --raw-dir /path/to/dataset --source clients=/path/to/clients.csv --header-row clients=5
```

Header overrides count CSV records from one. Inputs are comma-delimited UTF-8 files containing
one table each. Column aliases, automatic currency conversion and arbitrary table layouts
are not supported. Extra columns are reported and ignored by analytics. Raw files are preserved.

Each chart, metric and filter checks its own dependencies. Missing or invalid fields produce
notifications naming those fields while independent components remain usable. Invalid supplied
values make the entire column unavailable; rows are not silently dropped to calculate partial
totals. Missing values are never presented as zero amounts. Transaction-only data shows observed
event dates; zero-day rates and exposure-based analyses require confirmed complete coverage.
Coverage is established by a continuous daily fund file, or by a valuation-day fund file whose
cash totals reconcile and whose omitted dates contain no cash events. This preserves the
original weekend/holiday calendar; a sparse AUM-only file does not establish zero-flow days.

Data quality shows the selected files/header rows, available columns, validation problems and
reconciliation status. Unreadable or ambiguous imports leave the previous database unchanged.
Successful partial imports replace the whole snapshot, so previous data cannot leak into results.
Forecasting requires the complete validated original schema and sufficient history, and forecast
and segment artifacts must match the current source fingerprint. The one-year examples are for
descriptive testing, not model evaluation.
