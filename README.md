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

## Daily and monthly model comparison

The default daily view compares the newer whole-fund Random Forest and Ridge winners. `history-window-backtest` generates their historical evaluation; `selected-forecast` refits the selected configurations at the latest observed cutoff. The first backtest can take several minutes.

`monthly-forecast` separately selects the best history and settings within Random Forest and Ridge for each sector/direction and horizon 1/6/12, ranking validation cumulative-total error. It compares rolling five-year, expanding and weighted-expanding histories. The current full run takes about fifteen minutes with two bounded workers. The displayed fund monthly forecasts sum sector paths; direct fund comparisons are saved separately.

Open **Forecasts → Monthly model comparison**, then choose **Daily next month** or **Monthly**. Seasonal, mean and weekday benchmarks remain available in the monthly comparison. Ranges are optional diagnostics; their observed coverage and small calibration samples are shown explicitly. Forecast windows overlap and results use synthetic data already reviewed in earlier experiments, so they are retrospective rather than an untouched prospective trial.

The monthly outlook starts after the latest complete month; daily models use the later observation cutoff. See [results and previous-model comparisons](docs/monthly_history_results.md) and the dashboard details below.

Historical experiment commands remain available for reproducibility: `forecast`, then `daily-forest-backtest`, recreate the earlier daily baseline comparison. Their legacy dashboard view has been removed.

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
