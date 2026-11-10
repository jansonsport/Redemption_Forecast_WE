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
