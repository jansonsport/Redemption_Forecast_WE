# Unified forecasting and monthly history selection

Run date: 11 November 2026. Data: supplied synthetic cash-flow history. The dashboard now has a single **Forecasts → Monthly model comparison** workspace, with **Daily next month** and **Monthly** resolutions. The legacy forecast and separate selected-model views have been removed. Each view offers the best Random Forest and best Ridge within its evaluation design; the two model families are retained even when a simple benchmark performs better.

## Monthly experiment

For each sector and direct-fund scope, flow direction and horizon 1/6/12, compare rolling five-year, expanding and recency-weighted expanding histories. Choose settings by validation cumulative-total MAE, then mean monthly MAE, then fixed candidate order. Weighted candidates have half-lives 1/3/5 years; forests use 200 trees, depth 3/5, leaf 6/12, seed 42; Ridge alpha is 0.1/1/10. Forests retain monthly flow/calendar features; Ridge retains daily calendar features whose predictions are aggregated into monthly totals. Fund-performance features are excluded.

Rolling forests restrict eligible training targets to the latest 60 months, while allowing older feature lookbacks. Expanding forests use all eligible targets. Forest recency weights use target-month age; Ridge weights use training-day age. Weights are normalized to mean one. Targets and features must be known at each fitting cutoff. Shared target outcomes across leads and overlapping evaluation paths are dependent.

- Validation origins: 2019-12–2020-11 (12 monthly origins).
- Calibration origins: 2021-11–2022-10 (12 monthly origins).
- Evaluation origins: 2023-10–2025-09 (24 monthly origins).

Twelve-month validation outcomes are known before calibration begins; calibration outcomes are known before evaluation begins. Settings are frozen before calibration and evaluation, and models refit at each origin. The displayed whole-fund monthly paths sum independently selected sector paths. Direct-fund fits are saved separately and are not substituted for these coherent totals. Net actuals and predictions are derived before calibration.

## Whole-fund cumulative-total errors

MAE is in GBP and lower is better. These are errors in the sum across the next 1, 6 or 12 months, not just the final month's amount. The previous column uses original forest results or fixed 36-month Ridge alpha 1 on the same evaluation origins. For horizon 1, the previous individual-month error equals its one-month total error. Positive seasonal improvement means the new model has lower error than the seasonal benchmark.

| Months | Flow | Family | Previous MAE | New MAE | Improvement vs previous | Seasonal MAE | Improvement vs seasonal | Total-range coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | subscriptions | random_forest | £5.55m | £5.77m | -4.0% | £5.16m | -11.9% | 54.2% |
| 1 | subscriptions | ridge | £8.52m | £8.47m | +0.6% | £5.16m | -64.2% | 54.2% |
| 1 | redemptions | random_forest | £4.08m | £3.89m | +4.5% | £4.56m | +14.6% | 95.8% |
| 1 | redemptions | ridge | £7.57m | £6.83m | +9.7% | £4.56m | -49.9% | 75.0% |
| 6 | subscriptions | random_forest | £25.35m | £23.40m | +7.7% | £18.11m | -29.2% | 58.3% |
| 6 | subscriptions | ridge | £27.96m | £15.21m | +45.6% | £18.11m | +16.0% | 70.8% |
| 6 | redemptions | random_forest | £17.70m | £18.93m | -7.0% | £12.29m | -54.0% | 79.2% |
| 6 | redemptions | ridge | £27.92m | £19.42m | +30.4% | £12.29m | -58.0% | 75.0% |
| 12 | subscriptions | random_forest | £34.42m | £42.55m | -23.6% | £26.29m | -61.9% | 45.8% |
| 12 | subscriptions | ridge | £59.62m | £34.64m | +41.9% | £26.29m | -31.8% | 66.7% |
| 12 | redemptions | random_forest | £41.11m | £43.32m | -5.4% | £22.95m | -88.8% | 16.7% |
| 12 | redemptions | ridge | £45.68m | £21.67m | +52.6% | £22.95m | +5.6% | 25.0% |

Results are mixed. Six-month subscription Ridge totals beat the seasonal benchmark by about 16%; twelve-month redemption Ridge totals beat it by about 6%. One-month redemption forests also beat the seasonal benchmark, but both families have larger seasonal errors for twelve-month subscriptions and six-month redemptions. Forest twelve-month totals remain particularly weak. Do not interpret family selection as proof that a model beats simple forecasts.

This is a revised experiment rather than an isolated history-window ablation: history choice, parameter tuning and the cumulative-total selection objective all differ from the earlier comparison. Improvements against previous results cannot be attributed solely to expanding history.

## Selected histories

The counts below cover the 22 sector/direction configurations per family/horizon, whose predictions form displayed fund totals. Each configuration can differ by direction and horizon; the dashboard exposes its actual settings rather than renaming every model “expanding.”

| Family | Horizon | Rolling | Expanding | Weighted expanding |
| --- | --- | --- | --- | --- |
| random_forest | 1 | 3 | 4 | 15 |
| random_forest | 6 | 4 | 6 | 12 |
| random_forest | 12 | 3 | 3 | 16 |
| ridge | 1 | 8 | 3 | 11 |
| ridge | 6 | 4 | 9 | 9 |
| ridge | 12 | 6 | 9 | 7 |

## Newer daily comparison

Daily forecasts use the original reviewed daily-history experiment, not the monthly configurations. Select the lowest daily MAE within each family/flow, with monthly-total MAE and history order as tie-breaks. The resulting forest uses expanding history for both flows (depth 5, leaf 12 subscriptions / 24 redemptions, 200 trees, seed 42). Ridge uses rolling 60 months and alpha 0.1 for both flows. Net points and ranges are computed separately within each family. Daily sector forecasts are outside this change; sector forecasts are available at monthly resolution.

The daily outlook forecasts November 2026 from observations through 30 October 2026; no October 31 observation is fabricated. Monthly paths start October 2026 after the latest complete month, September 2026. These different origins are visible in the dashboard.

## Ranges and limitations

Daily range calibration remains November 2013–October 2015, with assessment November 2015–September 2026. Monthly ranges use only 12 calibration origins per lead and selected horizon, and cumulative ranges use cumulative residuals rather than sums of marginal bounds. The table reports observed cumulative coverage; it can be far below the nominal 80%. Small samples and dependence make this imprecise. All range diagnostics are hidden by default and display measured coverage when enabled. Weekly points sum forecast days only, including partial boundary weeks, and have no weekly ranges.

These are retrospective synthetic-data results. Past outcomes were reviewed in earlier experiments; this is not an untouched prospective trial and does not establish real-client behavior or reliable probabilities. Daily winners were selected after reviewing their historical evaluation scores.

## Reproduction and checks

Generate daily outlooks with `venv/bin/python -m fund_analytics selected-forecast` after the original daily history backtest, and monthly paths with `venv/bin/python -m fund_analytics monthly-forecast`. Artifacts use `selected-history-v2` under `data/clean/selected_forecasts/` and `monthly-history-v2` under `data/clean/monthly_forecasts/`. Stale versions show regeneration instructions. Monthly artifacts include candidate validation scores and original comparison metrics, settings and metadata as `previous_*` references; source fingerprints and exact splits are retained.

All 36 tests passed. Full-run checks confirmed 144 selected configurations against 2,520 candidate/horizon validation scores, complete 24-origin scoring, coherent fund/net predictions and cumulative totals, and finite nonnegative amount predictions. Streamlit checks cover both families, monthly 1/6/12 horizons, fund/sector/net scope, optional ranges, newer daily/weekly views, stale artifact rejection and absence of legacy navigation.

Monthly runtime: 923.0 seconds (about 15.4 minutes), using two bounded workers.
