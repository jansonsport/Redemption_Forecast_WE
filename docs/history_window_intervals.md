# Indicative 80% ranges for daily history-window forecasts

Run date: 11 November 2026. Data: supplied synthetic fund history. The point forecasts and selected model settings come from the [history-window backtest](history_window_results.md). These are empirical ranges based on historical errors; “80%” is the target coverage, not a guarantee.

## Calibration and evaluation split

- Model settings were selected on November 2011–October 2013.
- Residuals from the next 24 monthly forecasts (November 2013–October 2015) calibrate the ranges.
- Coverage is measured on the remaining 131 monthly forecasts (November 2015–September 2026): 3,987 daily observations or 131 monthly totals for each target/model/approach.
- For each forecast type, compute actual minus predicted residuals on calibration data. The 10th and 90th percentiles define a central empirical range. Daily errors and aggregated monthly-total errors are calibrated independently. Lower bounds are clipped at zero for subscriptions and redemptions.
- Settings are frozen before calibration. No evaluation residual is used to construct the bounds.

## Coverage and average range width

Coverage is the share of later actual outcomes inside the range. Width is the mean distance between upper and lower bounds. GBP widths are shown in millions or thousands.

| Forecast scale | Flow | Model | History | Observed coverage | Mean width | Calibration cases |
| --- | --- | --- | --- | --- | --- | --- |
| Daily Amount | Subscriptions | random_forest | Expanding | 64.5% | £1.16m | 730 |
| Daily Amount | Subscriptions | random_forest | 5y rolling | 64.2% | £1.11m | 730 |
| Daily Amount | Subscriptions | random_forest | Weighted expanding | 65.4% | £1.16m | 730 |
| Daily Amount | Subscriptions | ridge | Expanding | 67.8% | £1.40m | 730 |
| Daily Amount | Subscriptions | ridge | 5y rolling | 66.2% | £1.35m | 730 |
| Daily Amount | Subscriptions | ridge | Weighted expanding | 66.3% | £1.35m | 730 |
| Daily Amount | Redemptions | random_forest | Expanding | 69.5% | £1.11m | 730 |
| Daily Amount | Redemptions | random_forest | 5y rolling | 69.9% | £1.24m | 730 |
| Daily Amount | Redemptions | random_forest | Weighted expanding | 69.0% | £1.12m | 730 |
| Daily Amount | Redemptions | ridge | Expanding | 68.5% | £1.17m | 730 |
| Daily Amount | Redemptions | ridge | 5y rolling | 67.0% | £1.13m | 730 |
| Daily Amount | Redemptions | ridge | Weighted expanding | 67.3% | £1.06m | 730 |
| Monthly Total | Subscriptions | random_forest | Expanding | 43.5% | £20.54m | 24 |
| Monthly Total | Subscriptions | random_forest | 5y rolling | 45.8% | £19.00m | 24 |
| Monthly Total | Subscriptions | random_forest | Weighted expanding | 42.7% | £20.22m | 24 |
| Monthly Total | Subscriptions | ridge | Expanding | 47.3% | £10.78m | 24 |
| Monthly Total | Subscriptions | ridge | 5y rolling | 51.9% | £10.12m | 24 |
| Monthly Total | Subscriptions | ridge | Weighted expanding | 48.9% | £9.29m | 24 |
| Monthly Total | Redemptions | random_forest | Expanding | 37.4% | £7.06m | 24 |
| Monthly Total | Redemptions | random_forest | 5y rolling | 41.2% | £7.35m | 24 |
| Monthly Total | Redemptions | random_forest | Weighted expanding | 37.4% | £7.17m | 24 |
| Monthly Total | Redemptions | ridge | Expanding | 33.6% | £7.04m | 24 |
| Monthly Total | Redemptions | ridge | 5y rolling | 38.9% | £5.91m | 24 |
| Monthly Total | Redemptions | ridge | Weighted expanding | 32.1% | £5.09m | 24 |

## Interpretation

None of the tested ranges reached the nominal 80% coverage. Daily coverage ranged from 64% to 70%. Monthly-total coverage ranged from 32% to 52%. The monthly calibration set contains only 24 totals per model and target, so its tail estimates and coverage are especially uncertain. Daily residuals are also serially related, so 3,987 daily observations do not represent 3,987 independent market conditions.

The ranges are too narrow, miss in both tails, and may also be affected by changing forecast errors over time. Increasing the nominal range alone would make ranges wider but does not establish reliable calibration. Treat these as diagnostics; do not present them as dependable 80% probabilities. Improving calibration requires more independent evaluation history or a prospective period, plus testing whether residual spread changes over time and across flow regimes.

## Reproduction and files

Run `venv/bin/python -m fund_analytics history-window-intervals` after generating the point-forecast artifacts. Outputs are in `data/clean/history_window_backtest/intervals/`: `daily_ranges.csv`, `monthly_ranges.csv`, `metrics.csv`, `yearly_metrics.csv` and `metadata.json`. All evaluation months have previously been reviewed in earlier model experiments, so these results are retrospective rather than an untouched final test.
