# Fund performance feature experiment

Run date: 11 November 2026. Data: supplied synthetic fund history. Scope: daily next-month whole-fund subscriptions and redemptions. Reproduce with `venv/bin/python -m fund_analytics performance-backtest` after generating the history-window backtest.

## Method

Compare the same four model configurations with calendar-only inputs and calendar plus trailing 5-, 21- and 63-valuation-day accumulation-unit-price returns. Each return is `latest_price / lagged_price - 1`, expressed as a fraction. The latest valuation must be on or before the preceding calendar month-end. Every target day in that month receives the same performance vector, including historical training targets. No realized performance inside the target month enters its forecasts. Returns count observed valuation rows, not calendar days; missing weekend valuations are not filled with invented returns.

Configurations are fixed from the previously reviewed experiment: expanding forests with depth 5 and leaf 12 for subscriptions / leaf 24 for redemptions, 200 trees and seed 42; rolling 60-month Ridge alpha 0.1 for subscriptions; expanding Ridge alpha 0.1 with one-year recency half-life for redemptions. Weighted observations retain mean-one normalization. Added Ridge features are standardized using only eligible training rows.

Training starts at the earliest month with 64 available prices: March 2007. Both arms exclude earlier training months, so controls are freshly fitted on the same eligible observations. Original results are references, rather than matched controls: their expanding histories began in November 2006. The maximum valuation age at scored monthly origins was three days.

The full comparison covers November 2013–September 2026 (155 months, 4,717 daily targets per arm). A second comparison covers November 2015–September 2026 (131 months, 3,987 daily targets). Each arm's daily and monthly-total intervals are calibrated separately on November 2013–October 2015 (730 daily residuals or 24 monthly residuals). These are fixed empirical 10th/90th-percentile residual ranges, with nonnegative amount bounds, evaluated only afterward. Monthly bounds are not sums of daily bounds.

## Full-period point errors

Lower MAE is better. Positive improvement percentages mean performance features reduce error relative to the matched calendar-only control; negative values mean higher error. The original daily MAE column refers to the earlier history-window experiment.

| Flow | Model | History | Original daily MAE | Control daily MAE | Performance daily MAE | Daily improvement | Control monthly MAE | Performance monthly MAE | Monthly improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| redemptions | random_forest | expanding | £624.8k | £624.5k | £624.5k | -0.00% | £6.796m | £6.885m | -1.31% |
| redemptions | ridge | weighted | £695.4k | £695.4k | £696.6k | -0.17% | £4.841m | £4.795m | +0.96% |
| subscriptions | random_forest | expanding | £642.0k | £646.1k | £643.1k | +0.47% | £11.130m | £11.223m | -0.84% |
| subscriptions | ridge | rolling | £730.7k | £730.7k | £730.9k | -0.02% | £5.717m | £5.903m | -3.25% |

The subscription forest's daily MAE improves by only 0.47% versus its matched control, from £646k to £643k. It remains slightly worse than the original experiment's £642k result; changing the eligible training history matters. Redemption forest daily MAE is essentially unchanged. Added performance features slightly worsen Ridge daily errors. Weighted Ridge redemption monthly-total MAE improves by about 0.96%, while the other three monthly-total errors worsen.

## Later assessment period

The table below excludes interval-calibration months. It permits comparison of point errors on exactly the same months used to assess ranges.

| Flow | Model | History | Control daily MAE | Performance daily MAE | Daily improvement | Control monthly MAE | Performance monthly MAE | Monthly improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| redemptions | random_forest | expanding | £661.6k | £661.5k | +0.01% | £7.530m | £7.632m | -1.35% |
| redemptions | ridge | weighted | £739.0k | £740.2k | -0.16% | £5.215m | £5.147m | +1.29% |
| subscriptions | random_forest | expanding | £678.1k | £674.5k | +0.54% | £11.767m | £11.866m | -0.84% |
| subscriptions | ridge | rolling | £766.4k | £767.5k | -0.14% | £6.037m | £6.235m | -3.28% |

## Yearly stability

The first and last year blocks are partial. Counts below describe how many full-period calendar-year blocks saw lower errors with performance features; they are not significance tests.

| Flow | Model | History | Years with lower daily MAE | Years with lower monthly MAE |
| --- | --- | --- | --- | --- |
| redemptions | random_forest | expanding | 9/14 | 4/14 |
| redemptions | ridge | weighted | 5/14 | 5/14 |
| subscriptions | random_forest | expanding | 9/14 | 4/14 |
| subscriptions | ridge | rolling | 7/14 | 4/14 |

## Indicative 80% ranges

Coverage is the share of actual outcomes inside the range, and width is upper minus lower bound. Coverage and widths below apply to the later 131-month assessment.

| Scale | Flow | Model | Control coverage | Performance coverage | Control mean width | Performance mean width |
| --- | --- | --- | --- | --- | --- | --- |
| daily_amount | redemptions | random_forest | 69.2% | 69.3% | £1.103m | £1.101m |
| daily_amount | redemptions | ridge | 67.4% | 67.1% | £1.062m | £1.052m |
| daily_amount | subscriptions | random_forest | 65.2% | 65.6% | £1.181m | £1.187m |
| daily_amount | subscriptions | ridge | 66.2% | 66.2% | £1.351m | £1.364m |
| monthly_total | redemptions | random_forest | 35.1% | 35.1% | £7.240m | £7.528m |
| monthly_total | redemptions | ridge | 32.1% | 35.1% | £5.098m | £5.023m |
| monthly_total | subscriptions | random_forest | 49.6% | 45.0% | £21.603m | £21.368m |
| monthly_total | subscriptions | ridge | 51.9% | 52.7% | £10.116m | £10.535m |

Performance-feature daily ranges cover about 66–69% of actuals, and monthly-total ranges about 35–53%. No arm achieves nominal 80% coverage. Small coverage improvements do not repair the calibration problem. Only 24 monthly residuals calibrate each monthly range, and daily residuals are dependent over time. Increasing width alone is not evidence of a better forecast.

## Recommendation and limits

Do not adopt this feature set as a general improvement on current evidence. It produces tiny or mixed gains, does not consistently improve yearly errors, and leaves range undercoverage unresolved. It remains possible that different performance definitions or prediction horizons are useful, but this experiment does not establish that.

This is retrospective synthetic-data research. Configurations were chosen after reviewing historical performance, including the evaluation years; it is not an untouched test. Performance association does not establish investor behavior or causation. Statistical significance is not claimed. No further feature or parameter search was performed on these results.

## Artifacts and checks

Artifacts under `data/clean/performance_backtest/` include `detail.csv`, `monthly.csv`, `metrics.csv` (both comparison periods, overall/yearly scores, bias, normalized errors and benchmark improvements), `comparisons.csv`, `previous_reference_metrics.csv`, `features.csv`, `models.csv`, and `metadata.json`. Interval outputs have the `interval_` prefix and include daily/monthly ranges, overall/yearly metrics and calibration counts. `approach` encodes both training history and feature arm so existing grouping keeps each arm separate; point metrics also include explicit `history` and `feature_set` columns.

All twelve point groups (eight model arms plus four flow/benchmark groups) were checked for identical 155-month/4,717-day scoring coverage, finite nonnegative predictions and valid feature cutoffs. Both feature arms have matching training boundaries and row counts. All sixteen interval metric rows have 131 monthly outcomes or 3,987 daily outcomes, calibrated separately from 24 or 730 earlier residuals. Tests cover return calculations, month-end freezing, future-data isolation, invalid prices, insufficient history, paired training eligibility and Ridge scaling. All 29 tests passed.

Runtime: 309.4 seconds (about 5.2 minutes).
