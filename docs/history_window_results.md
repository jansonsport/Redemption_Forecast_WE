# Daily training-history comparison

Run date: 11 November 2026. Scope: direct whole-fund cash subscriptions and redemption magnitudes. Data: supplied synthetic history. Reproduce with `venv/bin/python -m fund_analytics history-window-backtest`.

## Setup

Three approaches are tested with Ridge and random forest: equal-weight five-year rolling history, equal-weight expanding history, and expanding history with exponential recency weights. Expanding models begin at November 2006 and approach twenty years of training history by the last forecast.

- Initial training: November 2006–October 2011.
- Validation: November 2011–October 2013 (24 monthly origins).
- Evaluation: November 2013–September 2026 (155 monthly origins, 4,717 daily predictions per method/target).
- All methods forecast the next calendar month and refit using observations available through the preceding month. Partial October 2026 is excluded from scoring. Evaluation observations become eligible training data only for subsequent forecasts.
- Settings are selected independently per target/model/approach by validation daily MAE, then monthly-total MAE, then listed grid order, and remain fixed during evaluation. Previously selected recent-period settings are not reused.

Ridge alpha candidates were 0.1, 1 and 10. Forest depth/leaf pairs were (3,6), (3,12), (5,12), (5,24), with 200 trees, seed 42 and one fitting worker. Calendar features are weekday, month, position within month and time trend. The trend is scaled using training data only. Negative amount predictions are clipped at zero.

Weights are `2 ** (-age_days / (365.25 * half_life_years))`, normalized to average 1. Validation compares half-lives of 1, 3 and 5 years. This keeps Ridge regularization comparable across differently weighted training sets.

## Overall results

Lower MAE is better. “Error / mean actual” is daily MAE divided by average actual daily flow, equivalent to total absolute error divided by total actual flow. It is not the mean daily percentage error or a forecast accuracy percentage. Average daily actual flows were £839k subscriptions and £765k redemptions.

| Flow | Model | History | Daily MAE | Error / mean actual | Daily improvement vs weekday | Monthly-total MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Subscriptions | mean | 12-month benchmark | £1,001k | 119.2% | -19.1% | £11.962m |
| Subscriptions | random_forest | Expanding | £642k | 76.5% | 23.6% | £11.205m |
| Subscriptions | random_forest | Rolling 5 years | £644k | 76.8% | 23.3% | £10.171m |
| Subscriptions | random_forest | Weighted expanding | £651k | 77.6% | 22.5% | £11.085m |
| Subscriptions | ridge | Expanding | £750k | 89.4% | 10.8% | £6.254m |
| Subscriptions | ridge | Rolling 5 years | £731k | 87.1% | 13.1% | £5.717m |
| Subscriptions | ridge | Weighted expanding | £735k | 87.5% | 12.6% | £5.790m |
| Subscriptions | weekday | 12-month benchmark | £840k | 100.1% | 0.0% | £11.896m |
| Redemptions | mean | 12-month benchmark | £937k | 122.6% | -17.0% | £17.026m |
| Redemptions | random_forest | Expanding | £625k | 81.7% | 22.0% | £6.776m |
| Redemptions | random_forest | Rolling 5 years | £699k | 91.4% | 12.8% | £4.982m |
| Redemptions | random_forest | Weighted expanding | £638k | 83.4% | 20.4% | £5.726m |
| Redemptions | ridge | Expanding | £707k | 92.4% | 11.8% | £6.423m |
| Redemptions | ridge | Rolling 5 years | £690k | 90.2% | 13.9% | £4.866m |
| Redemptions | ridge | Weighted expanding | £695k | 91.0% | 13.2% | £4.841m |
| Redemptions | weekday | 12-month benchmark | £801k | 104.8% | 0.0% | £16.982m |

## Interpretation

- Expanding-history forests achieve the lowest daily MAE for both targets: about £642k subscriptions (23.6% below the weekday benchmark) and £625k redemptions (22.0% below). For subscriptions, the rolling forest is close at £644k; expanding history improves it by only about 0.4%.
- Expanding history improves forest redemption daily MAE by about 10.6% versus rolling history. However, monthly-total MAE worsens from £4.98m to £6.78m. Lower daily error does not ensure more accurate monthly totals.
- Ridge benefits from the rolling window for daily predictions. Weighted expanding Ridge has the lowest redemption monthly-total MAE at £4.84m, but that is only about 0.5% below rolling Ridge (£4.87m). Rolling Ridge has the lowest subscription monthly-total MAE at £5.72m.
- Recency weighting is not a consistent winner. Weighted approaches have higher overall daily MAE than their rolling counterparts for Ridge and subscription forests; weighted redemption forests improve on rolling forests but remain behind equal-weight expanding forests.
- The expanding forest underpredicts on average by about £110k/day for subscriptions and £195k/day for redemptions. Its daily MAE remains 76.5% and 81.7% of mean actual flow. These are substantial remaining errors despite benchmark improvements.

For a daily-error objective, retain expanding forest as the leading challenger, with the rolling subscription forest a close alternative. For monthly cash planning, Ridge remains stronger. These reviewed historical results do not justify automatically replacing the dashboard model or presenting calibrated uncertainty ranges.

## Selected settings

| Flow | Model | History | Setting | Half-life |
| --- | --- | --- | --- | --- |
| subscriptions | ridge | Rolling 5 years | alpha 0.1 | — |
| subscriptions | ridge | Expanding | alpha 0.1 | — |
| subscriptions | ridge | Weighted expanding | alpha 0.1 | 1 years |
| subscriptions | random_forest | Rolling 5 years | depth 5, leaf 12 | — |
| subscriptions | random_forest | Expanding | depth 5, leaf 12 | — |
| subscriptions | random_forest | Weighted expanding | depth 5, leaf 12 | 5 years |
| redemptions | ridge | Rolling 5 years | alpha 0.1 | — |
| redemptions | ridge | Expanding | alpha 0.1 | — |
| redemptions | ridge | Weighted expanding | alpha 0.1 | 1 years |
| redemptions | random_forest | Rolling 5 years | depth 5, leaf 24 | — |
| redemptions | random_forest | Expanding | depth 5, leaf 24 | — |
| redemptions | random_forest | Weighted expanding | depth 5, leaf 24 | 5 years |

## Stability across evaluation years

The first and last years are partial (November–December 2013 and January–September 2026). Yearly scores are descriptive and are not used to retune settings. Daily winners among the six model combinations were:

| Year | Subscriptions: lowest daily MAE | Redemptions: lowest daily MAE |
| --- | --- | --- |
| 2013 | ridge, Rolling 5 years (£411k) | random_forest, Weighted expanding (£458k) |
| 2014 | random_forest, Expanding (£450k) | random_forest, Expanding (£415k) |
| 2015 | random_forest, Expanding (£555k) | random_forest, Expanding (£446k) |
| 2016 | random_forest, Expanding (£591k) | random_forest, Expanding (£503k) |
| 2017 | random_forest, Weighted expanding (£514k) | random_forest, Expanding (£519k) |
| 2018 | random_forest, Rolling 5 years (£538k) | random_forest, Expanding (£541k) |
| 2019 | random_forest, Expanding (£651k) | random_forest, Expanding (£582k) |
| 2020 | random_forest, Expanding (£555k) | random_forest, Expanding (£746k) |
| 2021 | random_forest, Expanding (£809k) | random_forest, Expanding (£663k) |
| 2022 | random_forest, Rolling 5 years (£697k) | random_forest, Expanding (£683k) |
| 2023 | random_forest, Rolling 5 years (£684k) | random_forest, Expanding (£726k) |
| 2024 | random_forest, Weighted expanding (£758k) | random_forest, Expanding (£755k) |
| 2025 | random_forest, Rolling 5 years (£804k) | random_forest, Expanding (£796k) |
| 2026 | random_forest, Weighted expanding (£737k) | random_forest, Expanding (£826k) |

Within each model, the number of year blocks where each expanding approach beats rolling history on daily MAE is:

| Flow | Model | Expanding beats rolling | Weighted beats rolling |
| --- | --- | --- | --- |
| redemptions | random_forest | 14/14 | 14/14 |
| redemptions | ridge | 3/14 | 6/14 |
| subscriptions | random_forest | 8/14 | 4/14 |
| subscriptions | ridge | 0/14 | 4/14 |

## Limits and artifacts

This is retrospective synthetic-data research, not a prospective real-client validation. Some later outcomes were examined in earlier experiments. There is one early validation block; settings selected there may not remain optimal throughout thirteen evaluation years. Recent errors can be larger than the overall averages, so examine yearly metrics as well as pooled results. Training-history effects are compared after separately tuning each approach, rather than holding model settings identical. This experiment reports point forecasts only and makes no claim about 80% interval coverage.

All 16 evaluation groups (eight methods × two targets) were checked for identical 4,717 dates and 155 months, finite nonnegative predictions, and training cutoffs before forecast dates. Tests cover missing/duplicate history, splits, weighting, future-data isolation, zero histories, selection tie-breaking, leap-year training status and metrics. The complete test suite passed (23 tests).

Artifacts under `data/clean/history_window_backtest/`:

- `detail.csv`: daily selected-validation and evaluation predictions, actuals, errors, training boundaries and status.
- `monthly.csv`: monthly totals and errors, aggregated before absolute values.
- `metrics.csv`: overall/yearly daily MAE, bias, monthly-total MAE, flow-normalized error and improvements over both benchmarks. Positive improvements mean lower error; zero denominators are unavailable.
- `comparisons.csv`: paired history-approach differences and improvements, overall and yearly.
- `models.csv`: twelve selected configurations and validation scores; alpha applies to Ridge, depth/leaf to forests.
- `metadata.json`: source fingerprint, exact splits, candidate grids, feature/library versions and runtime.

Backtest runtime: 496.1 seconds (about 8.3 minutes). Generated artifacts are excluded from Git by the existing `data/clean/*` rule.
