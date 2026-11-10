# Monthly random forest experiment

Run date: 9 November 2026. Dataset: supplied synthetic fund flows. This is a
retrospective comparison, not a prospective validation or evidence of real
client behaviour. Reproduce with `venv/bin/python -m fund_analytics monthly-forecast`.

## Method

Constrained random forests predict monthly subscription and redemption amounts
directly, jointly learning lead months 1–12 with lead as a feature. Inputs are
calendar seasonality, the last two observed monthly totals, trailing 3/12-month
means and the corresponding month last year. No future actuals enter features.
Only training targets observed by the fitting cutoff are eligible.

One configuration per sector and direction is chosen from 200-tree forests
with depth 3/5 and minimum leaf size 6/12, seed 42. Examples cover up to 120
past origin months. Shared target outcomes across examples are not independent.

- Validation origins: December 2019–November 2020.
- Calibration origins: November 2021–October 2022.
- Evaluation origins: October 2023–September 2025 (24 origins, every calendar month).

Twelve-month paths from validation finish before calibration begins; calibration
paths finish before evaluation begins. Settings are frozen before evaluation,
although models refit on data available at each origin. Overlapping evaluation
paths share outcomes. The previous experiments already reviewed historical data,
so these results must not be called an untouched final test.

Baselines are the corresponding month last year, mean daily flow over 12 months,
weekday averages over 12 months, and Ridge regression over 36 months (alpha 1).
Daily baseline predictions are aggregated to identical monthly targets. This
comparison uses fixed baseline settings, not exhaustive baseline retuning.

## Whole-fund results

Whole-fund predictions sum sector predictions. MAE is mean absolute error in GBP.
These rows concern the individual sixth/twelfth forecast month, not cumulative totals.

| Target | Lead month | Forest MAE | Seasonal baseline MAE | Forest error change |
| --- | ---: | ---: | ---: | ---: |
| Subscriptions | 1 | £5.55m | £5.16m | 7.6% higher |
| Subscriptions | 6 | £6.06m | £5.06m | 19.8% higher |
| Subscriptions | 12 | £5.95m | £5.78m | 3.0% higher |
| Redemptions | 1 | £4.08m | £4.56m | 10.5% lower |
| Redemptions | 6 | £5.03m | £4.51m | 11.5% higher |
| Redemptions | 12 | £5.96m | £4.48m | 33.0% higher |

For the total over twelve months, forest MAE is £34.42m for subscriptions and
£41.11m for redemptions, versus £26.29m and £22.95m for the seasonal baseline.
Those are approximately 31% and 79% higher errors, respectively. Cumulative
errors are calculated from complete paths, not by adding monthly MAEs.

The forest nevertheless improves on Ridge at the displayed whole-fund monthly
leads. Sector performance is mixed: it beats the seasonal baseline for 10 of 22
sector/direction combinations at lead 6 and 11 of 22 at lead 12. Sector results
cannot be inferred from whole-fund scores.

## Uncertainty and interpretation

Empirical 80% ranges use only 12 separate calibration origins per lead. Coverage
can be substantially lower than 80% during evaluation. For forest twelve-month
totals, observed coverage is approximately 33% for subscriptions and 29% for
redemptions. These ranges understate uncertainty in this evaluation and must not
be treated as reliable 80% probabilities. Calibration should be improved before
operational use; a narrower range alone is not an improvement.

These results do not establish a universal useful cutoff. Random forest has not
earned replacement of the seasonal baseline across the fund. Retain it as an
experimental challenger, inspect sector-specific results and consider seasonal
baseline forecasts for the longer horizons. Further tuning on these reviewed
results would require additional temporal evaluation before making stronger claims.

## Artifacts and outlook

Generated files under `data/clean/monthly_forecasts` include per-origin predictions,
monthly metrics, cumulative path predictions and metrics, selected configurations,
source fingerprint, exact temporal splits and runtime (about 441 seconds on this run).

The latest complete month is September 2026. Consequently the monthly outlook
covers October 2026–September 2027 and excludes the later partial October data.
This differs explicitly from the existing daily next-month forecast cutoff.
The dashboard exposes this experiment under **Forecasts → Monthly model comparison**.
