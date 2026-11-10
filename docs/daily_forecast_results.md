# Daily next-month forecast comparison

**Run date:** 9 November 2026 · **Scope:** whole fund · **Data:** supplied synthetic history

## Evaluation setup

The comparison uses the daily forecast's chronological split: October 2023–September 2025 for model selection, then October 2025–September 2026 for held-out evaluation. At each month boundary, each method forecasts the following month using only earlier data. All results cover the same 365 days and 12 monthly totals.

Benchmarks are the 12-month mean daily amount and weekday-specific 12-month mean. Ridge uses the settings selected for the current whole-fund forecast. The constrained forest configuration is selected on the first 24 months using daily MAE, with monthly total MAE as the tie-breaker, then held fixed for evaluation.

| Flow | Method | Daily MAE | Monthly-total MAE |
|---|---|---:|---:|
| Subscriptions | 12-month mean benchmark | £1.287m | £16.058m |
| Subscriptions | 12-month weekday benchmark | £1.070m | £15.811m |
| Subscriptions | Current Ridge (12 months, alpha 1.0) | £905k | £12.239m |
| Subscriptions | Constrained random forest | £754k | £12.219m |
| Redemptions | 12-month mean benchmark | £1.275m | £22.831m |
| Redemptions | 12-month weekday benchmark | £1.071m | £22.142m |
| Redemptions | Current Ridge (60 months, alpha 0.1) | £915k | £5.457m |
| Redemptions | Constrained random forest | £914k | £7.566m |

Lower MAE means smaller absolute errors. Versus the weekday benchmark, the forest reduced daily MAE by about 30% for subscriptions and 15% for redemptions. Ridge reduced it by about 15% for both. Versus the 12-month mean, forest reduced daily MAE by 41% and 28%, respectively; Ridge reduced it by 30% and 28%.

For monthly totals, forest and Ridge were nearly tied for subscriptions. Ridge was better for redemptions: £5.46m MAE compared with £7.57m for the forest. Both models beat the simple monthly-total benchmarks in this evaluation period.

## Forest settings and interpretation

The selected forest used 200 trees, depth 5, minimum leaf size 12, seed 42, and a 60-month training window for each direction. Whole-fund Ridge used 12 months and alpha 1.0 for subscriptions, and 60 months and alpha 0.1 for redemptions.

The forest is a promising challenger for daily subscription errors. It did not improve monthly redemption totals over Ridge, so this one held-out synthetic year does not justify replacing Ridge for both flow types. The experiment reports point errors and does not provide calibrated forest uncertainty ranges. Track subsequent outcomes before selecting a permanent model.

Detailed outputs are in `data/clean/daily_forest_backtest.csv`, `data/clean/daily_forest_backtest_detail.csv` and `data/clean/daily_forest_backtest_models.csv`. Reproduce with `venv/bin/python -m fund_analytics daily-forest-backtest`.
