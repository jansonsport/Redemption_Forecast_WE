# Development briefing: fund flow analytics

**For:** project presentation and review  
**Development date:** 9 November 2026  
**Data:** supplied synthetic charity fund files

## Project in one minute

This is a local Python dashboard for exploring subscriptions, redemptions, client and sector behaviour, fund income distributions, and fund value. The database is DuckDB; Streamlit and Plotly provide the dashboard. Imports and forecasts run as offline commands, and the dashboard reads prepared tables and saved forecast artifacts.

The project has two separate forecasting tracks:

1. **Daily next-month outlook:** cash-date predictions for each day of the next month. The whole-fund direct models selected Ridge for both flow types, with different history windows.
2. **Experimental monthly outlook:** monthly estimates through 12 leads, comparing constrained random forest against existing models and a same-month-last-year baseline. This is a research view, not a recommended production forecast.

Subscriptions are money invested into the fund; redemptions are money withdrawn. Income distributions are payments of fund income and remain separate from redemptions. Client AUM means the value of a client's investment at a recorded valuation date, not the total amount contributed.

## Work completed today

- Added a monthly forecasting module with constrained `RandomForestRegressor` models, monthly feature generation, chronological model selection, uncertainty calibration, evaluation, and saved artifacts.
- Compared the forest with recent-average, weekday-average, Ridge, and corresponding-month-last-year methods. The forest uses 200 trees, depth 3 or 5, minimum leaf size 6 or 12 and a fixed seed. Each sector and flow direction has a selected configuration.
- Added a CLI command: `venv/bin/python -m fund_analytics monthly-forecast`.
- Added a **Monthly model comparison** workspace under **Forecasts**. It displays six- and twelve-month paths, model choices, baselines, historical errors, empirical intervals and downloadable tables. Existing daily next-month forecasts remain available.
- Compared a constrained forest against the whole-fund daily next-month model using the same 24 selection and 12 evaluation months.
- Updated the PRD, setup documentation, pinned direct dependencies, and added forecasting invariant tests.
- The daily whole-fund model choices are Ridge, 12 months of training for subscriptions and 60 months for redemptions. The chosen sector models can differ.
- The primary data import is already built and reconciles to the source daily fund totals. No change to the database schema was needed for monthly modelling.

## Backtest design

The monthly data runs from November 2006 through September 2026 as complete months. October 2026 is partial in the supplied fund daily file, so monthly forecasts intentionally exclude it; the daily workflow still uses its later observations.

At each training cutoff, the forest sees only prior data. It predicts all twelve leads directly using calendar seasonality, lead month, recent monthly flows and the same month last year. It does not feed its earlier predictions back into later leads. The seasonal baseline repeats the matching month from the previous year.

The historical periods were divided chronologically:

| Purpose | Forecast origins | Count |
| --- | --- | ---: |
| Choose forest settings | Dec 2019–Nov 2020 | 12 |
| Calibrate error ranges | Nov 2021–Oct 2022 | 12 |
| Evaluate methods | Oct 2023–Sep 2025 | 24 |

The 12-month paths in each period finish before the next period starts. Test origins are six months apart, so their forecast windows overlap and share actual outcomes. The observations are therefore not independent. This is a retrospective evaluation, and earlier development has already examined parts of this history; present it as a historical backtest, not an untouched prospective trial.

## Main results

At whole-fund level, the seasonal baseline is hard to beat. Monthly figures below are MAE for the specific lead month, not the cumulative forecast-to-date.

| Target | Lead | Forest MAE | Seasonal baseline MAE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Subscriptions | 1 month | £5.55m | £5.16m | Forest 7.6% worse |
| Subscriptions | 6 months | £6.06m | £5.06m | Forest 19.8% worse |
| Subscriptions | 12 months | £5.95m | £5.78m | Forest 3.0% worse |
| Redemptions | 1 month | £4.08m | £4.56m | Forest 10.5% better |
| Redemptions | 6 months | £5.03m | £4.51m | Forest 11.5% worse |
| Redemptions | 12 months | £5.96m | £4.48m | Forest 33.0% worse |

For complete 12-month totals, forest MAE is **£34.42m subscriptions** and **£41.11m redemptions**, compared with **£26.29m** and **£22.95m** for the seasonal baseline. Forest total error is about 31% higher for subscriptions and 79% higher for redemptions. Cumulative errors are computed on the sum of each complete path; they are not sums of monthly MAEs.

The forest does improve on Ridge at the displayed whole-fund monthly leads, which shows why model comparisons need a strong baseline. At the sector/direction level, the forest beats the seasonal baseline in 14 of 22 combinations at lead 1, 10 of 22 at lead 6, and 11 of 22 at lead 12. Performance varies by sector, so the fund-wide result doesn't describe every sector.

The headline is not “random forest works” or “forecasts stop at month X.” The evidence says the forest often improves over Ridge but does not reliably improve over the simple seasonal repeat at whole-fund level. This backtest alone does not identify a universally useful horizon.

For the daily next-month experiment, random forest selected 60 months of history, depth 5 and leaf size 12 for both flow directions. Against the weekday benchmark, forest reduced daily MAE about 30% for subscriptions and 15% for redemptions; Ridge reduced it about 15% for both. On held-out months, subscription daily MAE was £754k for forest versus £905k for current Ridge; monthly-total MAE was nearly equal (£12.22m versus £12.24m). Redemption daily MAE was £914k versus £915k, while monthly-total MAE was £7.57m for forest versus £5.46m for Ridge. See [daily results](daily_forecast_results.md).

## Uncertainty and metrics

The nominal 80% monthly ranges use 12 calibration origins per lead. In evaluation, coverage varies and often falls below 80%. For twelve-month cumulative totals, observed forest coverage was about 33% for subscriptions and 29% for redemptions. Those intervals understate uncertainty and should not be presented as reliable 80% probability ranges. The dashboard now shows observed coverage and warns when it misses the nominal level.

Current reported metrics include:

- **MAE (mean absolute error):** typical error size in GBP; easy to explain, but strongly affected by large flow months.
- **Bias:** whether the model tends to overpredict or underpredict.
- **Baseline skill:** relative MAE reduction against same-month-last-year; positive means the candidate did better.
- **Normalised MAE:** absolute error relative to the typical size of the target flow.
- **Interval coverage and width:** whether ranges contain actual values, and how broad the ranges are.

Potential additions for discussion: median absolute error for robustness to spikes; a measure of error against total flow magnitude; sign accuracy for net flow; and client transaction frequency/probability metrics. If adding MAPE, avoid interpreting it around zero-flow months because percentage errors become undefined or misleading. A probabilistic score such as pinball loss is a future option after quantile forecasts are implemented and calibrated.

## Interesting discoveries and caveats to raise

- The seasonal baseline is a meaningful benchmark because it uses the same calendar month from the prior year. A complex model should demonstrate measurable improvement over it.
- Fund forecasts are made coherent by summing sector forecasts. Direct fund models are benchmarks; the displayed fund path comes from sectors.
- Forecasting an amount mixes two behaviours: whether a transaction happens and how large it is. A future two-part model could estimate transaction probability first, then conditional amount. Logistic regression is appropriate for that probability question, not directly for GBP amounts.
- The current monthly forest pools examples across forecast leads for each sector/direction. Many rows reuse the same realised month at different leads, so row count overstates independent information.
- Only 12 origins calibrate each lead's uncertainty. The poor observed interval coverage is a reason to improve calibration or avoid probability language, not to narrow ranges.
- The dataset is synthetic. Patterns and backtest metrics demonstrate the method and software; they do not establish real client behaviour.
- Classifications are static in the files. Current client AUM snapshots do not provide historical client balance trajectories.

## Suggested points for presentation or review

1. Is the same-month-last-year baseline the right business comparator, or should we also test trailing seasonal averages and a no-growth budget forecast?
2. Should long-range estimates be shown as experimental by default, given that the seasonal baseline beats the forest at whole-fund six- and twelve-month leads?
3. Should the useful horizon be selected for the whole fund only, or independently by sector and flow type? Sector results are mixed.
4. How should decision makers interpret a forecast range whose backtest coverage is below its stated 80% level?
5. Is monthly total accuracy more important than accuracy for each month along the path? The two can favour different models.
6. Would a two-part transaction-probability plus amount model answer the client-behaviour question better than predicting only aggregate GBP flows?
7. Which error cost matters more operationally: missing a large outflow, or forecasting too much cash need? That could motivate asymmetric scoring and quantile forecasts.
8. What information would be available in real use—client type changes, holidays, transaction instructions, scheduled distributions, or economic indicators—and when does it become known?

## Next steps

- Review the artifacts and sector-level results before deciding which horizons to expose as business forecasts.
- If long horizons remain in the dashboard, label the seasonal baseline and model evidence clearly; do not imply random forest is the winner.
- Improve uncertainty calibration and validate its coverage on new future observations as they arrive.
- Explore transaction occurrence and amount separately if client-level behaviour is a priority.
- Consider true seasonal patterns, zero-heavy or skewed target models and asymmetric outflow errors only with time-ordered validation.
- Continue timing query performance on the user's machine; no database indexes or partitions should be added without profiling.
- Refresh the current daily and monthly outputs after new source data is formally supplied. The initial v1 import remains limited to the supplied files.

## Where to look

- [Project requirements and architecture](PRD.md)
- [Monthly backtest results and split details](monthly_forecast_results.md)
- [Daily next-month model comparison](daily_forecast_results.md)
- Generated predictions, metrics, selected settings and metadata: `data/clean/monthly_forecasts/`
- Daily forecast choices: `data/clean/forecasts/models.csv`
- Run tests: `venv/bin/python -m pytest -q`
- Generate daily forecast: `venv/bin/python -m fund_analytics forecast`
- Compare daily next-month forest: `venv/bin/python -m fund_analytics daily-forest-backtest`
- Generate monthly comparison: `venv/bin/python -m fund_analytics monthly-forecast`
