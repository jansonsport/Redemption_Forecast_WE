# Fund Flow Analytics and Forecasting

Status: V1 and experimental monthly random forest extension implemented; model adoption subject to retrospective evaluation review  
Date: 9 November 2026

## 1. Purpose and success criteria

Build a local dashboard for analysing when clients subscribe to and withdraw from a single fund, how behaviour differs across client groups, and expected daily flows for the following month. Prioritise reliable database management, efficient SQL queries and reproducible analysis.

The primary user is an analyst exploring the supplied synthetic dataset on their own machine. Findings demonstrate analytical capabilities; they are not evidence about real clients.

Success means:

- Supplied data loads reproducibly into a validated, persistent database without duplicate records.
- Dashboard users can explore timing, client and sector behaviour without loading entire CSV files on each interaction.
- Separate subscription and redemption forecasts are available for every sector and the whole fund, with historical evaluation and uncertainty information.
- Sector forecasts sum exactly to the displayed fund forecast before presentation rounding.
- Query performance is measured and documented on the development machine.

## 2. Scope and source data

V1 uses only the four supplied files in `data/raw`. It includes a local dashboard, reusable SQL analysis, baseline forecasts, and historical forecast evaluation. The next extension adds constrained random forest experiments and monthly forecasts up to twelve months ahead, subject to backtest review. Arbitrary uploads, live feeds, authentication, shared hosting, client-level predictions and trading decisions remain outside scope. Logistic regression is deferred to a potential client transaction-probability feature.

The source brief is `docs/Gmail - Charity Fund Forecasting:Correlation.pdf`.

| Source | Observed rows | Meaning |
| --- | ---: | --- |
| synthetic_clients.csv | 11,103 | Client attributes, account dates and holding snapshots |
| synthetic_client_flows.csv | 901,088 | Signed subscriptions/redemptions, trade dates and cash dates |
| synthetic_income_distributions.csv | 462,696 | Paid and declared income distributions |
| synthetic_fund_daily.csv | 5,053 | Daily fund totals, valuations and reconciliation measures |

Observed fund history runs from 1 November 2006 to 30 October 2026; some trade dates precede that window and distribution payment dates extend beyond it. Preserve all source records and distinguish coverage dates from the last observed event date. The initial review found unique source keys, no orphan client references and exact agreement between event totals and the supplied daily fund totals over their common coverage.

## 3. Architecture and database management

Use Python 3.12 in the existing `venv`, DuckDB for persistent analytical storage, Streamlit for the local UI, Plotly for interactive charts, and pandas/scikit-learn where needed for modelling. Keep query, import and forecasting code independent of the dashboard so notebooks can reuse it.

Data flow: immutable raw CSVs → typed staging and validation → canonical tables → SQL summaries and forecast artifacts → dashboard.

### Storage and contracts

- Store the generated database at `data/clean/fund_analytics.duckdb`. Generated databases and caches are reproducible artifacts, not source files.
- Canonical tables: `clients`, `client_flows`, `income_distributions`, `fund_daily`, and `calendar`. Preserve the source grain and identifiers. Use primary keys, client foreign keys, explicit dates and `DECIMAL(20,2)` for GBP amounts; never infer money as floating point during import.
- Maintain `ingestion_runs` with source hashes, row counts, import timestamp, schema version, validation results and source coverage. An identical successful source fingerprint is a no-op on rerun.
- Import all four sources as one consistent snapshot. Validate staged records before committing replacement canonical tables and rebuilding dependent summaries. A failed import must leave the previous valid database available.
- Reject invalid types, duplicate identifiers, orphan references and impossible transaction dates. Report exceptions with source and record identifiers. Treat payments after account closure as reviewable warnings, since entitlement can precede closure.
- Keep trade and cash date semantics explicit. Cash dates drive money-movement forecasts; trade dates are available for behavioural exploration. Do not join daily fund totals to transaction rows and sum duplicated values.
- Preserve distributions separately from discretionary redemptions. Distinguish paid distributions from declared future obligations.
- Define reusable daily fund/sector aggregates and client weekly/monthly aggregates. Build large client comparison matrices only for the requested selection, not for every client pair.
- Use parameterised SQL filters. Filter and aggregate in DuckDB before returning results to Python. Cache small dashboard results using source fingerprint, query version and filter parameters; invalidate on successful rebuild.
- Refresh the database with the dashboard stopped in v1, avoiding cross-process write coordination. No background ingestion service is required.

### Correctness rules

Reconcile daily subscription/redemption amounts and counts on both date bases, and declared/paid distributions, against the daily fund file over shared coverage. Sum money with exact arithmetic.

Populate zero-flow periods only within each client's account lifetime and the dataset coverage. Historical client denominators must reflect accounts open in that period. Do not infer historical daily holdings from current snapshots or sporadic after-trade holdings. Client sector/type are treated as supplied static classifications because classification history is unavailable.

## 4. Dashboard requirements

Provide five pages with clear units, coverage dates, filter state and the latest successful import identifier. Show GBP subscriptions and redemption magnitudes separately, with net flow defined as subscriptions minus redemptions.

| Page | Required capabilities |
| --- | --- |
| Overview | Inflows, outflows, net flow, fund AUM, time series, sector contribution and transaction counts |
| Timing patterns | Weekday, position-in-month and month-of-year summaries; transaction frequency and average size; exposure-adjusted rates and observation counts |
| Client behaviour | Sector/type comparisons, selected-client history, transaction frequency/size, concentration, and weekly/monthly correlation heatmaps |
| Forecasts | Whole-fund or selected-sector next-month daily and monthly forecasts, subscriptions/redemptions/net flow, error ranges and historical evaluation |
| Data quality | Source counts, reconciliation results, warnings, coverage and freshness; query performance summary |

Historical analysis filters: date range, sector, client type, client and trade/cash date basis where meaningful. Fund AUM remains explicitly fund-wide when client filters are active; do not imply segment AUM history exists. Forecasts support fund/sector selection and a displayed cutoff, but not arbitrary client filters or trade-date forecasts.

Correlations default to monthly signed net flows and allow weekly frequency and separate subscriptions/redemptions. Restrict heatmaps to at most 50 explicitly selected clients. Use pairwise overlapping open-account periods including genuine zero-flow periods; require at least 12 monthly or 26 weekly observations. Display insufficient overlap or constant series as unavailable, not zero correlation. Show overlap counts and state that correlation does not establish causation.

Exports provide the displayed aggregate/forecast table as CSV. Empty selections, insufficient history and unavailable metrics produce explanatory messages rather than blank or misleading charts.

## 5. Forecasting specification

The initial daily specification is retained below for reproducibility. Current dashboard behavior is defined by **Unified dashboard adoption** and **Monthly comparison: history-selected models** later in this section.

### Targets and candidate methods

Forecast daily cash subscriptions and redemption magnitudes separately for each sector. Sum sector predictions to produce coherent fund totals, then derive net flow. Evaluate direct fund forecasts as a benchmark, but do not substitute them for coherent dashboard totals in v1.

Use a complete calendar over observed coverage, including genuine zero-event days. Forecast every calendar day of the next month and include weekday/weekend features; do not silently assume a holiday closure calendar. The first outlook is November 2026 from a 30 October 2026 observation cutoff. Do not silently fabricate observations for the unobserved October 31 tail.

Compare:

1. Mean daily amount over the training window.
2. Mean daily amount by weekday over the training window.
3. Ridge regression using one-hot weekday, month of year, day-of-month bucket (1–7, 8–14, 15–21, 22–end), and a linear time trend.

Evaluate training windows of 12, 36 and 60 months where available. For Ridge, standardise numeric features within training only and compare regularisation strengths 0.1, 1 and 10. Clip negative amount predictions to zero. Use the weekday baseline as fallback when a model cannot be fitted; all-zero histories produce zero predictions with an explicit explanation. Mark new/insufficient-history sectors when fewer than 12 months are available.

### Evaluation and selection

- Simulate forecasts at month boundaries, using only observations available at each cutoff. Use the 36 most recent complete months with full target coverage: the first 24 for model/window selection and error calibration, the final 12 as untouched evaluation months. Report actual month boundaries from source coverage.
- Select independently per sector and direction by mean daily absolute error across selection months; break ties by monthly total absolute error, then favour the simpler method and shorter window. Freeze selections before final evaluation.
- During final evaluation, refit the selected method on earlier available data at each cutoff without using later outcomes. Afterwards refit the same selected specification at the latest data cutoff for the next-month outlook.
- Report daily MAE in GBP, monthly total absolute error, signed bias, comparison to the recent-average baseline, and results by sector/direction. Do not promise that more complex models outperform the baselines.
- Produce indicative 80% empirical error ranges using selection-period out-of-sample residual quantiles. Calibrate daily and monthly-total ranges separately; monthly bounds must not be sums of daily bounds. For fund and net-flow ranges, aggregate predictions and actuals first, then calculate residuals, preserving cross-sector dependence. Clip lower bounds only for nonnegative amount targets. Show sample size and held-out coverage; describe ranges as indicative, not guaranteed confidence levels.
- Store predictions, evaluation metrics and model metadata with source fingerprint, cutoff, scope/sector, target, method, window, training dates and configuration. Store historical evaluations separately from the current outlook.

Avoid leakage from current client AUM, future transactions and future-known distribution schedules. Scheduled declared distributions may be displayed separately in the current outlook. Do not include them as historically known obligations in backtests because declaration availability timestamps are absent.

### Daily history-window experiment

This separate retrospective experiment extends daily next-month testing to the supplied history. Its split supersedes the original 36-month evaluation rule only for this experiment. Compare direct whole-fund subscriptions and redemptions; dashboard adoption and sector expansion require later review.

- Compare Ridge and constrained random forest under three approaches: 60-month rolling history with equal weights; expanding history with equal weights; expanding history with exponential recency weights. Use all observations available before each forecast, beginning November 2006 for expanding models.
- Reserve November 2006–October 2011 for initial training, November 2011–October 2013 for chronological validation, and November 2013–September 2026 for evaluation (155 monthly origins). Derive these boundaries from the first 60 complete months, next 24 months, and remaining complete months. Exclude incomplete October 2026. Refit at every month boundary using only earlier observations; require complete, unique daily coverage and finite nonnegative targets.
- Retain current calendar features, training-only trend scaling, and nonnegative predictions. Independently select settings per target/model/approach using validation daily MAE, then monthly-total MAE, then candidate order. Freeze settings during evaluation. Do not reuse settings selected on later historical periods.
- Ridge alpha candidates: 0.1, 1, 10. Forest depth/leaf candidates: (3,6), (3,12), (5,12), (5,24), with 200 trees and seed 42. Weighted candidates use half-lives of 1, 3, 5 years, considered in that order before model parameters. Weight each training day by `2 ** (-age_days / (365.25 * half_life_years))`, measured from the fitting cutoff, and normalize weights to mean 1 to preserve Ridge's relative regularization scale.
- Evaluate all six model/approach combinations and 12-month mean and weekday benchmarks on identical dates. Report overall and yearly daily MAE, monthly-total MAE, signed daily bias, benchmark improvements, and daily MAE divided by mean actual daily flow (equivalently total absolute error divided by total actual flow). Zero-denominator percentages are unavailable. Report paired differences between history approaches for each model and target.
- Run offline with `python -m fund_analytics history-window-backtest`; save predictions, monthly totals, metrics, paired comparisons, selected settings, exact training/split dates, source fingerprint, feature version, library version and runtime under `data/clean/history_window_backtest/`. Preserve the current forecast outputs.
- Publish a results document and a dated development-briefing update. These synthetic-data results are retrospective: portions of later history have already been examined. This experiment evaluates point forecasts; it does not recalibrate uncertainty ranges or extend monthly-horizon forecasts. Do not change the dashboard's active model automatically.

Results recorded 11 November 2026: expanding forests lead on daily MAE (subscriptions £642k, redemptions £625k), while rolling Ridge leads subscription monthly-total MAE (£5.72m) and weighted Ridge leads redemption monthly-total MAE (£4.84m, only slightly below rolling Ridge). Recency weighting does not consistently improve forecasts. Review [history-window results](history_window_results.md) before adoption; this experiment leaves the current dashboard model in place.

The follow-up diagnostic calibrates central empirical 80% intervals from November 2013–October 2015 and evaluates November 2015–September 2026 separately for daily values and monthly totals. Coverage was 64–70% daily and 32–52% monthly, below nominal for every approach. The monthly calibration has only 24 cases. These ranges are diagnostic and not reliable probability claims. Details: [history-window interval results](history_window_intervals.md).

### Fund-performance feature experiment

Compare calendar-only controls with calendar plus trailing 5-, 21-, and 63-valuation-day accumulation-price returns. Freeze returns at the preceding calendar month-end for every daily forecast and training target in its month; use only valuations available by that cutoff. Do not substitute within-month realized performance. Validate unique dated, finite positive prices. Omit early training months lacking 64 price observations from both arms; forecast origins with insufficient history fail explicitly. Scale added Ridge features using training-only statistics.

Use four previously reviewed configurations without retuning: expanding 200-tree forests (depth 5, leaf 12 subscriptions / leaf 24 redemptions, seed 42); rolling 60-month Ridge alpha 0.1 for subscriptions; expanding Ridge alpha 0.1 with one-year, mean-one recency weighting for redemptions. Evaluate November 2013–September 2026 (155 months) and separately November 2015–September 2026 (131 months). Calibrate each arm's daily and monthly-total 80% residual ranges independently on November 2013–October 2015; assess coverage only afterward. Compare matched controls, original history-experiment references and simple mean/weekday benchmarks on identical forecast dates. Report overall/yearly errors, bias, normalized error, paired changes, coverage and width. This is retrospective synthetic-data research; prior reviewed winners were chosen with knowledge of later performance.

Run `python -m fund_analytics performance-backtest`; save separate artifacts under `data/clean/performance_backtest/`, including predictions, performance-feature cutoffs, training eligibility, frozen configurations, interval diagnostics and source/feature versions. Publish findings in a results document and dated briefing update before any dashboard model adoption.

Performance experiment result (11 November 2026): the subscription forest's daily MAE improves only 0.47% against its matched control; redemption forest daily MAE is essentially unchanged and Ridge daily MAEs worsen slightly. Weighted Ridge redemption monthly-total MAE improves about 0.96%; other monthly totals worsen. Added features do not fix interval undercoverage. Do not adopt as a general improvement; see [performance-feature results](performance_forecast_results.md).

### Unified dashboard adoption

Following user review on 11 November 2026, Forecasts contains one **Monthly model comparison** workspace with **Daily next month** and **Monthly** resolution controls. Remove legacy and separate selected-model views. Keep the best Random Forest and best Ridge available, rather than choosing two configurations from the same family. Fund-performance features are excluded.

Daily scope is whole-fund cash flows. Use reviewed history-window daily MAE (monthly-total MAE and fixed history order break ties) to select the best history within each family/flow. Current daily winners are expanding forests, 200 trees, depth 5, seed 42, leaf 12 subscriptions / 24 redemptions, and rolling 60-month Ridge alpha 0.1 for both flows. `selected-forecast` publishes `selected-history-v2` model-specific daily outlooks, metrics, settings and net-flow ranges. Forecast the month following the latest cutoff; include observed partial-month history without inventing missing tail observations. Weekly points sum only days inside the forecast month, with partial boundary weeks labeled and no weekly ranges.

### Monthly comparison: history-selected models

#### Training and selection

- Preserve daily cash-flow aggregation, complete monthly targets, monthly forest features (lead, target-month seasonality, prior two monthly totals, trailing 3/12-month means, same month last year), calendar Ridge features and training-only trend scaling.
- Compare rolling 60-month, expanding and recency-weighted expanding history within both model families. Weighted half-lives: 1, 3, 5 years. Forest grid: depth 3/5, leaf 6/12, 200 trees, seed 42. Ridge alpha: 0.1, 1, 10. Clip negative amount predictions to zero.
- Each forest example's target must be known at the fitting cutoff. Rolling forests use the most recent 60 eligible target months; older observations may supply feature lookbacks. Expanding forests use all eligible examples. Weight forest examples by target-month age (`2 ** (-age_months/(12*half_life_years))`), normalized to mean one. Ridge uses training-day age and existing mean-one recency weights.
- Select separately for each family, subscription/redemption target, sector/direct-fund scope, and horizon 1/6/12. Rank validation cumulative-total MAE, then mean monthly MAE, then deterministic candidate order (rolling, expanding, weighted; half-life ascending; grid order). Freeze configurations before calibration/evaluation, then refit at each origin using available history only.
- Preserve twelve validation origins, twelve calibration origins and twenty-four evaluation origins, with sufficient gaps for all twelve-month targets to be observed before the following phase. Save exact dates. Shared outcomes across leads and overlapping evaluation paths are dependent.

#### Evaluation, aggregation and display

- Score all families and seasonal/mean/weekday benchmarks on identical origins and targets. Report individual monthly and cumulative-total errors, bias, normalized MAE, seasonal skill, interval coverage and width. Report zero-denominator ratios as unavailable.
- Preserve whole-fund and sector monthly forecasts. Whole-fund displayed paths sum independently selected sector paths; direct fund models are separate artifact comparisons. Derive net actuals/predictions before calibration. Net component settings are selected for subscription/redemption accuracy, not independently optimized for net error.
- Calibrate ranges separately by scope, target, family, selected horizon and lead; calculate cumulative bounds from cumulative residuals. Twelve calibration cases per lead are imprecise. Hide ranges by default; display measured coverage and sample limitations when enabled.
- Display family names and actual selected history/settings; do not relabel rolling models as expanding. Make one/six/twelve-month horizon choices explicit. Monthly outlooks start after the latest complete month, with later partial observations excluded, while daily outlooks use their later cutoff.
- `monthly-forecast` publishes `monthly-history-v2` under `data/clean/monthly_forecasts/`. Save candidate scores, configurations, source fingerprint, splits, runtime and version. Preserve original metrics/configurations/metadata as `previous_*` references on the same historical cases. Use two bounded offline workers and cache shared training examples/prediction paths. Refuse stale artifact formats or mismatched source imports instead of silently relabeling results.
- Update the README, dated development briefing and results report with actual outcomes, even if expanding models lose. Reviewed synthetic-data results do not establish prospective accuracy or real investor behavior.

## 6. Performance and acceptance checks

Provisional targets on the development machine: common uncached dashboard aggregate queries under one second and normal warm dashboard updates under two seconds. Record hardware, database size, query parameters and repeated-run median/p95 timings; report misses rather than treating targets as established results. Measure import time separately. Profile actual queries before adding indexes, extra materialisation or partitioning.

Acceptance checks must cover:

- Exact source row counts and reconciliation; duplicate/orphan/type rejection; warning handling; no-op reimports; rollback after failed import.
- Correct signs, cash/trade date selection, paid/declared separation, account-lifetime zero filling and filter totals.
- Correlation overlap thresholds, constant series, empty selections and population denominators.
- Forecast cutoff isolation, repeatability, nonnegative amount predictions, complete target dates, fund/sector coherence, correct monthly uncertainty calculation and frozen held-out model selection.
- Dashboard navigation, filters, exports, clear unavailable states and measured performance on supplied data.

Use focused automated checks for data and forecasting invariants and a local UI smoke check. A reproducible setup guide must document dependency installation, database import, forecast generation and dashboard launch. Pin resolved dependencies during implementation.

## 7. Delivery and subagent coordination

1. Foundation: repository structure, typed schema, import command, validation and source manifest.
2. Analysis: shared SQL contracts, daily and client-period summaries, focused correctness checks and initial query benchmarks.
3. Forecasting: candidate methods, rolling evaluation, coherent outputs and persisted results.
4. Dashboard: the five views, exports, cache invalidation and forecast/error presentation.
5. Integration: acceptance checks, performance report, setup documentation and known limitations.

The coordinating agent owns shared contracts and integration. Once schema and query contracts are fixed, delegate bounded work to database, analytics/forecasting and dashboard subagents. Assign separate files/modules, agree output contracts before parallel edits, and review integrated changes centrally. Subagent use is authorised for implementation; deployment here means assigning development work, not publishing a hosted service.

## 8. References

- Local project brief and supplied synthetic CSVs.
- [DuckDB persistent storage overview](https://duckdb.org/faq)
- [DuckDB workload tuning](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads)
- [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching)

The stack, local-only operation, supplied-files-only scope, baseline forecasts and fund/sector coverage reflect the planning discussion. Detailed thresholds, model grids and performance targets above are implementation defaults and should be changed explicitly if subsequent evidence warrants it.
