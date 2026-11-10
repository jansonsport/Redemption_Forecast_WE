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

### Next extension: constrained random forest and monthly horizons

This section specifies the monthly experiment. The daily next-month forecasting specification above documents the existing approach; the long-range extension uses the monthly evaluation rules below.

#### Architecture and model inputs

- Retain the existing pipeline: DuckDB aggregates → offline forecasting → saved predictions and metrics → Streamlit. Scikit-learn is already a dependency; no new database service is required.
- Build complete monthly sector histories from cash-date aggregates, excluding incomplete months from training targets. Preserve source provenance and distinguish zero activity from missing coverage.
- Forecast monthly subscription and redemption amounts separately per sector, for lead months 1–12. Sum sector predictions for coherent whole-fund totals and derive net flow by subtraction. Direct fund models remain benchmarks.
- Build features from target calendar month, lead month, prior monthly flows and trailing averages. Every historical feature must be available at its forecast cutoff. Current valuation snapshots, future actual flows and later-known distribution schedules are excluded.
- Use direct prediction for each lead month rather than feeding earlier predictions back as observations. A training example is eligible only when its target outcome is available by the fitting cutoff.
- Start random forest experiments with 200 trees, maximum depth 3 or 5, minimum samples per leaf 6 or 12, and a fixed random seed. Select settings through chronological validation; record the seed, training window and complete configuration. Limited monthly sample counts constrain model complexity.
- Implementation defaults: a shared direct model learns all twelve leads using lead month as an input, separately for each sector and direction. Fit examples from at most 120 prior origin months, retaining only outcomes known at the fitting cutoff. Choose one forest configuration by validation MAE averaged across monthly leads 1–12.

#### Baselines and backtesting

- Retain recent averages, weekday averages and Ridge regression as comparison methods. Aggregate their daily predictions to the same monthly targets used for random forest. Include the corresponding month's actual total from the previous year as a seasonal baseline.
- Compare every method on identical historical cutoffs and target months, covering forecast origins across all calendar months. Report monthly errors separately for lead months 1–12, plus six- and twelve-month cumulative-total errors calculated from complete forecast paths.
- Select model settings using earlier validation forecasts scored on monthly errors at the relevant horizon. Daily one-month errors must not determine long-range model selection. Keep later evaluation outcomes out of tuning and feature construction.
- Report monthly MAE in GBP, signed bias, improvement relative to the seasonal baseline, sample counts, performance across historical periods and errors relative to typical flow sizes. Report zero-denominator metrics as unavailable.
- Identify overlapping forecast windows and their dependence; repeated target months are not independent observations. Previously reviewed backtests are retrospective evidence and must not be described as untouched tests.
- Use 12 monthly validation origins, then 12 monthly calibration origins, then 24 monthly evaluation origins. Leave sufficient gaps so that every twelve-month validation outcome is known before calibration starts and every calibration outcome is known before evaluation starts. Save exact boundaries in metadata. Twelve calibration cases per lead provide imprecise empirical intervals; report that limitation.
- The existing long-range experiment is preliminary: it compares a reduced candidate set, selects using daily one-month errors and spaces origins six months apart. Its poorer performance versus the seasonal baseline does not establish an absolute useful-horizon limit.
- Calibrate uncertainty separately by lead month using out-of-sample errors; evaluate interval coverage and width on later periods. Tree-to-tree variation alone is not a calibrated prediction interval. Aggregate residuals before deriving fund, net-flow or cumulative-total ranges.

#### Outputs, dashboard and release criteria

- Keep training offline through the command-line workflow. Cache reusable data preparation; save run duration, source fingerprint, feature version, model settings, cutoff, target month and lead month with predictions and metrics.
- Generate the experiment with `python -m fund_analytics monthly-forecast`; publish artifacts separately under `data/clean/monthly_forecasts`. The outlook starts after the latest complete month. Partial later observations remain available to the daily pipeline but are not silently treated as complete monthly totals.
- Preserve existing next-month views. After backtest review, add six- and twelve-month views using monthly points, forecast ranges, model/baseline comparisons and downloadable values. Clearly label insufficient history and weak predictive evidence.
- Recommend useful horizons from observed error, baseline improvement and stability by sector and target. Do not assume random forest wins or promise narrower ranges. Confirm model adoption and horizon presentation after reviewing results.
- Verify feature/cutoff isolation, eligible training targets, complete monthly targets, repeatability, nonnegative amount forecasts, sector-to-fund coherence and matching evaluation cases across methods. Measure runtime before widening tuning grids.

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
