# Prompt: add the "Segments & stress" study to this dashboard

> Copy everything below the line into Claude Code (or another coding agent) opened in **your partner's repository**, after committing the `handoff/segments_stress/` folder to that repository. Replace the two `<…>` placeholders first.

---

You are working in an existing fund-dashboard repository. My teammate built this UI and its forecasting features; their product requirements are in `<PATH TO PARTNER PRD, e.g. docs/PRD.md>`. I ran a separate client-relationship study (client segmentation, segment behaviour in stress periods, concentration and liquidity stress tests). I want it added to **their** UI as a new section called **"Segments & stress"**, without breaking or restyling anything they built.

Everything you need from my side is in `handoff/segments_stress/` (committed on branch `<BRANCH NAME>`):

| Folder | Contents | How to use it |
|---|---|---|
| `analysis/segment_stress.py` | **Source of truth** for every number: config-driven pipeline (data checks, client features, k-means segments, cohesion, segments vs sector comparison, stress flags, segment flow correlation, concentration, stress tests). `run(cfg)` returns all results and writes `11_*.csv`. Depends on `analysis_helpers.py`, pandas, numpy, scikit-learn. | Run it to regenerate data; do not re-implement the statistics. |
| `analysis/make_11_notebook.py`, `11_segments_correlation_stress.ipynb/.html` | The executed notebook and its HTML export. | Read for methodology and wording; link the HTML from the new section. |
| `data_outputs/` | All `11_*.csv` tables, `11_report.json`, and `segments-data.js` (one JSON object assigned to `window.SEGMENTS`, about 3 MB). | Ready-made data contract. You may load it as-is or convert it to JSON for this app's data layer. |
| `reference_ui/segments.js` | My working implementation (vanilla JS, about 60 KB) of all seven tabs. | **Behavioural spec**: replicate content, controls and logic in this repo's framework and components. Do not paste it in wholesale unless the app is vanilla JS. |
| `reference_ui/stress-engine.js` | Pure functions that recompute every stress-dependent result in the browser when the user edits stress periods: `flags`, `correlations`, `lift`, `replay`, `tailHist`. | Port as-is (or to TypeScript). It must keep **exact parity** with the notebook (see tests). |
| `reference_ui/build_segments.py` | Builds `segments-data.js` from `segment_stress.run()`. Paths assume my repo layout (`ROOT/notebooks`, `ROOT/synthetic_*.csv`). | Adapt the paths, or the data step of this repo, to produce the same object. |
| `reference_ui/check_segments.cjs` | My test harness: render checks, parity of the live stress engine with the notebook, query-tool counts, reverse-stress agreement. | Port the assertions to this repo's test framework. |
| `reference_ui/charts.js`, `filters.js` | My chart and filter helpers. | Use only if this repo has no chart or filter components; otherwise use theirs. |

## Step 0 · Understand before changing anything
1. Read the PRD and the repo: framework, routing and navigation, design system and tokens, chart library, data loading, test setup, build and deploy commands.
2. Check that both projects use the **same source data**: `synthetic_clients.csv`, `synthetic_client_flows.csv`, `synthetic_fund_daily.csv`, `synthetic_income_distributions.csv`. If file names, columns or the data cutoff differ, map them through the `CONFIG` dictionary at the top of `segment_stress.py` (that is what it is for) and say exactly what you changed.
3. Write a short integration plan and show it to me **before coding**. It should cover: where the new section lives in their navigation, which of their components you will reuse, how the data gets in (static JSON from a build step vs API), and any conflicts with their PRD or existing features. Wait for my approval.

## What to build: one section, seven tabs, told as a story
Follow `reference_ui/segments.js` for exact content, wording and controls. In summary:

1. **Overview.** A clickable 7-step process diagram (collect & check → measure 9 habits → set aside rare traders → k-means → test in stress → stress the fund → explore clients), each step showing its key number. Below it, 4 headline cards and 5 plain-language findings with links to the evidence tabs.
2. **Data (brief).** Three columns: tables and fields taken from the database; what was missing and how it was derived (units = balance ÷ price; daily holdings = units × price; stress days from price drawdown and named episodes; transfers between accounts not identifiable; no yield; no portfolio holdings, so buffers are illustrative); and the data checks as a ✓ list. Full tables go behind a "details" toggle.
3. **Finding segments.** Six steps:
   1. the nine habits;
   2. rare traders set aside, with an eligibility-sensitivity table (min-years selector);
   3. k-means explained in four simple steps, plus a scatter **map of 2,500 clients coloured by segment** with numbered centres and a highlight-segment selector (data in `pca`);
   4. choosing k (silhouette and bootstrap stability chart);
   5. **how alike each group is**: verdict badge (Tight group / Loose group / Label only / Not clustered), share that clearly belong, spread, within-group flow correlation vs a random pair;
   6. out-of-time comparison of segments vs **sector, client type and region** (cards plus grouped bars per habit, and the "use both" recommendation).
4. **The segments.** Plain-English card per segment (top 3 differences vs a typical client, the "for liquidity" implication, verdict badge, and a "See these clients →" button that opens tab 6 pre-filtered). Also: a compare-segments chart with a measure selector, AUM over time (share/GBP toggle), monthly net flow with segment toggles, and a sector-mix heat table. Add this sentence near the top: *"Segments describe client behaviour; they did not improve forecasts (Notebook 09), so treat the largest clients individually."*
5. **Segments in stress.**
   - Seasonality chart, with a measure selector.
   - **Editable stress-period definition**: price trigger (off or 5–25% below the one-year high), table of periods (tick, rename, redate, remove), add-period form, reset to default. Saved in localStorage. A banner on tabs 4–5 shows the current definition.
   - Drawdown chart with shaded stress days and **three strips** (in use / from your periods / from price trigger), plus overlap counts and an "Also price-triggered" column, so unticking a period visibly changes the result.
   - Per-segment redemption rate on stress vs calm days, raw and **month-matched**, with a one-sentence answer.
   - **"Do the segments pull money out at the same time?"**: one grouped bar chart (actual flows vs after removing calendar habits, calm vs stress), a daily/weekly selector, a computed plain-language answer, and the full pair heatmap behind a toggle. Do **not** reintroduce the old "four-way comparison" wording.
   - Worst 5- and 20-day outflows inside each active period.
6. **Concentration & stress tests.**
   - Top-N concentration over time (N chips) and the top-100 holders table with segment filter.
   - Stress settings: buffer slider with presets including MMF reference floors, a pay-payables-first toggle, and a fire-sale haircut.
   - **Reverse stress test** explained in two sentences (normal test: "if X happens, is there enough cash?"; reverse test: "how bad before cash runs out?"). Show the largest clients in payment order, green if the buffer can pay them and red once it is empty, plus a sentence naming the client who empties the buffer.
   - Scenario families as filter chips with one-line definitions, a scenario outflow chart (covered vs exceeds), and a details table with second-round asset sales.
7. **Segment clients (query tool).** Segment chips; background filters (sector, country, AUM band, account status, client type, region, unit type); AND conditions on any behaviour field (percent fields entered as %); 4 quick queries; search; sort field and direction; 25 per page; CSV export. Clicking a row opens a client profile vs the segment's typical client. If this repo has a client-correlation or peer feature, link to it; otherwise omit the link.

## Rules
- **Reuse their design system**: colours, typography, cards, tabs, buttons and chart library. Keep my chart semantics: one fixed colour per segment everywhere; "Rare traders" grey; green = covered, red = shortfall; axis titles with units; tooltips; legends whenever there are 2 or more series; empty states that explain why.
- **Keep the honesty notes visible.** Data is synthetic. Segments are descriptive, not predictive. Stress scenarios are illustrative exposures, not probabilities. There are no portfolio holdings, so buffers are illustrative. MMF floors are reference only because this fund is not an MMF. CIs/p-values (where shown) are optimistic. The FCA June 2026 position is to keep current UK MMF minimums; further guidance is pending.
- Don't change their forecasting features, data or routes, except to add the navigation entry and the shared loaders you need. If something they already have duplicates my content (for example a concentration chart), say so in the plan and ask whether to link or merge.
- Performance: the data object is about 3 MB. Lazy-load it when the section opens. The live stress engine must stay under about 100 ms per recompute (mine takes about 60 ms).
- Accessibility: keyboard-reachable controls and tabs, aria-pressed on chips, focusable chart marks with tooltips, and no colour-only meaning.
- Work on a new branch, make small commits, and open a pull request with a description. Do not push to main.

## Tests and acceptance criteria (must all pass)
1. **Pipeline:** `segment_stress.run()` executes on this repo's data with every data check passing (`11_report.json` → `checks_passed: true`).
2. **Headline numbers match my results** if the data are the same (otherwise explain each difference):

   | Measure | Expected value |
   |---|---|
   | k | 7 |
   | Rare traders | 2,834 clients, 27.3% of AUM |
   | Stress days (default definition) | 781 |
   | Bootstrap ARI median | ≈ 0.72 |
   | Top-10 share at 2026-10-30 | 13.4% |
   | Top-100 share at 2026-10-30 | 40.7% |
   | 10% buffer (after £11.9m payables) | covers the 6 largest clients; the 7th is C024578, Larchcroft Family Foundation |
   | Mean daily segment correlation, raw | 0.11 (0.12 in stress) |
   | Mean daily segment correlation, deseasonalised | 0.05 |
   | Out-of-time η², mean across habits | segments 31%, sector 23%, client type 24%, region ≈ 0% |
3. **Stress-engine parity:** with the default definition (price trigger 10%, four named episodes), the ported engine reproduces the notebook to within 1e-5 for:
   - stress flags;
   - every correlation summary and matrix entry;
   - period replays and tail scenarios (to within £1);
   - value lift and month-matched lift.

   Unticking "Euro-area debt crisis" must change stress days from 781 to 716.
4. **Query tool:** "Big rare traders" (Rare traders, holding ≥ £5m) returns 19 clients holding about £382m, and the count equals a direct count over the data.
5. **Render tests:** every tab and control renders with no NaN, undefined or Infinity text. The existing test suite still passes.
6. **Visual QA:** screenshots at desktop and phone width show no clipped labels or horizontal page scroll, and the new section matches the existing look.

## Deliverables
- The new "Segments & stress" section integrated into their navigation.
- The data build step (or static data) and the ported `stress-engine`.
- Tests for points 1–5 above.
- The notebook HTML linked from the section.
- A short README section on how to regenerate the data.
- A PR description listing what was added, what was adapted to their stack, any number that differs from the table above and why, and anything left out.
