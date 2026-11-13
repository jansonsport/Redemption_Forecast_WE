# One-year CSV import examples

These fictional datasets cover every calendar day of 2025: 12 clients, 288 transactions,
48 paid quarterly distributions and 365 daily fund records. All use GBP. The amounts,
client snapshots and daily fund totals reconcile; generation uses decimal arithmetic
and no randomness. They are intended for descriptive UI checks, not model training.

`complete_2025` supplies all original columns. `missing_sector_2025` differs only by
removing the `client_sector` column and its values from the clients file. Source files
in `data/raw` are not modified.

Regenerate both examples from the project root:

```sh
venv/bin/python examples/csv_import/generate.py
```

Stop the dashboard before importing. Use a separate database so the main dataset stays
untouched. Run from the project root:

```sh
FUND_ANALYTICS_DB=/tmp/fund_missing_sector.duckdb venv/bin/python -m fund_analytics ingest --raw-dir examples/csv_import/missing_sector_2025
FUND_ANALYTICS_DB=/tmp/fund_missing_sector.duckdb venv/bin/python -m streamlit run app.py
```

For the control dataset, stop that dashboard and run:

```sh
FUND_ANALYTICS_DB=/tmp/fund_complete_2025.duckdb venv/bin/python -m fund_analytics ingest --raw-dir examples/csv_import/complete_2025
FUND_ANALYTICS_DB=/tmp/fund_complete_2025.duckdb venv/bin/python -m streamlit run app.py
```

In the missing-sector dashboard, expect a notice naming `clients.client_sector` beside
unavailable sector filtering and charts. Flow totals, client-type comparisons, individual
client selection, timing summaries and fund AUM should remain available where applicable.
Compare these supported results with the complete dataset: removing sectors must not
change their values. Forecasting is unavailable with missing sectors; the complete dataset
also has insufficient history for the existing forecasting evaluation protocol. No
forecast or segment artifacts are supplied with these examples.
