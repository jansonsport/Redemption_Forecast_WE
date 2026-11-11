"""Local fund-flow workbench. Run: streamlit run app.py."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fund_analytics import db

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'data/clean/fund_analytics.duckdb'
COLORS = {'subscriptions': '#16877c', 'redemptions': '#d26b45', 'net_flow': '#5773b7'}
LABELS = {'subscriptions': 'Subscriptions', 'redemptions': 'Redemptions', 'net_flow': 'Net flow'}
QUERY_VERSION = 'v1.0'

st.set_page_config(page_title='Fund Flow | Analytics', page_icon='◈', layout='wide')
st.markdown('''<style>
/* Keep text legible across Streamlit's light/dark system themes. */
html, body, [data-testid="stAppViewContainer"] {font-family:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
.stApp {background:#f7f8fa;color:#17212b;}
.stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] {color:#263746 !important;}
[data-testid="stSidebar"] {background:#edf1f4;}
[data-testid="stSidebar"] * {color:#263746 !important;}
h1,h2,h3 {color:#14232f !important;letter-spacing:-.025em;}
.block-container {padding-top:2rem;max-width:1600px;}
[data-testid="stMetric"] {background:#fff;border:1px solid #d5dfe5;border-radius:12px;padding:18px;}
[data-testid="stMetricLabel"] {color:#425767 !important;}
[data-testid="stMetricValue"] {color:#14232f !important;}
.eyebrow {color:#08766b;font-size:.85rem;font-weight:700;letter-spacing:.1em;}
/* Streamlit tables use a canvas grid whose header colors do not inherit page text. */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataEditor"] [role="columnheader"] {color:#172b3a !important;font-weight:700 !important;opacity:1 !important;}
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataEditor"] [role="gridcell"] {color:#263746 !important;}
</style>''', unsafe_allow_html=True)


def money(value):
    if pd.isna(value):
        return '—'
    sign = '−' if value < 0 else ''
    value = abs(float(value))
    if value >= 1e9:
        return f'{sign}£{value / 1e9:,.2f}bn'
    if value >= 1e6:
        return f'{sign}£{value / 1e6:,.2f}m'
    return f'{sign}£{value:,.0f}'


@st.cache_data(show_spinner=False, max_entries=160)
def query(fingerprint, version, sql, params=()):
    """Only cache selected results; source identity and SQL are part of the key."""
    with db.connect() as connection:
        return connection.execute(sql, list(params)).fetchdf()


def table_export(frame, name):
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button('Download this table · CSV', frame.to_csv(index=False).encode(),
                       f'{name}.csv', 'text/csv', key=f'download_{name}')


def chart(fig):
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=10, t=25, b=0),
                      legend_title_text='', font=dict(family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
                      color='#263746', size=14), hovermode='x unified')
    fig.update_xaxes(title_font=dict(color='#263746', size=14),
                     tickfont=dict(color='#425767', size=12),
                     gridcolor='#d6dee5', zerolinecolor='#aab7c2')
    fig.update_yaxes(title_font=dict(color='#263746', size=14),
                     tickfont=dict(color='#425767', size=12),
                     gridcolor='#d6dee5', zerolinecolor='#aab7c2')
    fig.update_layout(legend=dict(font=dict(color='#263746', size=13)))
    st.plotly_chart(fig, use_container_width=True)


def lines(frame, x='date', columns=('subscriptions', 'redemptions'), title=None):
    chart(px.line(frame, x=x, y=list(columns), color_discrete_map=COLORS,
                  labels={'value': 'GBP', 'variable': 'Flow', x: x.title()}, title=title))


@st.cache_data(show_spinner=False, max_entries=2)
def monthly_artifacts(artifact_identity):
    path = ROOT / 'data/clean/monthly_forecasts'
    meta = json.loads((path/'metadata.json').read_text())
    if meta.get('version') != 'monthly-history-v2':
        raise ValueError('Regenerate monthly forecasts for the current comparison format.')
    tables = {name: pd.read_csv(path/f'{name}.csv') for name in
              ('detail', 'metrics', 'cumulative', 'cumulative_metrics', 'models')}
    if any('horizon' not in frame for frame in tables.values()):
        raise ValueError('Monthly forecast artifacts do not contain horizon selections.')
    tables['metadata'] = meta
    return tables


@st.cache_data(show_spinner=False, max_entries=2)
def selected_artifacts(artifact_identity):
    path = ROOT / 'data/clean/selected_forecasts'
    meta = json.loads((path/'metadata.json').read_text())
    if meta.get('version') != 'selected-history-v2':
        raise ValueError('Regenerate daily forecasts for the current comparison format.')
    data = {name: pd.read_csv(path/f'{name}.csv') for name in ('daily', 'metrics', 'models')}
    if any('model' not in frame for frame in data.values()):
        raise ValueError('Daily forecast artifacts do not contain model selections.')
    data['metadata'] = meta
    return data


def artifact_stamp(path):
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                 for p in sorted(path.glob('*')) if p.is_file())


def daily_comparison_view(source_fingerprint, observation_cutoff):
    try:
        data = selected_artifacts(artifact_stamp(ROOT / 'data/clean/selected_forecasts'))
    except (FileNotFoundError, OSError, ValueError, KeyError):
        st.info('Generate the newer daily model comparison before opening this view.')
        st.code('venv/bin/python -m fund_analytics selected-forecast', language='bash')
        return
    meta = data['metadata']
    if meta.get('source_fingerprint') != source_fingerprint or meta.get('cutoff') != str(observation_cutoff):
        st.warning('The daily outlook belongs to an earlier import or cutoff. Regenerate it.')
        st.code('venv/bin/python -m fund_analytics selected-forecast', language='bash')
        return
    st.caption(f"Whole-fund cash forecasts · {meta['forecast_month']} · observations through {meta['cutoff']}. "
               'Daily sector forecasts are not part of this comparison.')
    target = st.radio('Daily flow target', list(LABELS), format_func=LABELS.get, horizontal=True)
    labels = {'random_forest': 'Random Forest', 'ridge': 'Ridge regression'}
    model = st.selectbox('Daily model', list(labels), format_func=labels.get)
    frame = data['daily'][(data['daily'].target == target) & (data['daily'].model == model)].copy()
    metrics = data['metrics'][(data['metrics'].objective == 'daily') &
                              (data['metrics'].target == target) & (data['metrics'].model == model)]
    if frame.empty or metrics.empty:
        st.info('No daily forecast is available for this model and flow.')
        return
    metric = metrics.iloc[0]
    targets = ['subscriptions', 'redemptions'] if target == 'net_flow' else [target]
    settings = data['models'][(data['models'].objective == 'daily') &
                              (data['models'].model == model) & data['models'].target.isin(targets)]
    histories = {'rolling': 'five-year rolling history', 'expanding': 'expanding history',
                 'weighted': 'recency-weighted expanding history'}
    names = ', '.join(histories.get(h, h) for h in settings.history.dropna().unique())
    st.subheader(labels[model])
    st.caption(f'Selected by daily error · {names} · calendar features. '
               'Net flows subtract the selected subscription and redemption forecasts.')
    a, b, c = st.columns(3)
    a.metric('Sum of daily estimates', money(frame.prediction.sum()))
    b.metric('Historical daily MAE', money(metric.assessment_mae_gbp))
    c.metric('Observed range coverage', f'{metric.coverage:.1%}')
    st.caption(f"Historical assessment: {meta['assessment_months'][0]} to {meta['assessment_months'][-1]} · "
               f"{int(metric.assessment_cases):,} outcomes. Retrospective synthetic-data evidence.")
    frequency = st.radio('Daily display', ['Daily', 'Weekly'], horizontal=True)
    show_ranges = st.checkbox('Show indicative 80% range diagnostics', value=False, key='daily_ranges')
    frame['date'] = pd.to_datetime(frame.date)
    if frequency == 'Weekly':
        frame['date'] = frame.date.dt.to_period('W-SUN').dt.start_time
        frame = frame.groupby('date', as_index=False).prediction.sum()
        st.caption('Weekly values sum forecast days inside the month; boundary weeks may be partial. Weekly ranges are unavailable.')
    elif show_ranges:
        st.warning(f'These ranges captured {metric.coverage:.1%} of historical outcomes against an 80% target. '
                   'Observed coverage is not a guarantee for future forecasts.')
        st.caption(f"Calibrated on {int(metric.calibration_cases):,} earlier errors from "
                   f"{meta['calibration_months'][0]} to {meta['calibration_months'][-1]}.")
    fig = go.Figure()
    if show_ranges and frequency == 'Daily':
        fig.add_trace(go.Scatter(x=frame.date, y=frame.upper, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=frame.date, y=frame.lower, line=dict(width=0), fill='tonexty',
                                fillcolor='rgba(22,135,124,.15)', name='Indicative 80% range'))
    fig.add_trace(go.Scatter(x=frame.date, y=frame.prediction, mode='lines+markers',
                            line=dict(color=COLORS[target], width=3), name=labels[model]))
    fig.update_yaxes(title='GBP')
    chart(fig)
    st.subheader('Daily model comparison')
    table_export(data['metrics'][(data['metrics'].objective == 'daily') &
                                 (data['metrics'].target == target)], 'daily_model_metrics')
    with st.expander('Daily forecast values, model settings and provenance'):
        export = frame if show_ranges else frame.drop(columns=['lower', 'upper'], errors='ignore')
        table_export(export, 'daily_forecast_values')
        st.dataframe(settings, hide_index=True)
        st.json(meta)


def monthly_view(source_fingerprint, sectors, observation_cutoff):
    st.subheader('Monthly model comparison')
    resolution = st.radio('Forecast resolution', ['Daily next month', 'Monthly'], horizontal=True)
    if resolution == 'Daily next month':
        daily_comparison_view(source_fingerprint, observation_cutoff)
        return
    st.caption('Monthly cash-flow paths · models selected by cumulative-total error · retrospective research')
    try:
        data = monthly_artifacts(artifact_stamp(ROOT / 'data/clean/monthly_forecasts'))
    except (FileNotFoundError, OSError, ValueError, KeyError):
        st.info('Generate the updated monthly comparison before opening this view.')
        st.code('venv/bin/python -m fund_analytics monthly-forecast', language='bash')
        return
    meta = data['metadata']
    if meta.get('source_fingerprint') != source_fingerprint or meta.get('observation_cutoff') != str(observation_cutoff):
        st.warning('Monthly results belong to an older import or observation cutoff. Regenerate them.')
        st.code('venv/bin/python -m fund_analytics monthly-forecast', language='bash')
        return
    scope = st.selectbox('Monthly forecast scope', ['Whole fund'] + sectors)
    scope_type = 'fund' if scope == 'Whole fund' else 'sector'
    target = st.radio('Monthly flow target', list(LABELS), format_func=LABELS.get, horizontal=True)
    horizon = st.radio('Months ahead', [1, 6, 12], horizontal=True)
    labels = {'random_forest': 'Random Forest', 'ridge': 'Ridge regression',
              'seasonal_naive': 'Same month last year', 'mean_12': 'Recent average',
              'weekday_12': 'Weekday average'}
    model = st.selectbox('Model', ['random_forest', 'ridge'], format_func=labels.get)
    benchmark = st.selectbox('Benchmark', ['seasonal_naive', 'mean_12', 'weekday_12'], format_func=labels.get)
    show_ranges = st.checkbox('Show indicative 80% range diagnostics', value=False, key='monthly_ranges')

    def selected(frame):
        return frame[(frame.scope == scope_type) & (frame.sector == scope) &
                     (frame.target == target) & (frame.horizon == horizon) &
                     (frame.lead_month <= horizon)].copy()

    outlook = selected(data['detail'])
    outlook = outlook[(outlook.phase == 'outlook') & (outlook.model == model)].sort_values('lead_month')
    if outlook.empty:
        st.info('No monthly outlook is available for this selection.')
        return
    st.caption(f"Forecast origin: end of {meta['forecast_origin']} · observed data through {meta['observation_cutoff']}. "
               'Only complete months enter these models; later partial-month observations are excluded.')
    component_targets = ['subscriptions', 'redemptions'] if target == 'net_flow' else [target]
    models = data['models']
    mask = models.target.isin(component_targets) & (models.scope == 'sector') & (models.model == model) & (models.horizon == horizon)
    if scope_type == 'sector':
        mask &= models.sector == scope
    settings = models[mask]
    if not settings.empty:
        descriptions = {'rolling': 'Five-year rolling', 'expanding': 'Expanding', 'weighted': 'Recency-weighted expanding'}
        counts = settings.history.value_counts()
        summary = '; '.join(f'{descriptions.get(name, name)}: {count}' for name, count in counts.items())
        st.caption(f'Selected configurations · {summary}. Settings are shown below.')
    if scope_type == 'fund':
        st.caption('Whole-fund paths sum independently selected sector forecasts. Selection is made within each model family.')
    outlook['date'] = pd.to_datetime(outlook.target_month)
    fig = go.Figure()
    if show_ranges:
        if len(outlook) == 1:
            fig.add_trace(go.Scatter(x=[outlook.date.iloc[0]] * 2,
                y=[outlook.lower.iloc[0], outlook.upper.iloc[0]], mode='lines',
                line=dict(width=5), name='Indicative 80% monthly range'))
        else:
            fig.add_trace(go.Scatter(x=outlook.date, y=outlook.upper, line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=outlook.date, y=outlook.lower, line=dict(width=0), fill='tonexty',
                                    fillcolor='rgba(22,135,124,.15)', name='Indicative 80% monthly range'))
    fig.add_trace(go.Scatter(x=outlook.date, y=outlook.prediction, mode='lines+markers', name=labels[model]))
    comparison = selected(data['detail'])
    comparison = comparison[(comparison.phase == 'outlook') & (comparison.model == benchmark)].sort_values('lead_month')
    fig.add_trace(go.Scatter(x=pd.to_datetime(comparison.target_month), y=comparison.prediction,
                            line=dict(dash='dot'), name=labels[benchmark]))
    fig.update_yaxes(title='Monthly total · GBP')
    chart(fig)
    totals = selected(data['cumulative'])
    totals = totals[(totals.phase == 'outlook') & (totals.model == model) & (totals.lead_month == horizon)]
    total_metrics = selected(data['cumulative_metrics'])
    chosen_total = total_metrics[(total_metrics.model == model) & (total_metrics.lead_month == horizon)]
    if not totals.empty:
        row = totals.iloc[0]
        cols = st.columns(3)
        cols[0].metric(f'{horizon}-month total', money(row.prediction))
        if not chosen_total.empty:
            cols[1].metric('Historical total MAE', money(chosen_total.mae_gbp.iloc[0]))
            cols[2].metric('Observed total-range coverage', f'{chosen_total.interval_coverage.iloc[0]:.1%}')
        if show_ranges:
            st.caption(f'Total range: {money(row.lower)} to {money(row.upper)}. '
                       'Calibrated from cumulative forecast errors, not summed monthly bounds.')
    metrics = selected(data['metrics'])
    chosen = metrics[metrics.model == model]
    if show_ranges:
        if not chosen.empty:
            st.warning(f"Historical monthly range coverage varies from {chosen.interval_coverage.min():.1%} "
                       f"to {chosen.interval_coverage.max():.1%} against an 80% target. These are indicative diagnostics.")
        st.caption('Ranges use 12 calibration origins per lead. Historical forecast windows overlap; '
                   'calibration estimates are imprecise and are not reliable probability guarantees.')
    st.subheader('Historical model and benchmark comparison')
    curve = metrics.copy()
    curve['Method'] = curve.model.map(labels)
    chart(px.line(curve, x='lead_month', y='mae_gbp', color='Method', markers=True,
                  labels={'lead_month': 'Months ahead', 'mae_gbp': 'Mean absolute monthly error · GBP'}))
    if not chosen_total.empty and (chosen_total.skill_vs_seasonal < 0).any():
        st.warning('The selected model has larger historical cumulative errors than the seasonal benchmark. '
                   'Family selection does not guarantee a benchmark improvement.')
    table_export(metrics, 'monthly_model_metrics')
    st.subheader(f'Historical {horizon}-month total errors')
    table_export(total_metrics, 'monthly_total_metrics')
    with st.expander('Monthly predictions, selected configurations and provenance'):
        export = outlook.drop(columns='date')
        if not show_ranges:
            export = export.drop(columns=['lower', 'upper'], errors='ignore')
        table_export(export, 'monthly_outlook')
        parameter_columns = ['alpha'] if model == 'ridge' else ['depth', 'leaf', 'trees', 'seed']
        setting_columns = ['sector', 'target', 'horizon', 'history', 'window_months', 'half_life'] + parameter_columns
        st.dataframe(settings[[column for column in setting_columns if column in settings]], hide_index=True)
        st.json(meta)


if not DB_PATH.exists():
    st.title('Fund Flow')
    st.info('The analytics database has not been built yet. Import the supplied files before opening the dashboard.')
    st.code('venv/bin/python -m fund_analytics ingest', language='bash')
    st.stop()

try:
    with db.connect() as connection:
        manifest = connection.execute('SELECT * FROM ingestion_runs ORDER BY imported_at').fetchdf()
        coverage = connection.execute('SELECT min(date), max(date) FROM fund_daily').fetchone()
except Exception as exc:
    st.error(f'The database is not ready: {exc}')
    st.stop()

# Source hashes are also exposed below; the file identity protects against database rebuilds.
identity = f'{DB_PATH.stat().st_mtime_ns}:{DB_PATH.stat().st_size}'
if not manifest.empty:
    for field in ('source_fingerprint', 'fingerprint'):
        if field in manifest.columns:
            identity += ':' + str(manifest.iloc[-1][field])
            break


def q(sql, params=()):
    return query(identity, QUERY_VERSION, sql, tuple(params))


minimum, maximum = coverage
st.sidebar.markdown('### ◈ Fund Flow')
st.sidebar.caption('RESEARCH WORKBENCH · V1')
page = st.sidebar.radio('Workspace', ['Overview', 'Timing patterns', 'Client behaviour', 'Forecasts', 'Data quality'])
st.sidebar.divider()
sectors = q('SELECT DISTINCT client_sector FROM clients ORDER BY 1').iloc[:, 0].tolist()

if page not in ('Forecasts', 'Data quality'):
    default_start = max(minimum, maximum - timedelta(days=365 * 3))
    selected_dates = st.sidebar.date_input('Analysis period', (default_start, maximum),
                                          min_value=minimum, max_value=maximum)
    if len(selected_dates) != 2:
        st.info('Choose a start and end date to explore the data.')
        st.stop()
    start, end = selected_dates
    selected_sectors = st.sidebar.multiselect('Sectors · all when empty', sectors)
    type_sql = 'SELECT DISTINCT client_type FROM clients'
    type_params = []
    if selected_sectors:
        type_sql += ' WHERE client_sector IN (' + ','.join('?' for _ in selected_sectors) + ')'
        type_params = selected_sectors
    types = q(type_sql + ' ORDER BY 1', type_params).iloc[:, 0].tolist()
    selected_types = st.sidebar.multiselect('Client types · all when empty', types)
    client_predicates, client_params = [], []
    for column, values in [('client_sector', selected_sectors), ('client_type', selected_types)]:
        if values:
            client_predicates.append(column + ' IN (' + ','.join('?' for _ in values) + ')')
            client_params.extend(values)
    client_where = ' AND '.join(client_predicates) or 'TRUE'
    clients = q(f'SELECT client_id, client_name, client_sector FROM clients WHERE {client_where} ORDER BY client_name', client_params)
    names = dict(zip(clients.client_id, clients.client_name + ' · ' + clients.client_id))
    selected_clients = st.sidebar.multiselect('Clients · all when empty', clients.client_id.tolist(),
                                              format_func=lambda key: names.get(key, key))
    if selected_clients:
        client_predicates.append('client_id IN (' + ','.join('?' for _ in selected_clients) + ')')
        client_params.extend(selected_clients)
    client_where = ' AND '.join(client_predicates) or 'TRUE'
    basis = st.sidebar.radio('Date basis', ['Cash movement', 'Trade decision'])
    date_column = 'cash_flow_date' if basis == 'Cash movement' else 'trade_date'
    filter_caption = f'{start:%d %b %Y} – {end:%d %b %Y} · {basis} · ' + (', '.join(selected_sectors) if selected_sectors else 'All sectors')
    if selected_types:
        filter_caption += f' · {len(selected_types)} client types'
    if selected_clients:
        filter_caption += f' · {len(selected_clients)} clients'
else:
    filter_caption = f'Observed fund coverage {minimum:%d %b %Y} – {maximum:%d %b %Y}'

st.sidebar.divider()
st.sidebar.caption('Synthetic data · GBP\n\nLocal analytical workspace. Income distributions are separate from client withdrawals.')
st.markdown('<div class="eyebrow">FUND INTELLIGENCE / LOCAL ANALYTICS</div>', unsafe_allow_html=True)
st.title(page)
st.caption(filter_caption)


def daily_data():
    # Aggregate events in SQL before joining a small calendar spine.
    return q(f'''WITH selected AS (SELECT * FROM clients WHERE {client_where}),
        events AS (
          SELECT f.{date_column} AS date,
            sum(greatest(flow_gbp, 0)) AS subscriptions,
            sum(greatest(-flow_gbp, 0)) AS redemptions,
            count(*) AS transaction_count
          FROM client_flows f JOIN selected c USING(client_id)
          WHERE f.{date_column} BETWEEN ? AND ? GROUP BY 1),
        days AS (SELECT CAST(day AS DATE) AS date FROM generate_series(CAST(? AS DATE), CAST(? AS DATE), INTERVAL 1 DAY) t(day))
        SELECT d.date, coalesce(e.subscriptions, 0) AS subscriptions,
          coalesce(e.redemptions, 0) AS redemptions,
          coalesce(e.subscriptions, 0)-coalesce(e.redemptions, 0) AS net_flow,
          coalesce(e.transaction_count, 0) AS transaction_count
        FROM days d LEFT JOIN events e USING(date) ORDER BY date''', client_params + [start, end, start, end])


def sector_summary(group='client_sector'):
    if group not in ('client_sector', 'client_type'):
        raise ValueError('Unsupported group')
    return q(f'''SELECT c.{group}, sum(greatest(flow_gbp,0)) AS subscriptions,
        sum(greatest(-flow_gbp,0)) AS redemptions, sum(flow_gbp) AS net_flow,
        count(*) AS transaction_count, count(DISTINCT f.client_id) AS transacting_clients,
        avg(abs(flow_gbp)) AS average_transaction_gbp
        FROM client_flows f JOIN (SELECT * FROM clients WHERE {client_where}) c USING(client_id)
        WHERE f.{date_column} BETWEEN ? AND ? GROUP BY 1 ORDER BY redemptions DESC''', client_params + [start, end])


if page == 'Overview':
    daily = daily_data()
    totals = daily[['subscriptions', 'redemptions', 'net_flow', 'transaction_count']].sum()
    cols = st.columns(4)
    for col, key in zip(cols[:3], COLORS):
        col.metric(LABELS[key], money(totals[key]))
    cols[3].metric('Transactions', f'{totals.transaction_count:,.0f}')
    if totals.transaction_count == 0:
        st.info('No transactions match this selection. The calendar below still includes zero-flow days.')
    st.subheader('Money in, money out')
    granularity = st.segmented_control('Display frequency', ['Daily', 'Weekly', 'Monthly'], default='Monthly')
    rule = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'MS'}[granularity or 'Monthly']
    display = daily.set_index('date').resample(rule).sum().reset_index()
    lines(display, columns=tuple(COLORS))
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader('Sector contribution')
        sectors_data = sector_summary()
        if sectors_data.empty:
            st.info('No sector activity in this selection.')
        else:
            chart(px.bar(sectors_data, y='client_sector', x=['subscriptions', 'redemptions'],
                         orientation='h', barmode='group', color_discrete_map=COLORS,
                         labels={'value': 'GBP', 'client_sector': ''}))
    with right:
        st.subheader('Fund value')
        st.caption('Whole-fund AUM. Sector and client filters do not apply.')
        aum = q('SELECT date, closing_aum_gbp FROM fund_daily WHERE date BETWEEN ? AND ? ORDER BY date', [start, end])
        if not aum.empty:
            st.metric('Closing AUM · latest selected date', money(aum.iloc[-1].closing_aum_gbp))
            chart(px.line(aum, x='date', y='closing_aum_gbp', labels={'closing_aum_gbp': 'GBP'}, color_discrete_sequence=['#5773b7']))
    with st.expander('Show flow totals and CSV download'):
        table_export(display, 'overview_flows')

elif page == 'Timing patterns':
    st.write('Compare calendar patterns using all observed days, including days with no flows.')
    daily = daily_data()
    # Account exposure is counted on each date, respecting historical opening and closure.
    exposure = q(f'''WITH selected AS (SELECT account_open_date, account_close_date FROM clients WHERE {client_where}),
        days AS (SELECT CAST(day AS DATE) AS date FROM generate_series(CAST(? AS DATE), CAST(? AS DATE), INTERVAL 1 DAY) t(day))
        SELECT d.date, count(c.account_open_date) AS active_clients FROM days d LEFT JOIN selected c
        ON c.account_open_date <= d.date AND (c.account_close_date IS NULL OR c.account_close_date >= d.date)
        GROUP BY 1 ORDER BY 1''', client_params + [start, end])
    daily = daily.merge(exposure, on='date', how='left')
    pattern = st.radio('Calendar view', ['Weekday', 'Position in month', 'Month of year'], horizontal=True)
    if pattern == 'Weekday':
        daily['period'] = daily.date.dt.day_name()
        ordering = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    elif pattern == 'Position in month':
        daily['period'] = pd.cut(daily.date.dt.day, [0, 7, 14, 21, 31], labels=['1–7', '8–14', '15–21', '22–end'])
        ordering = ['1–7', '8–14', '15–21', '22–end']
    else:
        daily['period'] = daily.date.dt.month_name()
        ordering = pd.date_range('2024-01-01', periods=12, freq='MS').month_name().tolist()
    summary = daily.groupby('period', observed=True).agg(
        subscriptions=('subscriptions', 'sum'), redemptions=('redemptions', 'sum'),
        observations=('date', 'size'), transaction_count=('transaction_count', 'sum'),
        client_days=('active_clients', 'sum')).reindex(ordering).dropna(subset=['observations']).reset_index()
    summary['subscriptions_per_day'] = summary.subscriptions / summary.observations
    summary['redemptions_per_day'] = summary.redemptions / summary.observations
    summary['transactions_per_1000_client_days'] = summary.transaction_count / summary.client_days.replace(0, np.nan) * 1000
    summary['average_transaction_gbp'] = (summary.subscriptions + summary.redemptions) / summary.transaction_count.replace(0, np.nan)
    chart(px.bar(summary, x='period', y=['subscriptions_per_day', 'redemptions_per_day'], barmode='group',
                 color_discrete_sequence=list(COLORS.values()), labels={'period': '', 'value': 'Average GBP per observed day'}))
    a, b = st.columns(2)
    with a:
        chart(px.bar(summary, x='period', y='transactions_per_1000_client_days', labels={'period': '', 'transactions_per_1000_client_days': 'Transactions per 1,000 active client-days'}, color_discrete_sequence=['#5773b7']))
    with b:
        chart(px.bar(summary, x='period', y='average_transaction_gbp', labels={'period': '', 'average_transaction_gbp': 'Average transaction · GBP'}, color_discrete_sequence=['#16877c']))
    st.caption('Client-days count accounts open on each observed date. Rates are undefined when exposure or transaction count is zero. Patterns describe this synthetic sample.')
    table_export(summary, 'timing_patterns')

elif page == 'Client behaviour':
    group = st.radio('Compare groups', ['Sector', 'Client type'], horizontal=True)
    group_col = 'client_sector' if group == 'Sector' else 'client_type'
    summary = sector_summary(group_col)
    if summary.empty:
        st.info('No transactions match the selected period and filters.')
    else:
        chart(px.bar(summary, y=group_col, x='net_flow', orientation='h', labels={group_col: '', 'net_flow': 'Net flow · GBP'}, color_discrete_sequence=['#16877c']))
        with st.expander('Group frequency and transaction size'):
            table_export(summary, 'client_groups')
    concentration = q(f'''SELECT f.client_id,c.client_name, count(*) AS transactions,
        sum(abs(flow_gbp)) AS gross_flow_gbp, avg(abs(flow_gbp)) AS average_transaction_gbp
        FROM client_flows f JOIN (SELECT * FROM clients WHERE {client_where}) c USING(client_id)
        WHERE f.{date_column} BETWEEN ? AND ? GROUP BY 1,2 ORDER BY gross_flow_gbp DESC''', client_params + [start, end])
    if not concentration.empty:
        gross = concentration.gross_flow_gbp.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric('Transacting clients', f'{len(concentration):,}')
        c2.metric('Top 10 share of gross flows', f'{concentration.head(10).gross_flow_gbp.sum() / gross:.1%}' if gross else '—')
        c3.metric('Largest client share', f'{concentration.iloc[0].gross_flow_gbp / gross:.1%}' if gross else '—')
        with st.expander('Client concentration and frequency'):
            table_export(concentration, 'client_concentration')
    st.subheader('Compare client histories')
    st.caption('Select 2–50 clients to compare. Empty periods are zero while accounts are open, and unavailable outside their lifetime.')
    candidates = selected_clients or clients.client_id.tolist()
    picks = st.multiselect('Clients for comparison', candidates, format_func=lambda key: names.get(key, key), max_selections=50)
    frequency = st.radio('Comparison frequency', ['Monthly', 'Weekly'], horizontal=True)
    target = st.selectbox('Flow measure', list(LABELS), format_func=LABELS.get)
    if picks:
        unit = 'month' if frequency == 'Monthly' else 'week'
        threshold = 12 if unit == 'month' else 26
        amount = {'subscriptions': 'greatest(flow_gbp,0)', 'redemptions': 'greatest(-flow_gbp,0)', 'net_flow': 'flow_gbp'}[target]
        marks = ','.join('?' for _ in picks)
        histories = q(f'''WITH selected AS (SELECT * FROM clients WHERE client_id IN ({marks})),
            periods AS (SELECT CAST(p AS DATE) AS period FROM generate_series(date_trunc('{unit}',CAST(? AS DATE)),date_trunc('{unit}',CAST(? AS DATE)),INTERVAL 1 {unit}) t(p)),
            flows AS (SELECT client_id, date_trunc('{unit}',{date_column}) AS period, sum({amount}) AS amount
              FROM client_flows WHERE client_id IN ({marks}) AND {date_column} BETWEEN ? AND ? GROUP BY 1,2)
            SELECT p.period,c.client_id,coalesce(f.amount,0) AS amount FROM periods p CROSS JOIN selected c
            LEFT JOIN flows f ON f.client_id=c.client_id AND f.period=p.period
            WHERE c.account_open_date <= least(CAST(? AS DATE),p.period+INTERVAL 1 {unit}-INTERVAL 1 DAY)
              AND coalesce(c.account_close_date,CAST(? AS DATE)) >= greatest(CAST(? AS DATE),p.period)
            ORDER BY 1,2''', picks + [start, end] + picks + [start, end, end, end, start])
        if histories.empty:
            st.info('The selected clients have no open-account periods in this date range.')
        else:
            chart(px.line(histories, x='period', y='amount', color='client_id', labels={'period': '', 'amount': f'{LABELS[target]} · GBP'}))
            matrix = histories.pivot(index='period', columns='client_id', values='amount')
            correlation = matrix.corr(min_periods=threshold)
            present = matrix.notna().astype(int)
            overlap = present.T @ present
            if len(picks) < 2:
                st.info('Choose another client to display pairwise correlations.')
            elif correlation.notna().sum().sum() == 0:
                st.info(f'No valid correlations: at least {threshold} shared periods and variation in both histories are required.')
            else:
                figure = go.Figure(go.Heatmap(z=correlation.values, x=correlation.columns, y=correlation.index,
                    customdata=overlap.values, zmin=-1, zmax=1, zmid=0, colorscale='RdBu',
                    hovertemplate='%{y} / %{x}<br>Correlation: %{z:.3f}<br>Shared periods: %{customdata}<extra></extra>'))
                chart(figure)
            st.caption(f'Pearson correlation, minimum {threshold} shared {frequency.lower()} periods. Blank cells indicate insufficient overlap or a constant series. Correlation does not establish causation.')
            with st.expander('Shared observation counts'):
                st.dataframe(overlap)
            table_export(histories, 'client_histories')

elif page == 'Forecasts':
    monthly_view(str(manifest.iloc[-1]['fingerprint']), sectors, maximum)
    with st.expander('Scheduled income distributions · whole fund'):
        st.caption('Declared payments are separate obligations and are excluded from predicted client redemptions. '
                   'Historical schedule availability is unknown; these schedules are not used in backtests.')
        upcoming = q('''SELECT payment_date, payment_status, sum(distribution_gbp) AS scheduled_distribution_gbp
            FROM income_distributions WHERE payment_status = 'Declared' AND payment_date > ?
            GROUP BY 1,2 ORDER BY 1''', [maximum])
        if upcoming.empty:
            st.info('No declared future income payments for the fund.')
        else:
            table_export(upcoming, 'scheduled_distributions')

else:
    st.write('Trace the dashboard back to its source files and inspect import validation.')
    counts = q("SELECT 'Clients' AS dataset,count(*) AS rows FROM clients UNION ALL SELECT 'Client flows',count(*) FROM client_flows UNION ALL SELECT 'Income distributions',count(*) FROM income_distributions UNION ALL SELECT 'Daily fund',count(*) FROM fund_daily")
    for col, row in zip(st.columns(4), counts.itertuples()):
        col.metric(row.dataset, f'{row.rows:,}')
    st.caption(f'Database size: {DB_PATH.stat().st_size / 1024**2:,.1f} MiB · Fund observation window: {minimum} to {maximum}')
    st.subheader('Import provenance and validation')
    for row in manifest.to_dict('records'):
        for key, value in row.items():
            if isinstance(value, str) and value[:1] in ('{', '['):
                try:
                    row[key] = json.loads(value)
                except ValueError:
                    pass
        st.json(row, expanded=False)
    st.subheader('Query performance')
    report = ROOT / 'data/clean/benchmark.json'
    if report.exists():
        st.json(json.loads(report.read_text()))
    else:
        st.caption('No saved benchmark report yet. This check measures a fresh database connection and an uncached sector aggregate query.')
    if st.button('Measure an uncached query'):
        tick = perf_counter()
        with db.connect() as connection:
            connection.execute('SELECT client_sector,sum(subscriptions),sum(redemptions) FROM daily_sector GROUP BY 1').fetchall()
        elapsed = perf_counter() - tick
        st.metric('Uncached query including connection', f'{elapsed * 1000:,.1f} ms')
        st.caption('A single observation, not a full median/p95 benchmark. Target: common aggregate queries below one second.')
    st.info('Refresh data with the dashboard stopped. This workspace uses only the supplied files; raw files are preserved.')

st.divider()
st.caption(f'SYNTHETIC DATA · GBP · Source identity {identity[-16:]} · Net flow = subscriptions − redemptions')
