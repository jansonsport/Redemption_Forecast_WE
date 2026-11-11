"""Segments & stress page: descriptive client-relationship study (not a forecast).

Rendered by app.py, which passes its shared helpers (chart, money, table_export) so this page keeps the
dashboard's look. Behaviour follows handoff/segments_stress/reference_ui/segments.js; numbers come from
the offline bundle (`python -m fund_analytics segments`) and the live stress engine.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fund_analytics.segments import bundle as B
from fund_analytics.segments import views as V
from fund_analytics.segments.engine import Period, StressEngine, default_definition

ROOT = Path(__file__).resolve().parent
NOTEBOOK_URL = 'app/static/segments/11_segments_correlation_stress.html'
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
FAMILIES = ['Concentration', 'Historical replay', 'Stress period', 'Segment tails', 'Combined']
FLOW_COLOURS = {'sub': '#16877c', 'red': '#d26b45', 'net': '#5773b7'}
STRIP_COLOURS = {'used': '#d26b45', 'periods': '#a050c8', 'price': '#5773b7'}


@st.cache_resource(show_spinner='Loading the segment study…', max_entries=1)
def _load(identity):
    bundle = B.load()
    return bundle, StressEngine(bundle)


def _short(text: str, n: int = 30) -> str:
    """Axis labels stay readable at phone width; the full text is in the hover."""
    return text if len(text) <= n else text[:n - 1].rstrip() + '…'


def _scenario_label(text: str) -> str:
    """Compact scenario name for the chart axis (full name in the hover); stays unique per scenario."""
    for a, b in ((': worst ', ' · '), (' net outflow', ''), (' clients redeem in full', ' exit in full'),
                 ('99th pct, all segments together (ρ=1)', 'segment tails, ρ=1'), ('99th pct, historical stress correlation', 'segment tails, stress ρ'),
                 (' + top 10 full exit + known payables', ' + top 10 + payables'), (' (ended ', ' ('), ('5-day segment tails (ρ=1)', '5-day tails')):
        text = text.replace(a, b)
    return _short(text, 34)


def _swatch(colour: str) -> str:
    return f'<span style="color:{colour};font-size:1.1em" aria-hidden="true">●</span>'


def render(chart, money, table_export, current_fingerprint=None):
    if not B.BUNDLE.exists():
        st.info('The segment study has not been built yet. Build it after importing the database (about 10 seconds).')
        st.code('venv/bin/python -m fund_analytics segments', language='bash')
        return
    stat = B.BUNDLE.stat()
    bundle, engine = _load((stat.st_mtime_ns, stat.st_size))
    meta, o = bundle['meta'], bundle['out']
    if current_fingerprint and meta['source_fingerprint'] != current_fingerprint:
        st.warning('The segment study was built from a different data import. Rebuild it before using this page.')
        st.code('venv/bin/python -m fund_analytics segments', language='bash')
        return
    P = Page(chart, money, table_export, bundle, engine)
    P.show()


class Page:
    def __init__(self, chart, money, table_export, bundle, engine):
        self.chart, self.money, self.export = chart, money, table_export
        self.meta, self.o, self.clients, self.engine = bundle['meta'], bundle['out'], bundle['clients'], engine
        self.order = self.meta['order']
        self.col = V.colours(self.order)
        o = self.o
        self.prof = o['profiles'].set_index('segment')
        self.base = o['base']; self.aum = float(self.base['aum'])
        self.conc = o['concentration'].iloc[-1]
        self.top = o['top'].assign(client_name=lambda t: t.client_id.map(self.clients.client_name),
                                   client_sector=lambda t: t.client_id.map(self.clients.client_sector))
        self.ss = st.session_state
        self.ss.setdefault('sg_section', V.SECTIONS[0])
        self._init_stress_state()
        self.live = engine.run(self.ss.sg_dd, self.periods())

    # ------------------------------------------------------------------ state
    def _default_periods_frame(self):
        d = default_definition(self.meta['episodes'])
        return pd.DataFrame([dict(use=p.on, name=p.name, start=date.fromisoformat(p.start), end=date.fromisoformat(p.end)) for p in d['periods']])

    def _init_stress_state(self):
        self.ss.setdefault('sg_dd', 0.10)
        self.ss.setdefault('sg_editor_version', 0)
        if 'sg_periods' not in self.ss:
            self.ss.sg_periods = self._default_periods_frame()

    def periods(self, frame=None):
        frame = self.ss.sg_periods if frame is None else frame
        out = []
        for r in frame.itertuples():
            if pd.isna(r.start) or pd.isna(r.end):
                continue
            out.append(Period(str(r.name) if isinstance(r.name, str) and r.name.strip() else 'Custom period',
                              pd.Timestamp(r.start).date().isoformat(), pd.Timestamp(r.end).date().isoformat(), bool(r.use)))
        return out

    def is_default(self):
        d = self._default_periods_frame()
        cur = self.ss.sg_periods.reset_index(drop=True)
        return self.ss.sg_dd == 0.10 and len(cur) == len(d) and cur.astype(str).equals(d.astype(str))

    def goto(self, section):
        self.ss.sg_section = section

    @staticmethod
    def card(col, label, value, sub):
        col.metric(label, value)
        col.caption(sub)

    # ------------------------------------------------------------------ frame
    def show(self):
        st.markdown('**Descriptive analysis of relationships between clients, not a forecast.** '
                    'It shows which clients have behaved alike, whether groups of clients have withdrawn at the same time, '
                    'how concentrated holdings are, and how far a cash buffer would stretch if the largest clients left.')
        c1, c2, _ = st.columns([1, 1, 3])
        c1.link_button('Open full study notebook ↗', NOTEBOOK_URL, width='stretch')
        seg_csv = self.o['features'][['segment', 'eligible']].reset_index()
        c2.download_button('Client → segment file · CSV', seg_csv.to_csv(index=False).encode(), 'client_segments.csv', 'text/csv',
                           key='sg_dl_segments', width='stretch')
        st.caption(f"Position at {self.base['date']} · AUM {self.money(self.aum)} · study built {self.meta['built_at'][:10]} · "
                   'the notebook is the study as originally run; this page uses this repository\'s data and settings (see README).')
        section = st.segmented_control('Section', V.SECTIONS, key='sg_section', label_visibility='collapsed')
        if section is None:
            section = V.SECTIONS[0]
        st.divider()
        {V.SECTIONS[0]: self.overview, V.SECTIONS[1]: self.data, V.SECTIONS[2]: self.finding, V.SECTIONS[3]: self.segments,
         V.SECTIONS[4]: self.stress, V.SECTIONS[5]: self.tests, V.SECTIONS[6]: self.query}[section]()
        st.divider()
        st.caption('Synthetic data. Segments describe past behaviour, not future intentions, and are assigned using each client\'s full '
                   'history (hindsight), so they are never forecast inputs. Correlation shows co-movement, not causation. Stress scenarios '
                   'are illustrative exposures, not probabilities. The database has no portfolio holdings, so buffers are illustrative.')

    def banner(self):
        sp = self.periods(); on = [p for p in sp if p.on]; dd = self.ss.sg_dd
        with st.container(border=True):
            kind = 'study default' if self.is_default() else 'custom: results recomputed live'
            trig = f'price ≥{V.pct(dd, 0)} below its one-year high' if dd else 'no price trigger'
            pers = f" or inside {len(on)} period{'s' if len(on) != 1 else ''} ({', '.join(p.name for p in on)})" if on else ''
            st.markdown(f'**Stress definition ({kind}):** {trig}{pers} · **{self.live["days"]:,}** stress days '
                        f'({V.pct(self.live["days"] / len(self.engine.dates))} of history)')
            if self.ss.sg_section != V.SECTIONS[4]:
                st.button('Edit stress periods', key='sg_banner_edit', on_click=self.goto, args=(V.SECTIONS[4],))

    def bars(self, labels, series, y_title, x_title='', barmode='group', height=None, tickformat=None, orientation='v', hover=None):
        fig = go.Figure()
        for name, values, colour in series:
            kw = dict(x=labels, y=values) if orientation == 'v' else dict(y=labels, x=values, orientation='h')
            fig.add_bar(name=name, marker_color=colour, hovertemplate=hover, **kw)
        fig.update_layout(barmode=barmode, height=height, showlegend=len(series) > 1, bargap=0.25)
        if orientation == 'v':
            fig.update_yaxes(title_text=y_title, tickformat=tickformat); fig.update_xaxes(title_text=x_title)
        else:
            fig.update_xaxes(title_text=y_title, tickformat=tickformat); fig.update_yaxes(title_text=x_title, autorange='reversed')
        return fig

    # ================================================================ Overview
    def overview(self):
        o, prof = self.o, self.prof
        rare = prof.loc[V.RARE]; eligible = o['typical']['clients']
        top10 = o['tests'].set_index('scenario').loc['Top 10 clients redeem in full']
        rev10 = int(o['reverse'].set_index('buffer').loc['10% of AUM', 'clients_before_exhausted'])
        trades = int(o['inventory'].set_index('role').loc['Transactions', 'rows'])
        steps = [(V.SECTIONS[1], 'Collect & check', f'4 tables · {trades:,} trades', 'All flows reconcile to the fund to the penny'),
                 (V.SECTIONS[2], 'Measure behaviour', '9 habits per client', 'Size, frequency, lumpiness, timing, stress response'),
                 (V.SECTIONS[2], 'Set aside rare traders', f'{int(rare.clients):,} clients', f'Too little history · hold {V.pct(rare.aum_share, 0)} of AUM'),
                 (V.SECTIONS[2], 'Group with k-means', f"{self.meta['k']} segments", f'{eligible:,} measurable clients, tried 3–8 groups'),
                 (V.SECTIONS[4], 'Test in stress', f"{self.live['days']:,} stress days", 'How each segment has redeemed when markets fell'),
                 (V.SECTIONS[5], 'Stress the fund', f'Top-10 exit {V.pct(top10.outflow_share, 0)} of AUM', 'How many big clients cash could cover'),
                 (V.SECTIONS[6], 'Explore clients', 'Query any segment', 'Filter, sort and inspect individual clients')]
        st.subheader('How the study works')
        st.caption('Each step feeds the next. Open a step to see it.')
        for row in (steps[:4], steps[4:]):
            cols = st.columns(4)
            for col, (i, (target, title, big, sub)) in zip(cols, [(steps.index(s), s) for s in row]):
                with col.container(border=True):
                    st.markdown(f'**{i + 1} · {title}**\n\n### {big}')
                    st.caption(sub)
                    st.button('Open →', key=f'sg_step_{i}', on_click=self.goto, args=(target,), help=f'Go to {target}')
        tight = [s for s in self.order if V.verdict(s, o['cohesion'], o['random_pair_r'])[0] == 'Tight group']
        c = st.columns(4)
        self.card(c[0], 'Behavioural segments', f"{self.meta['k']}", f'plus rare traders · {len(tight)} tight groups; the rest loose or labels only')
        self.card(c[1], 'Rare traders hold', V.pct(rare.aum_share), f"{int(rare.clients):,} clients · {int(self.conc.top10_rare_traders)} of the 10 biggest")
        self.card(c[2], 'Top-10 holders', V.pct(self.conc.top10_share), 'of AUM: the main liquidity exposure')
        self.card(c[3], '10% buffer covers', f'{rev10} clients', 'of the largest, leaving in full')
        vl = o['value_lift'].sort_values('month_matched_value_lift', ascending=False).head(3)
        worst5 = o['tests'][(o['tests'].family == 'Historical replay') & o['tests'].scenario.str.contains('5-day')].outflow_gbp.max()
        st.subheader('What we found')
        st.markdown(
            f"1. **Behaviour forms loose groups, not sharp types.** {self.meta['k']} segments; {len(tight)} "
            f"({', '.join(tight) or 'none'}) are tight groups whose members have moved money together. The others are looser. *See 2 · Finding segments.*\n"
            '2. **Segments describe *how* clients use the fund; sector describes *when*.** Segments explain more of later behaviour overall, '
            'but sector does better on year-end (March/December) redemptions. Use both.\n'
            '3. **In past stress periods the typical client behaved as usual, but large redemptions rose in some segments:** '
            + ', '.join(f'{r.segment} {r.month_matched_value_lift:.2f}×' for r in vl.itertuples()) + ' the calm-day rate for the same month. *See 4 · Segments in stress.*\n'
            '4. **Segments have not piled on together in stress.** Their flows moved together about as much in stress as in calm periods; '
            'most of the togetherness is shared calendar habits.\n'
            f'5. **Concentration is the main liquidity exposure.** The 10 largest clients hold {V.pct(self.conc.top10_share)} ({self.money(top10.outflow_gbp)}), '
            f'about {top10.outflow_gbp / worst5:.0f}× the worst 5-day outflow on record, and {int(self.conc.top10_rare_traders)} of them are rare traders whose '
            'history tells us little. *See 5 · Concentration & stress tests.*')
        st.caption('Segments describe client behaviour; they did not improve forecasts (Notebook 09), so treat the largest clients individually.')

    # ================================================================ 1 Data
    def data(self):
        o = self.o; ok = bool(o['checks'].passed.all())
        st.subheader('What we took from the database')
        st.caption('Four tables, mapped through one configuration so another database can be used by changing column names only.')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Taken from the database**')
            for r in o['inventory'].itertuples():
                st.markdown(f'**{r.role}** ({r.table}, {r.rows:,} rows)  \n'
                            f"<small>{r.fields_present}{' · ' + r.notes if isinstance(r.notes, str) and r.notes else ''}</small>", unsafe_allow_html=True)
        with c2:
            err = o['checks'].set_index('check').loc['Holdings = units × price', 'result']
            st.markdown('**Missing, and how we derived it**')
            st.markdown(
                "- **Units held**: not stored → balance after each trade ÷ unit price of the client's share class. Checked: unit changes equal flow ÷ price.\n"
                f'- **Daily holdings by client and segment**: not stored → units × daily price, labelled *reconstructed* wherever shown ({err.split("; ")[-1]}).\n'
                '- **Stress days**: not stored → fund price ≥10% below its one-year high, plus named market episodes. You can change these in section 4.\n'
                '- **Switches/transfers between accounts**: no flow-type field → cannot be identified or netted out.\n'
                '- **Fund yield**: not available → yield sensitivity (an MMF measure) not computed.\n'
                '- **Portfolio holdings and maturities**: not available → liquid assets cannot be measured; buffers in section 5 are illustrative.')
        with c3:
            st.markdown(f"**Checks {'✓ all passed' if ok else '✗ failed'}**")
            for r in o['checks'].itertuples():
                st.markdown(f"{'✓' if r.passed else '✗'} {r.check}  \n<small>{r.result}</small>", unsafe_allow_html=True)
        with st.expander('Full inventory and check details'):
            self.export(o['inventory'], 'segments_inventory')
            self.export(o['checks'], 'segments_checks')

    # ================================================================ 2 Finding segments
    def finding(self):
        o, meta = self.o, self.meta
        rare = self.prof.loc[V.RARE]; rule = meta['eligibility']; sel = o['selection']; ch = sel[sel.chosen].iloc[0]
        st.subheader("Step 1 · Measure each client's habits")
        st.markdown("For every client we turned 20 years of trades into **nine numbers** describing how it uses the fund:")
        habits = [('Holding size', 'average balance'), ('Trading frequency', 'trades per year'), ('Redemption share', 'how much of its traded money went out'),
                  ('Trade size', 'typical trade vs its balance'), ('Lumpiness', 'largest redemption vs its balance'), ('Stress response', 'redemption rate in stress vs calm'),
                  ('March habit', 'share of redemptions in March'), ('December habit', 'share of redemptions in December'), ('Month-end habit', 'share of trades in the last 3 business days')]
        for row in range(3):
            cols = st.columns(3)
            for col, (a, b) in zip(cols, habits[row * 3:row * 3 + 3]):
                with col.container(border=True):
                    st.markdown(f'**{a}**'); st.caption(b)

        st.subheader("Step 2 · Set aside clients we can't measure")
        st.caption(f"Clients with fewer than {rule['min_trades']} trades or under {rule['min_years']:g} year of history give too few data points. "
                   f"They are kept as their own group, **rare traders** ({int(rare.clients):,} clients, {V.pct(rare.aum_share)} of AUM), rather than dropped.")
        years = st.segmented_control('Minimum years active', [0.5, 1.0, 2.0], default=1.0, key='sg_el_years', format_func=lambda v: f'{v:g}')
        years = 1.0 if years is None else years
        el = o['eligibility'][o['eligibility'].min_years == years].copy()
        used = (el.min_trades == rule['min_trades']) & (el.min_years == rule['min_years'])
        st.dataframe(pd.DataFrame({'Minimum trades': [f"{t}{' (used)' if u else ''}" for t, u in zip(el.min_trades, used)],
                                   'Measurable clients': el.eligible_clients.map('{:,}'.format), 'Share of clients': el.eligible_share.map(V.pct),
                                   'AUM left with rare traders': el.rare_aum_share.map(V.pct)}), hide_index=True, width='stretch')

        st.subheader('Step 3 · Let k-means form the groups')
        st.markdown('k-means is a simple sorting method. Picture every client as a dot placed by its nine habits '
                    '(the map below squeezes the nine into two directions so we can see it):')
        cols = st.columns(4)
        for col, (n, t) in zip(cols, [(1, f"Drop {int(ch.k)} group centres onto the map."), (2, 'Give every client to its nearest centre.'),
                                      (3, 'Move each centre to the middle of its clients.'), (4, 'Repeat until nothing moves. Each centre is now a group\'s "typical client".')]):
            with col.container(border=True):
                st.markdown(f'**{n}** · {t}')
        named = [s for s in self.order if s != V.RARE]
        hi = st.selectbox('Highlight', ['All segments'] + named, key='sg_map_seg')
        st.caption(f"The {o['typical']['clients']:,} measurable clients on a behaviour map: 2,500 sampled. Numbered circles are the group centres. "
                   'Nearby dots behave alike; overlapping colours show that the groups blend into each other.')
        self.chart(self.scatter_map(None if hi == 'All segments' else hi), hovermode='closest')
        st.caption(self.map_axes + ' Each axis combines all nine habits; these are the two habits that weigh most on it.')

        st.subheader('Step 4 · Choose how many groups')
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = self.bars(sel.k.astype(str).tolist(), [('Separation (silhouette)', sel.silhouette, V.SEGMENT_PALETTE[0]),
                                                         ('Stability (bootstrap agreement)', sel.ari_median_10boot, V.SEGMENT_PALETTE[1])],
                            'Score (0–1)', 'Number of groups tried', height=320, hover='%{fullData.name}: %{y:.3f}<extra>k = %{x}</extra>')
            self.chart(fig)
        with c2:
            auto = int(sel[sel.auto_choice].k.iloc[0])
            st.markdown(f"We tried **3 to 8 groups** and kept **{int(ch.k)}**. Separation (how much closer clients are to their own centre than to the next one) "
                        f"is almost tied between 7 and 8 groups ({sel.set_index('k').silhouette[7]:.3f} vs {sel.set_index('k').silhouette[8]:.3f}); "
                        + (f'the automatic rule picks {auto} in this environment, ' if auto != int(ch.k) else '')
                        + 'the study keeps 7 so its segments stay comparable with the original analysis.')
            st.markdown(f"**Stability:** re-running k-means on {len(o['ari'])} resampled versions of the data gave the same grouping most of the time "
                        f"(agreement {np.median(o['ari']):.2f}, where 1 = identical).")
            st.caption(f'Separation is low ({ch.silhouette:.2f} on 0–1): behaviour is a continuum, so the groups are a useful simplification, not natural types.')

        st.subheader('Step 5 · How alike is each group?')
        coh = o['cohesion'].set_index('segment').reindex(self.order)
        st.caption(f"Two tests. **Same habits:** share of members that clearly belong to their group. **Move money together:** how much members' monthly "
                   f"flows have moved together, compared with two random clients ({o['random_pair_r']:.2f}).")
        fig = self.bars(self.order, [('Members move money together (avg r)', coh.flow_r_within, V.SEGMENT_PALETTE[0]),
                                     ('Clearly belong (share)', coh.well_fitted_share, V.SEGMENT_PALETTE[1])],
                        '0 = none, 1 = all', 'Segment', height=340, hover='%{fullData.name}: %{y:.2f}<extra>%{x}</extra>')
        self.chart(fig)
        rows = []
        for s in self.order:
            c = coh.loc[s]; v, why = V.verdict(s, o['cohesion'], o['random_pair_r'])
            rows.append({'Segment': s, 'Verdict': v, 'Clearly belong': V.pct(c.well_fitted_share, 0), 'Spread vs whole fund': V.pct(c.spread_vs_fund, 0),
                         'Move together (avg r)': V.num2(c.flow_r_within),
                         'vs random pair': f'{c.flow_r_within / o["random_pair_r"] + 0.0:.1f}×'.replace('-0.0×', '0.0×') if V.finite(c.flow_r_within) else '—', 'What it means': why})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        st.subheader('Step 6 · Is this better than sector, client type or region?')
        G = o['grouping']; names = list(dict.fromkeys(G.grouping)); seg_name = next(g for g in names if 'segments' in g.lower())
        st.markdown(f"A fair test: groups were formed from behaviour in **2006–2016**, then scored on how much of each client's **2016–2026** behaviour "
                    f"they explain (0% = none, 100% = all), across {o['grouping_clients']:,} clients active in both periods.")
        avg = G.groupby('grouping').eta_squared.mean()
        cols = st.columns(len(names))
        for col, g in zip(cols, names):
            self.card(col, g.replace(' (formed 2006–2016)', ''), V.pct(avg[g], 0), 'of later behaviour explained')
        feats = list(V.FEATURE_LABELS)
        val = G.pivot(index='feature', columns='grouping', values='eta_squared').reindex(feats)
        palette = [V.SEGMENT_PALETTE[0], V.SEGMENT_PALETTE[1], V.SEGMENT_PALETTE[4], V.RARE_COLOUR]
        fig = self.bars([V.FEATURE_LABELS[f] for f in feats], [(g, 100 * val[g], palette[i]) for i, g in enumerate(names)],
                        '% of 2016–26 behaviour explained', 'Habit', height=380, hover='%{fullData.name}: %{y:.1f}%<extra>%{x}</extra>')
        self.chart(fig)
        better = [V.lower_label(V.FEATURE_LABELS[f]) for f in feats if val.loc[f, seg_name] > val.loc[f, 'Sector'] + .05]
        worse = [V.lower_label(V.FEATURE_LABELS[f]) for f in feats if val.loc[f, 'Sector'] > val.loc[f, seg_name] + .05]
        st.markdown(f"**Segments do better on** {', '.join(better) or 'no habit'}, which is *how* clients use the fund. "
                    f"**Sector and client type do better on** {', '.join(worse) or 'no habit'}, which is *when* they redeem, because a charity's type sets its "
                    f"financial year-end. **Region** explains almost nothing ({V.pct(avg['Region'])}). No grouping explains stress response well.")
        with st.container(border=True):
            st.markdown('**Use both:** read the historical **timing** of outflows by sector, and their **size and lumpiness** by segment.')

    def scatter_map(self, highlight):
        pca = self.o['pca']; pts = pca['points']; cen = pca['centres'].set_index('segment')
        lo_x, hi_x = pts.x.quantile([.005, .995]); lo_y, hi_y = pts.y.quantile([.005, .995])
        fig = go.Figure()
        for i, s in enumerate(self.order):
            if s == V.RARE:
                continue
            p = pts[pts.segment == s]; on = highlight in (None, s)
            fig.add_scatter(x=p.x, y=p.y, mode='markers', name=f'{i + 1} · {s}', customdata=p.client_id,
                            marker=dict(size=5 if on else 4, color=self.col[s] if on else '#cfd6e0', opacity=.7 if on else .45),
                            hovertemplate=f'{s}<br>%{{customdata}}<extra></extra>')
        for i, s in enumerate(self.order):
            if s == V.RARE or s not in cen.index:
                continue
            on = highlight in (None, s)
            fig.add_scatter(x=[cen.loc[s, 'x']], y=[cen.loc[s, 'y']], mode='markers+text', text=[str(i + 1)], showlegend=False,
                            textfont=dict(color='#ffffff', size=11), marker=dict(size=22 if on else 14, color=self.col[s], line=dict(color='#ffffff', width=3)),
                            hovertemplate=f'{s}<br>Centre of the group: its typical client<extra></extra>')

        def axis(pc):
            l = pca['loadings'].assign(a=lambda d: d[pc].abs()).sort_values('a', ascending=False).head(2)
            return ', '.join(('more ' if r[pc] > 0 else 'less ') + V.lower_label(V.FEATURE_LABELS[r.feature]) for _, r in l.iterrows())
        ex = pca['explained']
        fig.update_xaxes(title_text=f'Behaviour axis 1 ({V.pct(ex[0], 0)} of variation)', range=[lo_x, hi_x])
        fig.update_yaxes(title_text=f'Behaviour axis 2 ({V.pct(ex[1], 0)})', range=[lo_y, hi_y])
        self.map_axes = f'Axis 1 → {axis("pc1")}. Axis 2 → {axis("pc2")}.'
        fig.update_layout(height=520, legend=dict(orientation='h', y=-0.2))
        return fig

    # ================================================================ 3 The segments
    def segments(self):
        o, prof = self.o, self.prof
        st.markdown('*Segments describe client behaviour; they did not improve forecasts (Notebook 09), so treat the largest clients individually.*')
        st.subheader('The segments in plain words')
        st.caption('Each card shows what makes the group different from a typical client, how tight the group is, and what its past behaviour means for liquidity.')
        for row in range(0, len(self.order), 4):
            cols = st.columns(4)
            for col, s in zip(cols, self.order[row:row + 4]):
                p = prof.loc[s]; d = V.describe(s, o['profiles'], o['typical'], self.meta['eligibility'], int(self.conc.top10_rare_traders))
                v, why = V.verdict(s, o['cohesion'], o['random_pair_r'])
                with col.container(border=True):
                    st.markdown(f'{_swatch(self.col[s])} **{s}**', unsafe_allow_html=True)
                    st.caption(f'{int(p.clients):,} clients · {V.pct(p.aum_share)} of AUM · **{v}**', help=why)
                    st.markdown('\n'.join(f'- {l}' for l in d['lines']))
                    st.markdown(f"**For liquidity:** {d['implication']}")
                    st.button(f'See these {int(p.clients):,} clients →', key=f'sg_see_{s}', on_click=self.see_clients, args=(s,))

        st.subheader('Compare segments')
        metric = st.selectbox('Measure', list(V.PROFILE_METRICS), format_func=lambda k: V.PROFILE_METRICS[k][0], key='sg_profile_metric')
        label, kind = V.PROFILE_METRICS[metric]
        fig = go.Figure(go.Bar(y=self.order, x=prof.loc[self.order, metric], orientation='h', marker_color=[self.col[s] for s in self.order],
                               text=[V.fmt(prof.loc[s, metric], kind) for s in self.order], textposition='outside', cliponaxis=False,
                               hovertemplate='%{y}: %{text}<extra></extra>'))
        fig.update_xaxes(title_text=label + (' (GBP)' if kind == 'gbp' else ''), tickformat='.0%' if kind == 'pct' else None)
        fig.update_yaxes(autorange='reversed'); fig.update_layout(height=360, showlegend=False)
        self.chart(fig, hovermode='closest')
        with st.expander('All measures in one table'):
            t = o['profiles'].set_index('segment').reindex(self.order).reset_index()
            self.export(t, 'segment_profiles')

        st.subheader('Segment AUM over time')
        mode = st.segmented_control('Show', ['Share of fund AUM', 'Amount (GBP)'], default='Share of fund AUM', key='sg_aum_mode')
        aum = o['seg_aum'].resample('ME').last()[self.order]
        share = (mode or 'Share of fund AUM') == 'Share of fund AUM'
        data = aum.div(aum.sum(axis=1), axis=0) if share else aum
        fig = go.Figure()
        for s in self.order:
            fig.add_scatter(x=data.index, y=data[s], name=s, mode='lines', line=dict(color=self.col[s], width=2),
                            hovertemplate=f'{s}: %{{y:{".1%" if share else ",.0f"}}}<extra></extra>')
        fig.update_yaxes(title_text='Share of AUM' if share else 'GBP', tickformat='.0%' if share else None)
        fig.update_xaxes(title_text='Month-end'); fig.update_layout(height=380)
        self.chart(fig)
        err = o['checks'].set_index('check').loc['Holdings = units × price', 'result'].split('; ')[-1]
        st.caption(f"**Reconstructed** from units × price at each month-end; the rebuilt total matches fund AUM ({err}). History uses today's "
                   'segment labels, so it shows what clients now in each segment held then (hindsight), not a segment record kept at the time.')

        st.subheader('Monthly net flow by segment')
        st.caption('Subscriptions minus redemptions by trade date. Choose segments to compare.')
        picks = st.pills('Segments', self.order, selection_mode='multi', default=self.order, key='sg_net_segs')
        if not picks:
            st.info('Select at least one segment to show its monthly net flow.')
        else:
            net = o['seg_net'].resample('ME').sum()
            fig = go.Figure()
            for s in [s for s in self.order if s in picks]:
                fig.add_scatter(x=net.index, y=net[s], name=s, mode='lines', line=dict(color=self.col[s], width=2),
                                hovertemplate=f'{s}: £%{{y:,.0f}}<extra></extra>')
            fig.update_yaxes(title_text='Net flow per month · GBP'); fig.update_xaxes(title_text='Month')
            fig.update_layout(height=380, showlegend=True)
            self.chart(fig)

        st.subheader('Which sectors are in each segment?')
        mix = o['sector_mix'].reindex(self.order)
        cv = o['validation'].set_index('attribute').cramers_v
        st.caption(f"Share of each segment's clients by sector (rows sum to 100%). Segments overlap with sector but are not the same thing. "
                   f"Cramér's V: sector {cv.get('client_sector', np.nan):.2f}, client type {cv.get('client_type', np.nan):.2f}, "
                   f"country {cv.get('country', np.nan):.2f} (0 = unrelated, 1 = identical).")
        fig = go.Figure(go.Heatmap(z=mix.values, x=list(mix.columns), y=list(mix.index), zmin=0, zmax=.6,
                                   colorscale=[[0, '#f7f8fa'], [1, '#16877c']], text=[[f'{v:.0%}' if v >= .005 else '' for v in r] for r in mix.values],
                                   texttemplate='%{text}', hovertemplate='%{y} · %{x}: %{z:.1%}<extra></extra>', colorbar=dict(title='Share', tickformat='.0%')))
        fig.update_yaxes(autorange='reversed'); fig.update_layout(height=380)
        self.chart(fig, hovermode='closest')

    def see_clients(self, segment):
        self.ss.sg_q_segs = [segment]
        self.ss.sg_q_page = 1
        self.ss.sg_section = V.SECTIONS[6]

    # ================================================================ 4 Segments in stress
    def reset_periods(self):
        self.ss.sg_periods = self._default_periods_frame(); self.ss.sg_dd = 0.10
        self.ss.sg_editor_version += 1

    def add_period(self):
        name, a, b = self.ss.get('sg_new_name', ''), self.ss.get('sg_new_from'), self.ss.get('sg_new_to')
        if not a or not b:
            self.ss.sg_add_msg = 'Choose both dates.'; return
        if a > b:
            self.ss.sg_add_msg = 'From must be on or before To.'; return
        new = pd.DataFrame([dict(use=True, name=name.strip() or 'Custom period', start=a, end=b)])
        self.ss.sg_periods = pd.concat([self.ss.sg_periods, new], ignore_index=True)
        self.ss.sg_editor_version += 1; self.ss.sg_add_msg = f'Added {name.strip() or "Custom period"}.'

    def stress(self):
        o = self.o; dates = self.engine.dates; lo, hi = dates.min().date(), dates.max().date()
        self.banner()
        st.subheader('Normal rhythm first: when do clients redeem?')
        st.caption('Flows as basis points of AUM, summed over 20 years. Year-ends (March, December) dominate. Stress behaviour below is measured against this.')
        m = o['month']
        measure = st.segmented_control('Show', ['Redemptions', 'Subscriptions', 'Both', 'Net flow'], default='Redemptions', key='sg_ctx_measure') or 'Redemptions'
        spec = {'Subscriptions': [('Subscriptions', 'sub_rate_bp', FLOW_COLOURS['sub'])], 'Redemptions': [('Redemptions', 'red_rate_bp', FLOW_COLOURS['red'])],
                'Net flow': [('Net flow', 'net_rate_bp', FLOW_COLOURS['net'])]}
        spec['Both'] = spec['Subscriptions'] + spec['Redemptions']
        fig = self.bars([MONTHS[i - 1] for i in m.month], [(n, m[c], colour) for n, c, colour in spec[measure]], 'Basis points of AUM', 'Calendar month',
                        height=280, hover='%{fullData.name}: %{y:.0f} bp<extra>%{x}</extra>')
        self.chart(fig)

        st.subheader('Define stress periods')
        st.caption('A day counts as **stress** if the fund price is far enough below its one-year high *or* it falls inside a ticked period. '
                   'Edit, untick, add or delete rows (select a row and press Delete); everything in this section and in section 5 recalculates. '
                   'Your definition lasts for this session; download it to keep it and upload it later.')
        options = [0.0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
        st.selectbox('Price trigger', options, key='sg_dd',
                     format_func=lambda v: 'Off (periods only)' if not v else f"{V.pct(v, 1)} below 1-year high{' (study default)' if v == 0.10 else ''}")
        edited = st.data_editor(self.ss.sg_periods, key=f'sg_period_editor_{self.ss.sg_editor_version}', num_rows='dynamic', hide_index=True,
                                width='stretch',
                                column_config={'use': st.column_config.CheckboxColumn('Use', default=True),
                                               'name': st.column_config.TextColumn('Period name', default='Custom period', required=True),
                                               'start': st.column_config.DateColumn('From', min_value=lo, max_value=hi, required=True),
                                               'end': st.column_config.DateColumn('To', min_value=lo, max_value=hi, required=True)})
        if not edited.reset_index(drop=True).astype(str).equals(self.ss.sg_periods.reset_index(drop=True).astype(str)):
            self.ss.sg_periods = edited.reset_index(drop=True)
            st.rerun()
        price = self.engine.price_mask(self.ss.sg_dd)
        rows = []
        for p in self.periods():
            inside = (self.engine.iso >= p.start) & (self.engine.iso <= p.end)
            n = int(inside.sum()); k = int((inside & price).sum())
            rows.append({'Period': p.name, 'Use': '✓' if p.on else '—', 'Fund days': 'From is after To' if p.start > p.end else f'{n:,}',
                         'Also price-triggered': '—' if p.start > p.end or not self.ss.sg_dd else f"{k:,}" + (f' ({k / n:.0%})' if n else '')})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        with st.form('sg_add_period', clear_on_submit=False, border=True):
            st.markdown('**Add a period**')
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.text_input('Period name', placeholder='e.g. Mini-budget aftermath', key='sg_new_name')
            c2.date_input('From', value=None, min_value=lo, max_value=hi, key='sg_new_from')
            c3.date_input('To', value=None, min_value=lo, max_value=hi, key='sg_new_to')
            st.form_submit_button('Add period', on_click=self.add_period)
        if self.ss.get('sg_add_msg'):
            st.caption(self.ss.sg_add_msg)
        c1, c2, c3 = st.columns(3)
        c1.button('Reset to study default', on_click=self.reset_periods, key='sg_reset')
        spec_json = json.dumps(dict(dd=self.ss.sg_dd, periods=[p.__dict__ for p in self.periods()]), indent=2)
        c2.download_button('Download definition · JSON', spec_json.encode(), 'stress_definition.json', 'application/json', key='sg_dl_def')
        up = c3.file_uploader('Load a saved definition', type=['json'], key='sg_up_def', label_visibility='collapsed')
        if up is not None and self.ss.get('sg_up_done') != up.file_id:
            try:
                spec = json.loads(up.getvalue())
                frame = pd.DataFrame([dict(use=bool(p['on']), name=str(p['name']), start=date.fromisoformat(p['start']), end=date.fromisoformat(p['end']))
                                      for p in spec['periods']])
                self.ss.sg_periods = frame; self.ss.sg_dd = float(spec.get('dd', 0)) if float(spec.get('dd', 0)) in options else 0.10
                self.ss.sg_editor_version += 1; self.ss.sg_up_done = up.file_id
                st.rerun()
            except (KeyError, TypeError, ValueError) as exc:
                st.warning(f'That file is not a saved stress definition ({exc}).')

        self.drawdown_chart(price)

        st.subheader('How each segment has redeemed in your stress periods')
        st.caption("Redemptions per day as basis points of the segment's AUM, on stress days vs calm days. **Month-matched** compares each stress day with calm "
                   "days in the same calendar month, so a stress period that falls in March isn't mistaken for stress behaviour. Above 1× means more "
                   'redemptions in stress. Value-weighted, so a few large redemptions can drive it.')
        lf = self.live['lift'].set_index('segment').reindex(self.order)
        if not lf.stress_bp_per_day.map(V.finite).any():
            st.info('No stress days under this definition, so stress and calm cannot be compared. Tick a period or turn on the price trigger.')
        else:
            fig = self.bars(self.order, [('Calm days', lf.calm_bp_per_day, V.SEGMENT_PALETTE[4]), ('Stress days', lf.stress_bp_per_day, STRIP_COLOURS['used'])],
                            'Redemptions, bp of AUM per day', 'Segment', height=340, hover='%{fullData.name}: %{y:.2f} bp/day<extra>%{x}</extra>')
            self.chart(fig)
            st.markdown(V.lift_answer(self.live['lift'], o['profiles']))
            med = self.prof.stress_lift
            word = lambda v: '' if not V.finite(v) else ' (more)' if v >= 1.15 else ' (less)' if v <= 0.87 else ''
            st.dataframe(pd.DataFrame({'Segment': self.order, 'Calm (bp/day)': lf.calm_bp_per_day.map(V.num2).values,
                                       'Stress (bp/day)': lf.stress_bp_per_day.map(V.num2).values,
                                       'Stress ÷ calm': [f'{v:.2f}×' if V.finite(v) else '—' for v in lf.value_lift],
                                       'Stress ÷ calm, same month': [f'{v:.2f}×{word(v)}' if V.finite(v) else '—' for v in lf.month_matched_value_lift],
                                       'Median client (study default)': [f'{med[s]:.2f}×' for s in self.order]}), hide_index=True, width='stretch')

        st.subheader('Do the segments pull money out at the same time?')
        st.markdown('If every segment tended to redeem on the same bad days, spreading clients across different groups would protect the fund less. '
                    'So we ask: **when one segment has pulled money out, have the others too?** Each bar is the average correlation between segments\' '
                    'flows (0 = independent, 1 = always together).')
        freq = st.segmented_control('Measure flows', ['daily', 'weekly'], default='daily', key='sg_together_freq',
                                    format_func={'daily': 'Day by day', 'weekly': 'Week by week'}.get) or 'daily'
        ans = V.together_answer(self.live['corr_summary'], freq)
        rc, rs, dc, ds = ans['values']
        fig = self.bars(['Actual flows', 'After removing calendar habits'], [('Calm days', [rc, dc], V.SEGMENT_PALETTE[4]), ('Stress days', [rs, ds], STRIP_COLOURS['used'])],
                        'Average correlation between segments', '', height=300, hover='%{fullData.name}: %{y:.3f}<extra>%{x}</extra>')
        self.chart(fig)
        with st.container(border=True):
            st.markdown('**Answer:** ' + ans['text'])
        st.caption('Correlation shows co-movement, not causation, and is not a probability that segments act together in future.')
        with st.expander('Show every pair of segments'):
            c1, c2 = st.columns(2)
            basis = c1.selectbox('Basis', ['raw', 'deseasonalised'], key='sg_c_basis', format_func={'raw': 'Actual flows', 'deseasonalised': 'After removing calendar habits'}.get)
            regime = c2.selectbox('Days', ['stress', 'calm', 'all'], key='sg_c_regime', format_func=str.capitalize)
            cm = self.live['corr'][(freq, basis, regime)].reindex(index=self.order, columns=self.order)
            n = int(self.live['corr_summary'].set_index(['frequency', 'basis', 'regime']).loc[(freq, basis, regime), 'periods'])
            if cm.notna().sum().sum() == 0:
                st.info(f'No valid correlations: at least 20 shared {"days" if freq == "daily" else "weeks"} are needed in this regime.')
            else:
                unit = 'days' if freq == 'daily' else 'weeks'
                fig = go.Figure(go.Heatmap(z=cm.values, x=self.order, y=self.order, zmin=-1, zmax=1, zmid=0, colorscale='RdBu',
                                           text=[[f'{v:.2f}' if V.finite(v) else '' for v in r] for r in cm.values], texttemplate='%{text}',
                                           hovertemplate=f'%{{y}} / %{{x}}<br>Correlation: %{{z:.3f}}<br>{n:,} {unit}<extra></extra>'))
                fig.update_yaxes(autorange='reversed'); fig.update_layout(height=460)
                self.chart(fig, hovermode='closest')
                st.caption(f'{n:,} {unit} · Pearson correlation, minimum 20 shared periods; blank cells mean too few.')

        st.subheader('Worst outflows inside each stress period')
        st.caption(f"The worst 5- and 20-day net outflow (subscriptions minus redemptions) ending inside each period, as a share of AUM then, "
                   f"and the same share applied to today's AUM ({self.money(self.aum)}).")
        rp = self.live['replay']
        if rp.empty:
            st.info('No active periods. Tick or add a period above.')
        else:
            st.dataframe(pd.DataFrame({'Period': rp.scenario.str.split(': worst').str[0], 'Horizon': rp.horizon_days.map('{} days'.format),
                                       'Window ended': rp.ended, 'Share of AUM': rp.outflow_share.map(lambda v: V.pct(v, 2)),
                                       "At today's AUM": rp.outflow_gbp.map(self.money)}), hide_index=True, width='stretch')

    def drawdown_chart(self, price):
        e = self.engine; used = self.live['stress'].to_numpy(); per = e.period_mask(self.periods())
        both = int((price & per).sum()); per_only = int((per & ~price).sum()); price_only = int((price & ~per).sum())
        c = st.columns(4)
        c[0].metric('Stress days in use', f'{int(used.sum()):,}')
        c[1].metric('Only from your periods', f'{per_only:,}')
        c[2].metric('Only from the price trigger', f'{price_only:,}')
        c[3].metric('Flagged by both', f'{both:,}')
        if both:
            st.caption('Unticking a period removes only its days that the price trigger does not also flag. Set the price trigger to *Off* to judge periods on their own.')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.73, .09, .09, .09], vertical_spacing=.025)
        fig.add_scatter(x=e.dates, y=100 * e.drawdown, mode='lines', line=dict(color='#14232f', width=1.5), name='Price vs trailing one-year high',
                        hovertemplate='%{x|%d %b %Y}: %{y:.1f}%<extra></extra>', row=1, col=1)
        strips = [('Stress days in use', used, STRIP_COLOURS['used']), ('From your periods', per, STRIP_COLOURS['periods']),
                  ('From price trigger', price, STRIP_COLOURS['price'])]
        def bands(mask, y0, y1):
            # One filled rectangle per contiguous run of flagged days, so gaps stay visible and the chart stays light.
            runs = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])); xs, ys = [], []
            for a, b in zip(runs[::2], runs[1::2]):
                xs += [e.dates[a], e.dates[a], e.dates[b - 1] + pd.Timedelta(days=1), e.dates[b - 1] + pd.Timedelta(days=1), None]; ys += [y0, y1, y1, y0, None]
            return xs, ys
        for r, (label, mask, colour) in enumerate(strips, start=2):
            xs, ys = bands(mask, 0, 1)
            fig.add_scatter(x=xs or [None], y=ys or [None], fill='toself', mode='lines', line=dict(width=0), fillcolor=colour,
                            name=f'{label} ({int(np.sum(mask)):,})', hoverinfo='skip', row=r, col=1)
            fig.update_yaxes(range=[0, 1], tickvals=[0.5], ticktext=[label], showgrid=False, zeroline=False, tickfont=dict(size=11), row=r, col=1)
        lo = float(np.nanmin(100 * e.drawdown)) * 1.05
        xs, ys = bands(used, lo, 0)
        if xs:
            fig.add_scatter(x=xs, y=ys, fill='toself', mode='lines', line=dict(width=0), fillcolor='rgba(210,107,69,0.16)',
                            name='Shaded: stress days in use', hoverinfo='skip', row=1, col=1)
        if self.ss.sg_dd:
            fig.add_hline(y=-100 * self.ss.sg_dd, line=dict(color='#5773b7', dash='dash', width=1),
                          annotation_text=f'−{V.pct(self.ss.sg_dd, 0)} price trigger', annotation_position='bottom left', row=1, col=1)
        fig.update_yaxes(title_text='% below one-year high', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)
        fig.update_layout(height=520, legend=dict(orientation='h', y=-0.12), title_text='Drawdown and stress days')
        self.chart(fig)

    # ================================================================ 5 Concentration & stress tests
    def set_buffer(self, value):
        self.ss.sg_buffer = float(value)

    def tests(self):
        o = self.o; conc = o['concentration']; c = self.conc
        self.banner()
        cols = st.columns(4)
        for col, n in zip(cols, (1, 10, 20, 100)):
            r = int(c[f'top{n}_rare_traders'])
            self.card(col, f"Top {n} clients hold" if n > 1 else 'Largest client holds', V.pct(c[f'top{n}_share']), f"{r} rare trader{'' if r == 1 else 's'} among them")

        st.subheader('How concentrated is the fund?')
        st.caption('Share of AUM held by the largest N clients at each year-end (reconstructed holdings; last point is 30 Oct 2026).')
        n = st.segmented_control('Largest clients', [1, 5, 10, 20, 50, 100], default=10, key='sg_top_n', format_func=lambda v: f'Top {v}') or 10
        labels = [d[:4] + ('' if d[5:7] == '12' else ' (Oct)') for d in conc.date]
        fig = go.Figure(go.Scatter(x=labels, y=conc[f'top{n}_share'], mode='lines+markers', name=f'Top {n} share of AUM', line=dict(color=V.SEGMENT_PALETTE[0], width=2),
                                   marker=dict(size=8), hovertemplate=f'Top {n}: %{{y:.1%}}<extra>%{{x}}</extra>'))
        fig.update_yaxes(title_text='Share of AUM', tickformat='.0%', rangemode='tozero'); fig.update_xaxes(title_text='Year-end')
        fig.update_layout(height=280, showlegend=False)
        self.chart(fig)
        with st.expander('Largest 100 holders and their segment'):
            seg = st.selectbox('Segment', ['All segments'] + self.order, key='sg_top_seg')
            t = self.top if seg == 'All segments' else self.top[self.top.segment == seg]
            show = t[['rank', 'client_name', 'client_id', 'client_sector', 'segment', 'holding_gbp', 'share']].rename(columns={
                'rank': 'Rank', 'client_name': 'Client', 'client_id': 'ID', 'client_sector': 'Sector', 'segment': 'Segment', 'holding_gbp': 'Holding (GBP)', 'share': 'Share of AUM'})
            if show.empty:
                st.info('None of the 100 largest holders is in this segment.')
            else:
                self.export(show, 'top_100_holders')

        st.subheader('Stress test settings')
        floors = self.meta['floors']
        with st.container(border=True):
            self.ss.setdefault('sg_buffer', 0.10)
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.slider('Cash buffer (share of AUM)', 0.01, 0.50, step=0.005, key='sg_buffer', format='%.3f')
            pay_on = c2.toggle(f"Pay known payables first ({self.money(self.base['known_payables'])})", value=True, key='sg_payables')
            hc = c3.selectbox('Fire-sale haircut', [0.01, 0.03, 0.05, 0.10], index=1, key='sg_haircut', format_func=lambda v: V.pct(v, 0))
            st.caption('Presets')
            presets = [(f'{V.pct(b, 0)}', b) for b in self.meta['buffers']] + [
                (f"{k.split(' · ')[-1].replace(' (current)', '').replace('US Rule 2a-7 (2023 amendments)', 'US 2a-7')} {V.pct(v[0], 1)}", v[0]) for k, v in floors.items()]
            cols = st.columns(len(presets))
            for i, (col, (label, b)) in enumerate(zip(cols, presets)):
                col.button(label, key=f'sg_buf_{i}', on_click=self.set_buffer, args=(b,), width='stretch',
                           help=(list(floors)[i - len(self.meta['buffers'])] + ' daily floor: money market fund rule, reference only') if i >= len(self.meta['buffers']) else None)
            st.caption('The buffer is the cash (or assets sellable within a day) assumed available. The database has no portfolio holdings, so it is illustrative. '
                       'Money-market-fund minimums are shown for reference only; this fund is not an MMF. The FCA\'s June 2026 position is to keep current '
                       'UK MMF minimums; further guidance is pending.')
        buf = float(self.ss.sg_buffer); B_ = buf * self.aum; pay = float(self.base['known_payables']) if pay_on else 0.0; avail = B_ - pay

        st.subheader('Reverse stress test: how many big clients could we pay out?')
        c1, c2 = st.columns(2)
        c1.markdown('**A normal stress test** asks: *if this bad thing happens, is there enough cash?* (below)')
        c2.markdown('**A reverse stress test** turns it round: *how bad would things have to get before the cash runs out?* Here, the biggest clients ask for '
                    '**all** their money back one after another, largest first, and we count how many could be paid before the buffer is empty.')
        rv = V.reverse_stress(self.top, avail); cum = rv['cum']
        nxt = rv['next']
        sentence = (f"With a **{self.money(B_)}** buffer" + (f" ({self.money(avail)} after {self.money(pay)} of known payables)" if pay else '')
                    + f", the fund could pay the **{rv['label']} largest client{'s' if rv['n'] != 1 else ''}** in full.")
        if nxt is not None:
            sentence += (f" The next one, **{nxt.client_name}** ({nxt.client_id}, {self.money(nxt.holding_gbp)}, {nxt.segment.lower()}), "
                         f"would take the running total to {self.money(cum[rv['n']])} and empty the buffer.")
        with st.container(border=True):
            st.markdown(sentence)
        k = min(len(self.top), max(12, rv['n'] + 4)); t = self.top.head(k)
        paid = cum[:k] <= avail
        fig = go.Figure()
        for flag, name, colour in [(True, 'Paid within the buffer', V.COVERED), (False, 'Buffer already empty', V.SHORTFALL)]:
            sel = t[paid == flag]
            fig.add_bar(y=[_short(f'{r.rank}. {r.client_name}', 26) for r in sel.itertuples()], x=sel.holding_gbp, orientation='h', name=name, marker_color=colour,
                        customdata=np.c_[sel.segment, [self.money(v) for v in cum[:k][paid == flag]], [f'{r.rank}. {r.client_name} ({r.client_id})' for r in sel.itertuples()]],
                        hovertemplate='%{customdata[2]}<br>%{customdata[0]}<br>Holding £%{x:,.0f}<br>Running total %{customdata[1]}<br>' + name + '<extra></extra>')
        fig.update_yaxes(categoryorder='array', categoryarray=[_short(f'{r.rank}. {r.client_name}', 26) for r in t.itertuples()], autorange='reversed', automargin=True)
        fig.update_xaxes(title_text='Holding that would be redeemed (GBP)')
        fig.update_layout(height=max(320, 26 * k), barmode='overlay', title_text='Largest clients, in the order they would be paid', legend=dict(orientation='h', y=-0.18))
        self.chart(fig, hovermode='closest')
        others = ' · '.join(f"**{V.pct(b, 0)}** → {V.reverse_stress(self.top, b * self.aum - pay)['label']}" for b in (0.05, 0.10, 0.20, 0.30))
        st.caption(f'How many of the largest clients other buffers would cover: {others}. Because a handful of clients hold so much, even large buffers cover '
                   'few full exits. That is why the biggest holders deserve individual attention.')

        st.subheader('Stress scenarios against the buffer')
        fams = st.pills('Scenario family', FAMILIES, selection_mode='multi', key='sg_fams')
        st.markdown('- **Concentration:** the top N clients redeem everything.\n- **Historical replay:** the worst 1/5/20-day net outflow on record, at today\'s size.\n'
                    '- **Stress period:** the worst outflow inside each of your stress periods (section 4).\n'
                    '- **Segment tails:** each segment\'s worst 1-in-100 outflow, all at once (ρ=1) or as correlated in past stress.\n'
                    '- **Combined:** a bad 5 days plus the top 10 leaving plus known payables.')
        T = self.scenarios(pay_on, pay, B_, hc)
        if fams:
            T = T[T.family.isin(fams)]
        fail = T[T.short > 0]
        c = st.columns(4)
        self.card(c[0], 'Buffer', self.money(B_), f'{V.pct(buf, 1)} of AUM')
        self.card(c[1], 'Scenarios the buffer covers', f'{len(T) - len(fail)} of {len(T)}',
                  f'largest shortfall {self.money(fail.short.max())}' if len(fail) else 'all covered')
        self.card(c[2], 'Worst scenario', self.money(T.total.max()) if len(T) else '—', 'including payables')
        self.card(c[3], 'Loss if assets must be sold', self.money(T.loss.max() if len(T) else 0), f'worst shortfall at a {V.pct(hc, 0)} haircut')
        if T.empty:
            st.info('Select at least one scenario family.')
        else:
            S = T.sort_values('total', ascending=False)
            fig = go.Figure()
            for flag, name, colour in [(False, 'Covered by the buffer', V.COVERED), (True, 'Exceeds the buffer', V.SHORTFALL)]:
                s = S[(S.short > 0) == flag]
                fig.add_bar(y=s.scenario.map(_scenario_label), x=s.total / self.aum, orientation='h', name=name, marker_color=colour,
                            customdata=np.c_[s.family, [self.money(v) for v in s.total], [self.money(v) for v in s.short], s.scenario],
                            hovertemplate='%{customdata[3]}<br>%{customdata[0]}<br>Outflow %{customdata[1]} (%{x:.1%} of AUM)' + ('<br>Shortfall %{customdata[2]}' if flag else '<br>Covered') + '<extra></extra>')
            fig.add_vline(x=buf, line=dict(color='#14232f', dash='dash', width=1), annotation_text=f'Buffer {V.pct(buf, 1)}', annotation_position='top')
            fig.update_yaxes(categoryorder='array', categoryarray=list(S.scenario.map(_scenario_label)), autorange='reversed', automargin=True)
            fig.update_xaxes(title_text='Outflow as a share of AUM', tickformat='.0%')
            fig.update_layout(height=max(360, 24 * len(S)), barmode='overlay', title_text='Scenario outflows', legend=dict(orientation='h', y=-0.12))
            self.chart(fig, hovermode='closest')
            with st.expander('Scenario table with second-round asset sales'):
                self.export(pd.DataFrame({'Family': T.family, 'Scenario': T.scenario, 'Outflow (GBP)': T.outflow_gbp.round(0), 'Share of AUM': T.outflow_share,
                                          'Incl. payables (GBP)': T.total.round(0), 'Shortfall (GBP)': T.short.round(0),
                                          f'Assets to sell, {V.pct(hc, 0)} haircut (GBP)': (T.short / (1 - hc)).round(0), 'Loss from selling (GBP)': T.loss.round(0)}),
                            'stress_scenarios')

    def scenarios(self, pay_on, pay, buffer_gbp, haircut):
        o = self.o; L = self.live
        base = o['tests'][(o['tests'].family != 'Named episode') & ~o['tests'].scenario.str.contains('historical stress correlation')]
        tails = pd.DataFrame([dict(family='Segment tails', scenario=f'{h}-day 99th pct, historical stress correlation', outflow_gbp=v,
                                   outflow_share=v / self.aum, payables_included_gbp=0.0) for h, v in L['tail'].items()])
        T = pd.concat([base, L['replay'].drop(columns=['horizon_days', 'ended']), tails], ignore_index=True)
        comb = T.family.eq('Combined')
        T['total'] = T.outflow_gbp + np.where(comb, 0.0 if pay_on else -T.payables_included_gbp, pay)
        T['short'] = (T.total - buffer_gbp).clip(lower=0)
        T['loss'] = np.where(T.short > 0, T.short * haircut / (1 - haircut), 0.0)
        return T

    # ================================================================ 6 Segment clients
    def apply_preset(self, name):
        p = V.PRESETS[name]
        self.ss.sg_q_segs = list(p['segs'])
        self.ss.sg_q_conds = self.conditions_frame(p['conds'])
        self.ss.sg_q_version = self.ss.get('sg_q_version', 0) + 1
        self.ss.sg_q_page = 1

    def clear_query(self):
        self.ss.sg_q_segs = []; self.ss.sg_q_conds = self.conditions_frame([]); self.ss.sg_q_search = ''
        for k in ('sectors', 'countries', 'aum', 'status', 'types', 'regions', 'units'):
            self.ss[f'sg_bg_{k}'] = []
        self.ss.sg_q_version = self.ss.get('sg_q_version', 0) + 1; self.ss.sg_q_page = 1

    @staticmethod
    def conditions_frame(conds):
        rows = []
        for f, op, v in conds:
            kind = V.QUERY_FIELDS[f][1]
            rows.append(dict(field=V.QUERY_FIELDS[f][0], op='at least' if op == '>=' else 'at most', value=100 * v if kind == 'pct' else v))
        return pd.DataFrame(rows, columns=['field', 'op', 'value'])

    def query(self):
        o = self.o; ss = self.ss; feat = o['features']; cl = self.clients
        ss.setdefault('sg_q_segs', []); ss.setdefault('sg_q_version', 0); ss.setdefault('sg_q_page', 1)
        if 'sg_q_conds' not in ss:
            ss.sg_q_conds = self.conditions_frame([])
        st.subheader('Find and query clients by segment')
        st.caption('Choose segments, narrow by background, add conditions on any behaviour, then sort. Select a client row for its full profile.')
        segs = st.pills('Segments', self.order, selection_mode='multi', key='sg_q_segs')
        st.caption('Quick queries')
        cols = st.columns(len(V.PRESETS) + 1)
        for col, name in zip(cols, V.PRESETS):
            col.button(name, key=f'sg_qp_{name}', on_click=self.apply_preset, args=(name,), width='stretch')
        cols[-1].button('Clear everything', key='sg_q_clear', on_click=self.clear_query, width='stretch')
        with st.expander('Client background'):
            c1, c2, c3, c4 = st.columns(4)
            bg = dict(sectors=c1.multiselect('Sector', sorted(cl.client_sector.unique()), key='sg_bg_sectors'),
                      countries=c2.multiselect('Country', sorted(cl.country.unique()), key='sg_bg_countries'),
                      aum=c3.multiselect('AUM at 30 Oct 2026', V.AUM_BANDS, key='sg_bg_aum'),
                      status=c4.multiselect('Account', ['Open', 'Closed'], key='sg_bg_status'))
            c1, c2, c3 = st.columns(3)
            bg.update(types=c1.multiselect('Client type', sorted(cl.client_type.unique()), key='sg_bg_types'),
                      regions=c2.multiselect('Region', sorted(cl.region.unique()), key='sg_bg_regions'),
                      units=c3.multiselect('Unit type', sorted(cl.unit_type.unique()), key='sg_bg_units'))
        st.markdown('**Conditions (all must be true)**')
        st.caption('Percent fields are entered as % (for example 40 for 40%); money in GBP. Add a row with ＋, delete with the row menu. '
                   + ' · '.join(f'{v[0]}: {v[2]}' for v in V.QUERY_FIELDS.values()))
        labels = {v[0]: k for k, v in V.QUERY_FIELDS.items()}
        conds_df = st.data_editor(ss.sg_q_conds, key=f'sg_q_editor_{ss.sg_q_version}', num_rows='dynamic', hide_index=True, width='stretch',
                                  column_config={'field': st.column_config.SelectboxColumn('Field', options=list(labels), required=True, default='Holding at 30 Oct 2026'),
                                                 'op': st.column_config.SelectboxColumn('Condition', options=['at least', 'at most'], required=True, default='at least'),
                                                 'value': st.column_config.NumberColumn('Value', required=True, default=0.0)})
        conds = []
        for r in conds_df.itertuples():
            if r.field in labels and r.op in ('at least', 'at most') and V.finite(r.value):
                f = labels[r.field]
                conds.append((f, '>=' if r.op == 'at least' else '<=', float(r.value) / (100 if V.QUERY_FIELDS[f][1] == 'pct' else 1)))
        c1, c2, c3 = st.columns([2, 2, 1])
        search = c1.text_input('Search', placeholder='Name or ID', key='sg_q_search')
        sort = c2.selectbox('Sort by', ['name'] + list(V.QUERY_FIELDS), index=1, key='sg_q_sort',
                            format_func=lambda k: 'Name' if k == 'name' else V.QUERY_FIELDS[k][0])
        desc = (c3.segmented_control('Order', ['Highest first', 'Lowest first'], default='Highest first', key='sg_q_dir') or 'Highest first') == 'Highest first'
        rows = V.query_clients(feat, cl, segs or [], bg, conds, search, sort, desc)
        tot = rows.current_holding_gbp.sum()
        mix = rows.segment.value_counts().reindex(self.order).dropna().astype(int)
        c = st.columns(4)
        self.card(c[0], 'Clients matching', f'{len(rows):,}', f'of {len(feat):,}')
        self.card(c[1], 'They hold', self.money(tot), f'{V.pct(tot / self.aum)} of AUM')
        self.card(c[2], 'Largest', self.money(rows.current_holding_gbp.max()) if len(rows) else '—', 'single holding')
        self.card(c[3], 'Segments represented', f'{len(mix)}', ' · '.join(f'{s} {n}' for s, n in mix.head(3).items()) or 'none')
        if rows.empty:
            st.info('No clients match. Remove a condition or widen the filters.')
            return
        pages = max(1, -(-len(rows) // 25))
        page = int(min(max(1, ss.sg_q_page), pages))
        ss.sg_q_page = page
        cols_show = ['current_holding_gbp', 'trades_per_year', 'redemption_share', 'largest_redemption_rel', 'stress_lift', 'red_december_share']
        chunk = rows.iloc[(page - 1) * 25: page * 25]
        table = pd.DataFrame({'Client': chunk.client_name.values, 'ID': chunk.index, 'Sector': chunk.client_sector.values, 'Segment': chunk.segment.values,
                              **{V.QUERY_FIELDS[c][0].replace(' at 30 Oct 2026', ' now'): [V.fmt(v, V.QUERY_FIELDS[c][1]) for v in chunk[c]] for c in cols_show}})
        event = st.dataframe(table, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key=f'sg_q_table_{page}')
        c1, c2, c3 = st.columns([1, 2, 2])
        c1.number_input(f'Page (of {pages})', 1, pages, key='sg_q_page')
        export = rows.reset_index()[['client_id', 'client_name', 'client_sector', 'segment'] + list(V.QUERY_FIELDS)]
        c3.download_button('Download results · CSV', export.to_csv(index=False).encode(), 'segment_clients.csv', 'text/csv', key='sg_q_csv')
        picked = event.selection.rows if event and event.selection else []
        if picked:
            self.profile(chunk.index[picked[0]])

    def profile(self, cid):
        f = self.o['features'].loc[cid]; cl = self.clients.loc[cid]; s = f.segment; p = self.prof.loc[s]
        v, _ = V.verdict(s, self.o['cohesion'], self.o['random_pair_r'])
        st.subheader(cl.client_name)
        st.caption(f'{cid} · {cl.client_sector} · {cl.client_type} · {cl.region}')
        rank = self.top.set_index('client_id').get('rank', pd.Series(dtype=int)).get(cid)
        c = st.columns(4)
        self.card(c[0], 'Segment', s, v)
        self.card(c[1], 'Holding now', self.money(f.current_holding_gbp), f'rank {rank} by size' if rank else 'rank >100 by size')
        self.card(c[2], 'Trades', V.fmt(f.trades, 'int'), f'{V.num2(f.years_active)} years active')
        self.card(c[3], 'Stress lift', V.num2(f.stress_lift), '1 = same redemption rate in stress')
        pairs = {'trades_per_year': 'trades_per_year', 'redemption_share': 'redemption_share', 'median_trade_rel_holding': 'trade_rel_holding',
                 'largest_redemption_rel': 'largest_red_rel', 'stress_lift': 'stress_lift', 'red_march_share': 'march_red_share',
                 'red_december_share': 'december_red_share', 'month_end_trade_share': 'month_end_share', 'mean_holding_gbp': 'median_holding'}
        st.dataframe(pd.DataFrame([{'Habit': V.QUERY_FIELDS[k][0], 'This client': V.fmt(f[k], V.QUERY_FIELDS[k][1]),
                                    f'Typical {s}': V.fmt(p[m], V.QUERY_FIELDS[k][1]), 'What it means': V.QUERY_FIELDS[k][2]} for k, m in pairs.items()]),
                     hide_index=True, width='stretch')
        if s == V.RARE:
            st.caption('Rare traders have too little history to be grouped, so their numbers say little about how they might redeem.')
