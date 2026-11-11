"""Segments & stress: pipeline checks, headline numbers, live stress-engine parity, query tool and page render.

Ported from handoff/segments_stress/reference_ui/check_segments.cjs. Headline values are this repository's
(k fixed at 7, 50 k-means restarts); README "Segments & stress" lists where they differ from the study.
"""
import re
import statistics
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fund_analytics.segments import bundle as B
from fund_analytics.segments import pipeline as S
from fund_analytics.segments import views as V
from fund_analytics.segments.engine import Period, StressEngine, default_definition

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'handoff/segments_stress/data_outputs'


class SegmentStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = S.run(write=False)
        cls.bundle = B.make_bundle(cls.out)
        cls.engine = StressEngine(cls.bundle)
        cls.default = default_definition(cls.bundle['meta']['episodes'])

    # ---------------------------------------------------------------- 1 pipeline
    def test_pipeline_data_checks_pass(self):
        checks = self.out['checks']
        self.assertTrue(checks.passed.all(), checks[~checks.passed])
        self.assertIn('Holdings = units × price', set(checks.check))

    # ---------------------------------------------------------------- 2 headline numbers
    def test_headline_numbers(self):
        o, meta = self.out, self.bundle['meta']
        self.assertEqual(meta['k'], 7)
        rare = o['profiles'].set_index('segment').loc['Rare traders']
        self.assertEqual(int(rare.clients), 2834)
        self.assertAlmostEqual(rare.aum_share, 0.273, places=3)
        self.assertEqual(o['ctx']['stress_days'], 781)
        self.assertAlmostEqual(float(np.median(o['ari'])), 0.70, delta=0.03)  # study: 0.72
        last = o['concentration'].iloc[-1]
        self.assertEqual(last.date, '2026-10-30')
        self.assertAlmostEqual(last.top10_share, 0.134, places=3)
        self.assertAlmostEqual(last.top100_share, 0.407, places=3)
        rev = o['reverse'].set_index('buffer').loc['10% of AUM']
        self.assertEqual(int(rev.clients_before_exhausted), 6)
        self.assertAlmostEqual(o['base']['known_payables'], 11.9e6, delta=0.05e6)
        seventh = o['top'].iloc[6]
        self.assertEqual(seventh.client_id, 'C024578')
        self.assertEqual(self.bundle['clients'].client_name['C024578'], 'Larchcroft Family Foundation')
        cs = o['corr_summary'].set_index(['frequency', 'basis', 'regime']).mean_pairwise_r
        self.assertEqual(round(cs['daily', 'raw', 'all'], 2), 0.11)
        self.assertEqual(round(cs['daily', 'raw', 'stress'], 2), 0.13)  # study: 0.12
        self.assertEqual(round(cs['daily', 'deseasonalised', 'all'], 2), 0.05)
        eta = o['grouping'].groupby('grouping').eta_squared.mean()
        self.assertEqual(round(100 * eta['Behavioural segments (formed 2006–2016)']), 31)
        self.assertEqual(round(100 * eta['Sector']), 23)
        self.assertEqual(round(100 * eta['Client type']), 24)
        self.assertLess(eta['Region'], 0.01)

    def test_segments_match_the_study(self):
        mine = self.out['features'].segment
        if not (HANDOFF / '11_client_segments.csv').exists():
            self.skipTest('handoff outputs not present')
        theirs = pd.read_csv(HANDOFF / '11_client_segments.csv', index_col='client_id').segment
        self.assertEqual(set(mine), set(theirs))
        self.assertGreaterEqual((mine.reindex(theirs.index) == theirs).mean(), 0.95)

    def test_bundle_integrity(self):
        o = self.out
        self.assertEqual(int(o['profiles'].clients.sum()), len(self.bundle['clients']))
        for _, r in o['concentration'].iterrows():
            shares = [r[f'top{n}_share'] for n in (1, 5, 10, 20, 50, 100)]
            self.assertEqual(shares, sorted(shares))
        order = self.bundle['meta']['order']
        self.assertEqual(order[-1], 'Rare traders')
        self.assertEqual(set(order), set(o['profiles'].segment))
        self.assertEqual(len(set(V.colours(order).values())), len(order))

    # ---------------------------------------------------------------- 3 engine parity
    def test_engine_reproduces_pipeline_with_default_definition(self):
        o, e, d = self.out, self.engine, self.default
        r = e.run(d['dd'], d['periods'])
        self.assertEqual(r['days'], o['ctx']['stress_days'])
        self.assertTrue((r['stress'].to_numpy() == o['flags']['stress'].to_numpy()).all())
        m = r['corr_summary'].merge(o['corr_summary'], on=['frequency', 'basis', 'regime'], suffixes=('_e', '_p'))
        self.assertEqual(len(m), 12)
        self.assertTrue((m.periods_e == m.periods_p).all())
        for c in ('mean_pairwise_r', 'min_r', 'max_r'):
            self.assertLess((m[c + '_e'] - m[c + '_p']).abs().max(), 1e-5, c)
        for key, matrix in o['corr'].items():
            mine = r['corr'][key].reindex(index=matrix.index, columns=matrix.columns)
            self.assertTrue((mine.isna() == matrix.isna()).all().all(), key)
            self.assertLess(np.nanmax(np.abs((mine - matrix).to_numpy())), 1e-5, key)
        tests = o['tests'].set_index('scenario').outflow_gbp
        for row in r['replay'].itertuples():
            self.assertLess(abs(tests[row.scenario] - row.outflow_gbp), 1, row.scenario)
        self.assertEqual(len(r['replay']), 8)
        for h, v in r['tail'].items():
            self.assertLess(abs(tests[f'{h}-day 99th pct, historical stress correlation'] - v), 1)
        lift = r['lift'].set_index('segment'); ref = o['value_lift'].set_index('segment')
        self.assertLess((lift.value_lift - ref.value_lift).abs().max(), 1e-5)
        self.assertLess((lift.month_matched_value_lift - ref.month_matched_value_lift).abs().max(), 1e-5)

    def test_unticking_euro_crisis_removes_its_non_price_days(self):
        periods = [Period(p.name, p.start, p.end, p.name != 'Euro-area debt crisis') for p in self.default['periods']]
        self.assertEqual(int(self.engine.flags(0.10, periods).sum()), 716)

    def test_custom_definition(self):
        cv = self.bundle['meta']['episodes']['COVID-19 market shock']
        periods = [Period('COVID', cv[0], cv[1]), Period('Test period', '2015-01-05', '2015-02-27'), Period('Off', '2010-01-01', '2010-12-31', False)]
        iso = self.engine.iso
        expected = int((((iso >= cv[0]) & (iso <= cv[1])) | ((iso >= '2015-01-05') & (iso <= '2015-02-27'))).sum())
        self.assertEqual(int(self.engine.flags(0, periods).sum()), expected)
        self.assertEqual(len(self.engine.replay(periods)), 4)
        self.assertEqual(int(self.engine.flags(0, [Period(p.name, p.start, p.end, False) for p in periods]).sum()), 0)
        empty = self.engine.run(0, [])
        self.assertEqual(empty['days'], 0)
        self.assertIn('No stress days', V.lift_answer(empty['lift'], self.out['profiles']))
        self.assertIn('Not enough', V.together_answer(empty['corr_summary'], 'daily')['text'])

    def test_engine_recompute_is_fast(self):
        d = self.default; times = []
        for _ in range(5):
            t = time.perf_counter(); self.engine.run(d['dd'], d['periods']); times.append(time.perf_counter() - t)
        self.assertLess(statistics.median(times), 0.1)

    # ---------------------------------------------------------------- 4 query tool and reverse stress
    def test_big_rare_traders_query(self):
        p = V.PRESETS['Big rare traders']
        rows = V.query_clients(self.out['features'], self.bundle['clients'], p['segs'], None, p['conds'])
        f = self.out['features']
        direct = f[(f.segment == 'Rare traders') & (f.current_holding_gbp >= 5e6)]
        self.assertEqual(len(rows), 19)
        self.assertEqual(len(rows), len(direct))
        self.assertAlmostEqual(rows.current_holding_gbp.sum() / 1e6, 382, delta=1)

    def test_query_filters_and_sorts(self):
        f, cl = self.out['features'], self.bundle['clients']
        self.assertTrue(V.query_clients(f, cl, conds=[('current_holding_gbp', '>=', 1e12)]).empty)
        rows = V.query_clients(f, cl, conds=[('stress_lift', '>=', 1.5), ('red_december_share', '<=', 0.1)], sort='stress_lift')
        self.assertTrue(rows.stress_lift.is_monotonic_decreasing)
        sector = cl.client_sector.iloc[0]
        rows = V.query_clients(f, cl, background={'sectors': [sector], 'status': ['Open'], 'aum': ['£1m–£10m']})
        self.assertTrue((rows.client_sector == sector).all() and rows.account_close_date.isna().all())
        self.assertTrue(rows.client_aum_gbp.between(1e6, 1e7, inclusive='left').all())
        hit = V.query_clients(f, cl, search='larchcroft family')
        self.assertIn('C024578', hit.index)

    def test_reverse_stress_view_matches_pipeline(self):
        aum, pay = self.out['base']['aum'], self.out['base']['known_payables']
        for r in self.out['reverse'].itertuples():
            if r.buffer.endswith('of AUM'):
                b = float(r.buffer.split('%')[0]) / 100
                self.assertEqual(V.reverse_stress(self.out['top'], b * aum - pay)['n'], r.clients_before_exhausted)
        rv = V.reverse_stress(self.out['top'], 0.10 * aum - pay)
        self.assertEqual(rv['next'].client_id, 'C024578')


class SegmentPageRenderTests(unittest.TestCase):
    """Every section and its main controls render with no NaN/undefined/Infinity text or forecasting wording."""
    BAD = re.compile(r'NaN|\bnan\b|undefined|Infinity|\binf\b|\[object')
    FORECASTY = re.compile(r'\bwill\b|\bpredict|expected outflow', re.I)

    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'data/clean/fund_analytics.duckdb').exists() or not B.BUNDLE.exists():
            raise unittest.SkipTest('Build the database and the segment study first (ingest, then segments).')
        from streamlit.testing.v1 import AppTest
        cls.at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180)
        cls.at.run()
        cls.at.sidebar.radio[0].set_value('Segments & stress').run()

    def texts(self):
        at = self.at; out = []
        for kind in ('markdown', 'caption', 'info', 'warning', 'error', 'subheader'):
            out += [str(e.value) for e in getattr(at, kind)]
        out += [f'{m.label} {m.value} {m.delta}' for m in at.metric]
        return out

    def check(self, label):
        self.assertFalse(self.at.exception, f'{label}: {[e.value for e in self.at.exception]}')
        for t in self.texts():
            self.assertIsNone(self.BAD.search(t), f'{label}: {t[:200]}')
            if 'did not improve forecasts' not in t:
                self.assertIsNone(self.FORECASTY.search(t), f'{label}: {t[:200]}')

    def go(self, section):
        self.at.session_state['sg_section'] = section
        self.at.run()

    def test_every_section_renders(self):
        for s in V.SECTIONS:
            self.go(s); self.check(s)
        self.go(V.SECTIONS[0])
        self.assertTrue(any('not a forecast' in t for t in self.texts()))

    def test_section_controls(self):
        at = self.at
        self.go(V.SECTIONS[4])
        for measure in ('Redemptions', 'Subscriptions', 'Both', 'Net flow'):
            for freq in ('daily', 'weekly'):
                at.session_state['sg_ctx_measure'] = measure; at.session_state['sg_together_freq'] = freq
                at.run(); self.check(f'stress {measure} {freq}')
                self.assertTrue(any(t.startswith('**Answer:**') for t in self.texts()))
        self.assertTrue(any('**781** stress days' in t for t in self.texts()))
        periods = at.session_state['sg_periods'].copy()
        periods.loc[periods.name == 'Euro-area debt crisis', 'use'] = False
        at.session_state['sg_periods'] = periods; at.session_state['sg_editor_version'] += 1; at.run()
        self.assertTrue(any('**716** stress days' in t for t in self.texts()))
        at.session_state['sg_dd'] = 0.0; at.run(); self.check('price trigger off')
        periods['use'] = False
        at.session_state['sg_periods'] = periods; at.session_state['sg_editor_version'] += 1; at.run()
        self.check('no stress days')
        self.assertTrue(any('**0** stress days' in t for t in self.texts()))
        at.button(key='sg_reset').click().run()
        self.assertTrue(any('**781** stress days' in t for t in self.texts()))

        self.go(V.SECTIONS[3])
        for metric in V.PROFILE_METRICS:
            at.session_state['sg_profile_metric'] = metric; at.run(); self.check(metric)
        at.session_state['sg_net_segs'] = []; at.run()
        self.assertTrue(any('Select at least one segment' in t for t in self.texts()))

        self.go(V.SECTIONS[5])
        for b in (0.05, 0.10, 0.20, 0.30):
            for pay in (True, False):
                at.session_state['sg_buffer'] = b; at.session_state['sg_payables'] = pay; at.run(); self.check(f'buffer {b} {pay}')
        at.session_state['sg_buffer'] = 0.10; at.session_state['sg_payables'] = True; at.run()
        self.assertTrue(any('largest clients** in full' in t and 'Larchcroft Family Foundation' in t and '**6 ' in t for t in self.texts()))

        self.go(V.SECTIONS[6])
        at.button(key='sg_qp_Big rare traders').click().run(); self.check('big rare traders')
        matching = next(m for m in at.metric if m.label == 'Clients matching')
        self.assertEqual(matching.value, '19')
        at.button(key='sg_q_clear').click().run(); self.check('cleared')
