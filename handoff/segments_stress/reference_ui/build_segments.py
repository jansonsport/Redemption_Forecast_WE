"""Dashboard bundle for the Segments & stress page (Notebook 11 outputs).

Recomputes nothing statistical: reads the run of notebooks/segment_stress.py (re-running it if its
tables are missing) and writes dist/segments-data.js.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / 'notebooks'))
import segment_stress as S
o = S.run(write=True)
num = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
def recs(df): return [{k: (num(v) if not isinstance(v, (str, bool, np.bool_)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in r.items()} for r in df.to_dict('records')]
segs = list(o['profiles'].segment)
order = [s for s in o['profiles'].sort_values('aum_gbp', ascending=False).segment if s != 'Rare traders'] + ['Rare traders']
fl = o['flags']; fund_d = pd.read_csv(ROOT / 'synthetic_fund_daily.csv', parse_dates=['date']).set_index('date').reindex(o['seg_aum'].index); aum_m = o['seg_aum'].resample('M').last(); net_m = o['seg_net'].resample('M').sum()
feat = o['features']; R = pd.read_csv(ROOT / 'synthetic_clients.csv', usecols=['client_id']).sort_values('client_id').client_id
f = feat.reindex(R)
CC = ['current_holding_gbp', 'mean_holding_gbp', 'trades', 'years_active', 'trades_per_year', 'redemption_share', 'median_trade_rel_holding', 'largest_redemption_rel', 'stress_lift', 'red_march_share', 'red_december_share', 'month_end_trade_share']
corr = {'|'.join(k): {'labels': list(m.index), 'r': [[num(x) for x in row] for row in m.values]} for k, m in o['corr'].items()}
data = dict(
    report=dict(ctx={k: num(v) if not isinstance(v, str) else v for k, v in o['ctx'].items()}, base={k: v if isinstance(v, (str, dict)) else num(v) for k, v in o['base'].items()},
                k=int(o['selection'].query('chosen').k.iloc[0]), ari=[num(x) for x in o['ari']], config=S.CONFIG['name'], eligibility_rule=S.CONFIG['eligibility'],
                episodes=S.CONFIG['named_episodes'], floors={k: list(v) for k, v in S.CONFIG['floors'].items()}, previous=S.CONFIG['previous']),
    order=order, inventory=recs(o['inventory']), checks=recs(o['checks']), month=recs(o['month']),
    flags=dict(dates=[d.strftime('%Y-%m-%d') for d in fl.index], drawdown=[num(x) for x in fl.drawdown_1y], stress=[int(x) for x in fl.stress], episode=list(fl.episode_name)),
    eligibility=recs(o['eligibility']), selection=recs(o['selection']), profiles=recs(o['profiles']), validation=recs(o['validation']),
    sector_mix=dict(segments=list(o['sector_mix'].index), sectors=list(o['sector_mix'].columns), share=[[num(x) for x in r] for r in o['sector_mix'].values]),
    seg_aum=dict(months=[d.strftime('%Y-%m') for d in aum_m.index], series={s: [num(x) for x in aum_m[s]] for s in aum_m}),
    seg_net=dict(months=[d.strftime('%Y-%m') for d in net_m.index], series={s: [num(x) for x in net_m[s]] for s in net_m}),
    corr=corr, corr_summary=recs(o['corr_summary']), concentration=recs(o['concentration']), top=recs(o['top']),
    cohesion=recs(o['cohesion']), random_pair_r=o['random_pair_r'],
    pca=dict(points=dict(id=list(o['pca']['points'].client_id), x=[round(float(v), 3) for v in o['pca']['points'].x], y=[round(float(v), 3) for v in o['pca']['points'].y], seg=[order.index(v) for v in o['pca']['points'].segment]),
             centres=recs(o['pca']['centres']), loadings=recs(o['pca']['loadings']), explained=o['pca']['explained']),
    value_lift=recs(o['value_lift']), typical=o['typical'], grouping=recs(o['grouping']), grouping_clients=o['grouping_clients'],
    daily=dict(dates=[d.strftime('%Y-%m-%d') for d in o['seg_aum'].index],
               price=[num(x) for x in fund_d['accumulation_unit_price_gbp']], fund_sub=[round(float(x), 2) for x in fund_d['subscriptions_trade_date_gbp']],
               fund_red=[round(float(x), 2) for x in fund_d['redemptions_trade_date_gbp']], fund_open=[round(float(x), 2) for x in fund_d['opening_aum_gbp']],
               seg_aum={g: [round(float(x), 2) for x in o['seg_aum'][g]] for g in order}, seg_net={g: [round(float(x), 2) for x in o['seg_net'][g]] for g in order},
               seg_red={g: [round(float(x), 2) for x in o['seg_red'][g]] for g in order}),
    tests=recs(o['tests']), tails=recs(o['tails']), reverse=recs(o['reverse']),
    clients=dict(ids=list(R), segment=[order.index(s) if isinstance(s, str) else -1 for s in f.segment],
                 cols=CC, values=[[num(x) for x in f[c]] for c in CC]),
)
txt = 'window.SEGMENTS=' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n'
(ROOT / 'fund-manager-ui/dist/segments-data.js').write_text(txt)
assert o['checks'].passed.all()
import shutil
for n in ['11_client_segments.csv', '11_stress_tests.csv', '11_segment_profiles.csv', '11_concentration.csv', '11_segment_correlations.csv']: shutil.copy(ROOT / 'outputs/tables' / n, ROOT / 'fund-manager-ui/dist' / n)
if (ROOT / 'outputs/11_segments_correlation_stress.html').exists(): shutil.copy(ROOT / 'outputs/11_segments_correlation_stress.html', ROOT / 'fund-manager-ui/dist/11_segments_correlation_stress.html')
print(f'Segments bundle: k={data["report"]["k"]}, {len(order)} segments, {len(data["tests"])} stress scenarios, {len(txt)/1e6:.1f} MB.')
