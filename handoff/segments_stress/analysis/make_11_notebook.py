import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from pathlib import Path
ROOT = Path('/Users/gabrielsiallagan/Desktop/MMFproject')
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
cells = [
md("""# Client segments, flow correlation, concentration and liquidity stress

This notebook works through the eight-phase brief: inventory, data checks, fund context, client features, segmentation, segment flow correlation, concentration, stress tests and write-up. All logic lives in `notebooks/segment_stress.py`. **Every table and column name is taken from one `CONFIG` dictionary (Phase 0)**, so the notebook runs on a different fund database by editing a single cell.

Amounts are in GBP. Behaviour, segments, correlation and stress use **trade dates** (when the instruction is given). Cash-date reconciliation is checked separately. The data are synthetic."""),
code("""from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = Path.cwd() if (Path.cwd()/'synthetic_fund_daily.csv').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT/'notebooks'))
import segment_stress as S
FIG = ROOT/'outputs/figures'
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
gbp = lambda v: f'£{v/1e6:,.1f}m'
pct = lambda v: f'{100*v:.1f}%'"""),
md("""## Phase 0 · Inventory and mapping

The analysis needs four kinds of information. The configuration below maps this database onto them: column names, date format, the sign convention for redemptions, which statuses count, and which date drives behaviour. **To run on another database, edit only this cell.**"""),
code("""CONFIG = S.CONFIG   # edit S.CONFIG fields here for another database, e.g. CONFIG = {**S.CONFIG, 'files': {...}}
print(json.dumps({k: v for k, v in CONFIG.items() if k in ['name', 'files', 'flows', 'redemption_sign', 'keep_status', 'behaviour_date', 'currency']}, indent=1, default=str))
o = S.run(CONFIG, write=True)
display(o['inventory'])"""),
md("""All four roles are present. Unit prices exist for both share classes, so holdings can be rebuilt as units × price. There is no units column, so units are derived from balance-after-trade ÷ class price. There is no flow-type or currency column, so internal moves and currencies are checked heuristically."""),
md("""## Phase 1 · Data checks

The analysis does not continue until client flows reconcile to fund flows and holdings reconcile to units × price."""),
code("""display(o['checks'])
assert o['checks'].passed.all(), 'Data checks failed: stop here'
print('Unit reconstruction:', {k: round(v, 6) for k, v in o['unit_check'].items()})"""),
md("""* **Status:** every transaction is settled. The filter is kept so that cancelled, pending or reversed rows would be dropped in another database.
* **Duplicates:** none by ID or by content.
* **Internal moves:** the database has no flow-type column. Same-day equal-and-opposite amounts between different clients are counted as *candidates* but not removed. With 900k transactions, many small amounts coincide by chance. In a database with switch or transfer codes, set `flows.flow_type` and these rows are netted out.
* **Omnibus / nominee accounts:** no client type suggests a platform or nominee, and no top-1% holder has an extreme trade count.
* **Currency:** single currency (GBP).
* **Trade vs cash date:** client flows reconcile to fund flows on *both* dates to the penny. Behaviour uses trade date; cash timing (settlement lag) matters only for when cash leaves.
* **Holdings:** daily units × price reproduce reported fund AUM every day, within rounding. Segment AUM history (Phase 5) depends on this."""),
md("""## Phase 2 · Fund context and stress windows

The fund type decides how stress is defined. A money market fund's price is flat, so it cannot signal stress, and only external markers (named episodes, market indicators) would be used. This fund's price is volatile, so it is **not** an MMF. Stress days are those where the accumulation price is at least 10% below its trailing one-year high, **plus** named market episodes (GFC, euro-area crisis, COVID-19, the 2022 UK gilt/LDI crisis)."""),
code("""display(pd.Series(o['ctx']))
m = o['month']; fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(m.month - .2, m.sub_rate_bp, .4, label='Subscriptions'); ax[0].bar(m.month + .2, m.red_rate_bp, .4, label='Redemptions')
ax[0].set_xticks(range(1, 13)); ax[0].set_ylabel('Daily flow, bp of AUM (summed over history)'); ax[0].set_title('Seasonality by calendar month'); ax[0].legend()
fl = o['flags']; ax[1].plot(fl.index, 100*fl.drawdown_1y, lw=.8, color='#0f2a4a'); ax[1].fill_between(fl.index, -35, 0, where=fl.stress, color='#e34948', alpha=.2, label='Stress day')
ax[1].set_ylabel('Price vs 1-year high (%)'); ax[1].set_title('Stress windows'); ax[1].legend(); fig.tight_layout(); fig.savefig(FIG/'11_context.png', dpi=110); plt.show()
display(m.round(2))"""),
md("""## Phase 3 · Client features

The core features are holding size, trades per year, redemption share, trade size relative to holding, largest redemption (relative to the average holding) and **stress lift**. Stress lift is a client's redemption rate on stress days divided by its rate on calm days, shrunk towards 1 so that small clients do not produce extreme ratios.

The calendar features fit a charity client base: the share of redemptions in March and in December (common financial year-ends), and the share of trades in the last three business days of a month or quarter. Corporate clients would instead get tax and payroll dates. **Yield sensitivity** (flows against the gap between fund yield and Bank Rate) applies to MMFs and needs a yield series, which this database lacks, so it is left empty.

**Eligibility rule recheck.** The table shows how many clients pass different thresholds of trades and years, and how much AUM the excluded *rare traders* hold."""),
code("""display(o['eligibility'].assign(eligible_share=lambda d: d.eligible_share.map(pct), rare_aum_share=lambda d: d.rare_aum_share.map(pct)))
f = o['features']; print(f"Rule used: >= {CONFIG['eligibility']['min_trades']} trades over >= {CONFIG['eligibility']['min_years']} year(s): {f.eligible.sum():,} eligible; rare traders hold {pct(f.loc[~f.eligible,'current_holding_gbp'].sum()/f.current_holding_gbp.sum())} of AUM")"""),
md("""Clients are many and active (median 75 trades), so the original **8 trades over at least 1 year** bar does not need lowering for this fund. Rare traders are **kept as their own segment** rather than dropped, because they hold about 27% of AUM, the same share found last time."""),
md("""## Phase 4 · Segmentation

Features are log-transformed where skewed, standardised, then clustered with k-means for k = 3–8. k is chosen as the best silhouette among *usable* solutions (every segment at least 3% of eligible clients). **Stability:** k-means is refitted on bootstrap resamples and every client is reassigned. The adjusted Rand index (ARI) measures agreement with the reference labels (1 = identical, 0 = chance)."""),
code("""display(o['selection'].round(3))
a = o['ari']; print(f"Chosen k = {int(o['selection'].query('chosen').k.iloc[0])}; bootstrap ARI over {len(a)} resamples: median {np.median(a):.2f}, 10th percentile {np.percentile(a,10):.2f}, min {a.min():.2f}")
p = o['profiles'].copy(); display(p.assign(aum_gbp=p.aum_gbp.map(gbp), aum_share=p.aum_share.map(pct), median_holding=p.median_holding.map(lambda v: f'£{v:,.0f}')).round(3))"""),
md("""**Naming.** Segments are named from their profile: the feature on which each segment stands out most, in standard deviations.
**Validation against client attributes** (Cramér's V: 0 = unrelated, 1 = determined):"""),
code("""display(o['validation'].round(3)); display((100*o['sector_mix']).round(0))
fig, ax = plt.subplots(figsize=(14, 4)); v = o['seg_aum'].resample('M').last()/1e9
ax.stackplot(v.index, v.T.values, labels=v.columns); ax.set_ylabel('AUM (£bn)'); ax.set_title('Rebuilt AUM by segment (units × price)'); ax.legend(loc='upper left', fontsize=8, ncol=4)
fig.tight_layout(); fig.savefig(FIG/'11_segment_aum.png', dpi=110); plt.show()"""),
md("""**Why segments rather than sector?** A fair, out-of-time test: segments are formed from behaviour in 2006–2016. Then segments, sector, client type and region are each scored on how much of every client's 2016–2026 behaviour they explain (η², the between-group share of variance), for clients eligible in both periods."""),
code("""g = o['grouping'].pivot(index='feature', columns='grouping', values='eta_squared'); display((100*g).round(1)); display((100*g.mean()).round(1).rename('mean % explained'))
print(f"{o['grouping_clients']:,} clients eligible in both periods")"""),
md("""Segments explain more of later behaviour overall, especially trading frequency, trade and holding size and month-end habits (*how* clients use the fund). Sector and client type explain year-end timing (March and December redemptions) better: *when* clients redeem is set by the type of charity. Region explains almost nothing. The practical conclusion is to use both: plan timing by sector, and judge size and lumpiness by segment."""),
md("""## Phase 5 · Segment flow correlation

Each segment's daily AUM is rebuilt from units × price. Its daily flow rate is net flow divided by the previous day's AUM. Correlation between segments is compared four ways: **raw vs deseasonalised** (month × part-of-month and weekday effects removed), and **calm vs stress**. Because runs on liquid funds develop within days, it is computed on **daily** data as well as weekly. A week counts as a stress week if it has 3 or more stress days."""),
md("""**Stress response by value.** The client-level stress lift above is a median, so it describes the *typical* client. The table below measures each segment's total redemptions as a share of its AUM on stress days versus calm days, and versus calm days in the **same calendar month** (so a stress period that happens to fall in March is not mistaken for stress behaviour)."""),
code("""display(o['value_lift'].round(3))"""),
code("""display(o['corr_summary'].round(3))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
for a_, key in zip(ax, [('daily','raw','calm'), ('daily','raw','stress'), ('daily','deseasonalised','calm'), ('daily','deseasonalised','stress')]):
    cm = o['corr'][key]; im = a_.imshow(cm, vmin=-1, vmax=1, cmap='RdBu'); a_.set_title(' · '.join(key), fontsize=9)
    a_.set_xticks(range(len(cm)), [c[:12] for c in cm.columns], rotation=90, fontsize=7); a_.set_yticks(range(len(cm)), [c[:14] for c in cm.index], fontsize=7)
fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(FIG/'11_segment_correlation.png', dpi=110, bbox_inches='tight'); plt.show()"""),
md("""## Phase 6 · Concentration

Holdings at each year-end (and at the latest date) are rebuilt as units × price. The table shows the share held by the top 1, 5, 10, 20, 50 and 100 clients, and how many of each top-N group are rare traders."""),
code("""c = o['concentration']; display(c[['date','holders']+[f'top{n}_share' for n in [1,5,10,20,50,100]]].assign(**{f'top{n}_share': c[f'top{n}_share'].map(pct) for n in [1,5,10,20,50,100]}).tail(8))
display(c[['date']+[f'top{n}_rare_traders' for n in [1,5,10,20,50,100]]].tail(3))
t = o['top']; display(t.head(20).assign(holding_gbp=t.holding_gbp.map(gbp), share=t.share.map(pct)))
display(t.groupby('segment').agg(clients_in_top100=('client_id','size'), holding=('holding_gbp','sum')).assign(holding=lambda d: d.holding.map(gbp)))"""),
md("""## Phase 7 · Stress tests

Five families of scenario are measured against AUM at 30 October 2026:
1. **Concentration:** the top N clients redeem in full.
2. **Historical replay:** the worst observed net outflow over 1, 5 and 20 days (as a share of AUM) and the worst inside each named episode, applied to today's AUM.
3. **Segment tails:** each segment's 99th-percentile worst outflow rate, either assuming all segments hit together (ρ = 1) or combined with the historical *stress-period* correlation matrix.
4. **Combined:** a 5-day tail plus the top-10 exit plus known payables (declared distributions and redemptions awaiting payment).
5. **Reverse stress test:** how many of the largest clients could leave before a liquidity buffer is exhausted.

**What is tested against.** This database has no portfolio holdings or maturities, so daily and weekly liquid assets cannot be measured. Buffers of 5/10/20% of AUM are **illustrative**. The money-market regulatory floors are shown for reference only, because this fund is not an MMF. UK MMFR currently requires 10% daily / 30% weekly liquid assets for LVNAV/CNAV funds and 7.5% / 15% for short-term VNAV funds. US Rule 2a-7 (2023) requires 25% / 50%. In June 2026 the FCA said it would keep the current UK minimums rather than raise them as proposed in CP23/28; further guidance is pending. **Second round:** once a 10% liquid buffer is used up, the rest must come from asset sales. The loss at illustrative fire-sale haircuts of 1/3/5% is shown."""),
code("""T = o['tests']; display(T.assign(outflow_gbp=T.outflow_gbp.map(gbp), outflow_share=T.outflow_share.map(pct), payables_included_gbp=T.payables_included_gbp.map(gbp)))
display(o['tails'].assign(tail_rate=lambda d: d.tail_rate.map(pct), segment_aum_gbp=lambda d: d.segment_aum_gbp.map(gbp), tail_outflow_gbp=lambda d: d.tail_outflow_gbp.map(gbp)))
display(o['reverse'].assign(buffer_gbp=lambda d: d.buffer_gbp.map(gbp), after_payables_gbp=lambda d: d.after_payables_gbp.map(gbp)))
sr = o['second_round']; display(sr[sr.shortfall_gbp > 0].assign(shortfall_gbp=lambda d: d.shortfall_gbp.map(gbp), sale_needed_gbp=lambda d: d.sale_needed_gbp.map(gbp), loss_gbp=lambda d: d.loss_gbp.map(gbp)))
print('Base:', o['base'])"""),
md("""## Phase 8 · Findings

Outputs are this executed notebook, its HTML export (`outputs/11_segments_correlation_stress.html`), the client-to-segment file (`outputs/tables/11_client_segments.csv`), all tables (`outputs/tables/11_*.csv`) and the dashboard page *Segments & stress*. The findings below are generated from the results, so they update when the configuration changes."""),
code("""p = o['profiles'].set_index('segment'); c = o['concentration'].iloc[-1]; T = o['tests'].set_index('scenario'); cs = o['corr_summary'].set_index(['frequency','basis','regime']).mean_pairwise_r
vl = o['value_lift']; rare = p.loc['Rare traders']; biggest = p.drop('Rare traders').aum_share.idxmax(); lift = p.stress_lift
rev = o['reverse'].set_index('buffer').clients_before_exhausted
w5 = T[(T.family=='Historical replay') & T.index.str.contains('5-day')].outflow_gbp.max()
txt = f'''
1. **Data reconcile.** Client flows equal fund flows to the penny on trade and cash dates, and units × price rebuild fund AUM every day. No cancelled, duplicate, multi-currency or omnibus records were found. Internal moves cannot be identified without a flow-type code.
2. **Fund type.** {o['ctx']['fund_type']}: annualised price volatility {pct(o['ctx']['annualised_price_vol'])}, maximum drawdown {pct(o['ctx']['max_drawdown'])}. Stress is defined from the price plus named episodes ({o['ctx']['stress_days']:,} stress days).
3. **Segments.** k = {int(o['selection'].query('chosen').k.iloc[0])} behavioural segments plus rare traders. Silhouette is low ({o['selection'].query('chosen').silhouette.iloc[0]:.2f}): client behaviour is a continuum rather than sharply separated groups. Assignments are fairly stable (median bootstrap ARI {np.median(o['ari']):.2f}). The largest segment by AUM is **{biggest}** ({pct(p.loc[biggest,'aum_share'])}). Segments relate moderately to sector and client type (Cramér's V {o['validation'].set_index('attribute').cramers_v.iloc[0]:.2f} / {o['validation'].set_index('attribute').cramers_v.iloc[1]:.2f}) and not to country, so they partly recover sector without being a relabelling of it.
4. **Rare traders** ({int(rare.clients):,} clients) hold **{pct(rare.aum_share)}** of AUM, matching the 27% found last time. They trade about once every three years, so their history says little about how they would redeem, but **{int(c.top10_rare_traders)} of the top 10** and **{int(c.top100_rare_traders)} of the top 100** holders are rare traders.
5. **Stress barely changes the typical client, but it changes large redemptions.** The median client's stress lift is close to 1 in every segment ({lift.min():.2f}–{lift.max():.2f}): most clients redeem at their usual rate. Measured **by value** and compared with calm days in the same calendar month, {', '.join(f"{r.segment} {r.month_matched_value_lift:.2f}×" for r in vl.sort_values('month_matched_value_lift', ascending=False).head(3).itertuples())} redeemed more on stress days. These are driven by fewer, larger redemptions, so they are less certain, but rare traders alone hold {pct(rare.aum_share)} of AUM. The calendar (March and December year-ends, month-ends) remains the main driver of outflows.
6. **Segment flows are weakly correlated.** The mean pairwise correlation of daily flow rates is {cs[('daily','raw','all')]:.2f} raw and {cs[('daily','deseasonalised','all')]:.2f} after removing calendar effects. In stress it is {cs[('daily','raw','stress')]:.2f} raw and {cs[('daily','deseasonalised','stress')]:.2f} deseasonalised, so it does not rise materially at daily frequency. Weekly figures are higher ({cs[('weekly','raw','all')]:.2f} raw; {cs[('weekly','raw','stress')]:.2f} in stress weeks) because shared calendar timing accumulates within a week. Most of the co-movement is shared seasonality.
7. **Concentration is the dominant liquidity risk.** The top 10 clients hold {pct(c.top10_share)} and the top 100 hold {pct(c.top100_share)}, broadly stable since 2022. A full exit by the top 10 ({gbp(T.loc['Top 10 clients redeem in full','outflow_gbp'])}) is about {T.loc['Top 10 clients redeem in full','outflow_gbp']/w5:.0f}× the worst 5-day net outflow ever observed ({gbp(w5)} at today's AUM).
8. **Reverse stress.** A 10% liquid buffer, after £{o['base']['known_payables']/1e6:.1f}m of known payables, covers the full exit of only the **{rev['10% of AUM']} largest clients**; 20% covers {rev['20% of AUM']}. Historical and segment-tail scenarios (≤ {pct(max(T.loc[T.family.isin(['Historical replay','Segment tails']),'outflow_share']))} of AUM) stay well inside a 5% buffer.
'''
display(Markdown(txt))"""),
md("""### Comparison with the first dataset

The brief asks for a comparison with the first dataset. The project folder contains only one database (the charity fund above). Its rare-trader share (27%) matches the figure quoted from the earlier analysis, so this appears to be the **same** dataset. The table below is written so that a second database can be added: set `CONFIG_NEW` to that database's mapping and the same metrics are computed side by side. Until then, the second column shows the figures quoted from last time where they exist."""),
code("""CONFIG_NEW = None   # e.g. {**S.CONFIG, 'name': 'New fund', 'files': {...}, 'flows': {...}}
def headline(r):
    p = r['profiles'].set_index('segment'); c = r['concentration'].iloc[-1]; cs = r['corr_summary'].set_index(['frequency','basis','regime']).mean_pairwise_r
    return {'Fund type': r['ctx']['fund_type'], 'Segments (k)': int(r['selection'].query('chosen').k.iloc[0]), 'Rare-trader AUM share': pct(p.loc['Rare traders','aum_share']),
            'Median stress lift (range across segments)': f"{p.stress_lift.min():.2f}–{p.stress_lift.max():.2f}", 'Mean daily segment correlation (raw / deseasonalised)': f"{cs[('daily','raw','all')]:.2f} / {cs[('daily','deseasonalised','all')]:.2f}",
            'Mean correlation in stress (daily, raw)': f"{cs[('daily','raw','stress')]:.2f}", 'Top-10 share': pct(c.top10_share), 'Top-100 share': pct(c.top100_share),
            'Largest clients covered by a 10% buffer': int(r['reverse'].set_index('buffer').clients_before_exhausted['10% of AUM'])}
cmp = pd.DataFrame({CONFIG['name']: headline(o)})
cmp['New database'] = pd.Series(headline(S.run(CONFIG_NEW, write=True, prefix='11new_'))) if CONFIG_NEW else pd.Series({'Rare-trader AUM share': f"{pct(CONFIG['previous']['rare_trader_aum_share'])} (quoted from last analysis)"})
display(cmp.reindex(list(headline(o))).fillna('—'))"""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
path = ROOT / 'notebooks/11_segments_correlation_stress.ipynb'
NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
nbf.write(nb, path)
html, _ = HTMLExporter().from_notebook_node(nb)
(ROOT / 'outputs/11_segments_correlation_stress.html').write_text(html)
errs = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
print('executed; errors:', len(errs))
