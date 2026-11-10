// Segments & stress page: every tab and control renders finite values; live stress maths agrees with Notebook 11.
const fs=require('fs'),vm=require('vm'),assert=require('assert');const root=__dirname+'/dist/';
const els={};const stub=id=>els[id]||(els[id]={id,value:'',checked:false,innerHTML:'',textContent:'',style:{},classList:{add(){},remove(){}},setAttribute(){},querySelectorAll:()=>[],querySelector:s=>stub(s.replace(/^#/,'')+':'+id)});
const app={innerHTML:'',querySelector:s=>stub(s.replace(/^#/,'')),querySelectorAll:()=>[]};
const ctx={window:{},console,Math,Intl,Date,JSON,setTimeout,atob,Buffer};vm.createContext(ctx);
for(const f of ['charts.js','filters.js','rankings-data.js','segments-data.js','stress-engine.js','segments.js'])vm.runInContext(fs.readFileSync(root+f,'utf8'),ctx);
const W=ctx.window,ST=W.__segState,D=W.SEGMENTS,bad=h=>/NaN|undefined|Infinity|\[object/.test(h);let n=0;
const all=()=>Object.values(els).map(e=>e.innerHTML).join('');
const render=()=>{for(const k in els)els[k].innerHTML='';W.renderSegments(app);const h=all();if(bad(h)){const i=h.search(/NaN|undefined|Infinity|\[object/);console.log(ST.tab,JSON.stringify(h.slice(i-200,i+50)));}assert(!bad(h),ST.tab);n++;return h;};
const want={overview:'How the study works',data:'Missing, and how we derived it',find:'Let k-means form the groups',segments:'The segments in plain words',stress:'Define stress periods',tests:'Reverse stress test',clients:'Find and query clients'};
for(const tab of Object.keys(want)){ST.tab=tab;const h=render();assert(h.includes(want[tab]),tab);}
ST.tab='stress';for(const m of ['red','sub','both','net'])for(const tf of ['daily','weekly'])for(const b of ['raw','deseasonalised'])for(const r of ['all','calm','stress']){ST.ctxMeasure=m;ST.togetherFreq=tf;ST.corr={frequency:tf,basis:b,regime:r};const h=render();assert(h.includes('Answer:'));}
// Live stress engine with the notebook default definition reproduces every notebook stress result.
{const SE=W.StressEngine,def={dd:0.10,periods:Object.entries(D.report.episodes).map(([n,[a,b]])=>({name:n,from:a,to:b,on:true}))},fl=SE.flags(D,def);assert.strictEqual(fl.days,D.report.ctx.stress_days);assert(fl.stress.every((s,i)=>s===D.flags.stress[i]));
 const C=SE.correlations(D,fl);for(const r of D.corr_summary){const c=C[[r.frequency,r.basis,r.regime].join('|')];assert.strictEqual(c.periods,r.periods);for(const k of ['mean_pairwise_r','min_r','max_r'])assert(Math.abs(c[k]-r[k])<1e-5,k);}
 for(const [k,m] of Object.entries(D.corr)){const c=C[k];m.labels.forEach((a,i)=>m.labels.forEach((b,j)=>{const x=m.r[i][j],y=c.r[c.labels.indexOf(a)][c.labels.indexOf(b)];assert((x===null)===(y===null));if(x!==null)assert(Math.abs(x-y)<1e-5);}));}
 for(const r of SE.replay(D,def.periods,D.report.base.aum)){const n=D.tests.find(x=>x.scenario===r.scenario);assert(Math.abs(n.outflow_gbp-r.outflow_gbp)<1);}
 for(const x of SE.tailHist(D,C['daily|raw|stress'])){const n=D.tests.find(t=>t.scenario===x.h+'-day 99th pct, historical stress correlation');assert(Math.abs(n.outflow_gbp-x.gbp)<1);}
 const L=SE.lift(D,fl);for(const v of D.value_lift){assert(Math.abs(L[v.segment].lift-v.value_lift)<1e-6);assert(Math.abs(L[v.segment].month_matched-v.month_matched_value_lift)<1e-6);}
 // Custom definition: price trigger off, COVID only, plus a user period.
 const cv=D.report.episodes['COVID-19 market shock'],cust={dd:0,periods:[{name:'COVID',from:cv[0],to:cv[1],on:true},{name:'Test period',from:'2015-01-05',to:'2015-02-27',on:true},{name:'Off',from:'2010-01-01',to:'2010-12-31',on:false}]},f2=SE.flags(D,cust);
 assert.strictEqual(f2.days,D.daily.dates.filter(t=>(t>=cv[0]&&t<=cv[1])||(t>='2015-01-05'&&t<='2015-02-27')).length);assert.strictEqual(SE.replay(D,cust.periods,1).length,4);
 ST.sp=JSON.parse(JSON.stringify(cust));for(const tab of ['stress','tests']){ST.tab=tab;const h=render();assert(h.includes('custom'),tab);}
 ST.tab='tests';assert(render().includes('Test period: worst 20-day'));ST.tab='stress';assert(render().includes('Test period'));ST.sp=null;}
// Unticking a period updates the stress-day strips (regression: the marker used to stay unchanged).
{ST.tab='stress';ST.sp=null;let h=render();assert(h.includes('Stress days in use · 781'));const sp=ST.sp;sp.periods.find(p=>p.name==='Euro-area debt crisis').on=false;h=render();assert(h.includes('Stress days in use · 716'),'euro off');
 sp.dd=0;h=render();assert(h.includes('From price trigger · 0'));sp.periods.forEach(p=>p.on=false);h=render();assert(h.includes('Stress days in use · 0'));ST.sp=null;}
ST.tab='segments';for(const m of ['aum_share','clients','median_holding','trades_per_year','redemption_share','stress_lift','month_end_share'])for(const a of ['share','gbp']){ST.profileMetric=m;ST.aumMode=a;render();}
ST.tab='find';for(const y of [0.5,1,2])for(const ms of ['',...D.order.slice(0,-1)]){ST.elYears=y;ST.mapSeg=ms;render();}
ST.tab='segments';ST.netSegs.clear();render();assert(all().includes('Select at least one segment'));D.order.forEach(s=>ST.netSegs.add(s));
ST.tab='tests';for(const N of [1,5,10,20,50,100])for(const s of ['',...D.order]){ST.topN=N;ST.topSeg=s;render();}
ST.tab='tests';for(const b of [0.05,0.1,0.2,0.3])for(const p of [true,false])for(const h of [0.01,0.05]){ST.buffer=b;ST.payables=p;ST.haircut=h;const html=render();
 // Live reverse-stress count agrees with Notebook 11 for the illustrative buffers (with payables).
 const ref=D.reverse.find(r=>r.buffer===Math.round(b*100)+'% of AUM');if(ref&&p){const m=all().match(/could pay the <b>(\d+\+?) largest client/);assert(m&&(m[1]===String(ref.clients_before_exhausted)||(m[1]==='100+'&&ref.clients_before_exhausted>=100)),'reverse '+b+' '+(m&&m[1])+' vs '+ref.clients_before_exhausted);}}
ST.fam.add('Concentration');render();ST.fam.clear();
ST.tab='clients';for(const id of [D.top[0].client_id,D.clients.ids[D.clients.segment.indexOf(D.order.length-1)],'C000001']){ST.client=id;const h=render();assert(h.includes('Typical'),id);}
// Query tool: segment filter, conditions, presets, sort; results match a direct count.
{const Q=ST.q,K=D.clients,ci=Object.fromEntries(K.cols.map((c,j)=>[c,j]));Q.segs=new Set(['Rare traders']);Q.conds=[{f:'current_holding_gbp',op:'>=',v:5e6}];let h=render();
 const exp=K.ids.filter((_,i)=>D.order[K.segment[i]]==='Rare traders'&&K.values[ci.current_holding_gbp][i]>=5e6).length;assert(new RegExp('Clients matching</div><div class="metric">'+exp.toLocaleString()).test(h),'query count '+exp);
 Q.segs=new Set();Q.conds=[{f:'stress_lift',op:'>=',v:1.5},{f:'red_december_share',op:'<=',v:0.1}];for(const srt of ['name','stress_lift','current_holding_gbp'])for(const d of [1,-1]){Q.sort=srt;Q.dir=d;assert(!bad(render()));}
 Q.conds=[{f:'current_holding_gbp',op:'>=',v:1e12}];assert(render().includes('No clients match'));Q.conds=[];Q.segs=new Set();}
// Bundle integrity: segment counts add up; concentration share monotone in N.
assert.strictEqual(D.profiles.reduce((s,p)=>s+p.clients,0),D.clients.ids.length);
for(const r of D.concentration){let last=0;for(const k of [1,5,10,20,50,100]){assert(r[`top${k}_share`]>=last);last=r[`top${k}_share`];}}
assert(D.checks.every(c=>c.passed));
console.log(`PASS segments page: ${n} views across 7 tabs and every control; live stress engine reproduces notebook (correlations, replays, tails, value lift) and handles custom periods; reverse stress matches notebook; segment counts and concentration consistent.`);
