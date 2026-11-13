/* Recomputes every stress-dependent result for a user-defined stress definition, mirroring
 * notebooks/segment_stress.py exactly (checked by check_segments.cjs against the notebook defaults):
 *  - stress days: price ≥ threshold below its trailing 252-day high (min 20 days) OR inside any active period
 *  - segment flow correlation: daily/weekly × raw/deseasonalised × all/calm/stress
 *  - segment redemption rate on stress vs calm days
 *  - worst 5/20-day net outflow inside each period, applied to current AUM
 *  - segment-tail scenario combined with the stress-period correlation matrix */
(function(root){'use strict';
const isNum=x=>x===x&&x!==null&&x!==undefined;
function flags(D,cfg){const d=D.daily,p=d.price,N=p.length,dd=new Array(N).fill(null),out=new Uint8Array(N);
 for(let i=0;i<N;i++){if(i<19)continue;let m=-Infinity;for(let j=Math.max(0,i-251);j<=i;j++)if(p[j]>m)m=p[j];dd[i]=p[i]/m-1;}
 const per=(cfg.periods||[]).filter(x=>x.on&&x.from&&x.to);
 for(let i=0;i<N;i++){const t=d.dates[i];if(cfg.dd&&dd[i]!==null&&dd[i]<=-cfg.dd)out[i]=1;else if(per.some(x=>t>=x.from&&t<=x.to))out[i]=1;}
 return {stress:out,drawdown:dd,days:out.reduce((a,b)=>a+b,0)};}
function quantile(s,q){const pos=(s.length-1)*q,lo=Math.floor(pos),hi=Math.ceil(pos);return s[lo]+(s[hi]-s[lo])*(pos-lo);}
function pearson(a,b,min){let n=0,sa=0,sb=0;for(let i=0;i<a.length;i++){if(!isNum(a[i])||!isNum(b[i]))continue;n++;sa+=a[i];sb+=b[i];}if(n<min)return null;const ma=sa/n,mb=sb/n;let xy=0,xx=0,yy=0;
 for(let i=0;i<a.length;i++){if(!isNum(a[i])||!isNum(b[i]))continue;const x=a[i]-ma,y=b[i]-mb;xy+=x*y;xx+=x*x;yy+=y*y;}return xx>0&&yy>0?xy/Math.sqrt(xx*yy):null;}
const parts=t=>{const y=+t.slice(0,4),m=+t.slice(5,7),d=+t.slice(8,10),w=(new Date(Date.UTC(y,m-1,d)).getUTCDay()+6)%7;return {m,p:Math.min(Math.floor((d-1)/10),2),w,weekEnd:new Date(Date.UTC(y,m-1,d+((4-w+7)%7))).toISOString().slice(0,10)};};
let cacheRates=null;
function rates(D){if(cacheRates&&cacheRates.D===D)return cacheRates;const d=D.daily,segs=Object.keys(d.seg_aum).sort(),N=d.dates.length,R={},S={};
 for(const g of segs){const a=d.seg_aum[g],n=d.seg_net[g],r=new Array(N-1);for(let i=1;i<N;i++){const prev=a[i-1];r[i-1]=prev>1e5?n[i]/prev:null;}
  const s=r.filter(isNum).sort((x,y)=>x-y),lo=quantile(s,.001),hi=quantile(s,.999);for(let i=0;i<r.length;i++)if(isNum(r[i]))r[i]=Math.min(hi,Math.max(lo,r[i]));R[g]=r;}
 const P=d.dates.slice(1).map(parts);
 for(const g of segs){const r=R[g],z=r.map(x=>isNum(x)?x:0),mean=z.reduce((a,b)=>a+b,0)/z.length,gm={},gc={},wm={},wc={};
  z.forEach((v,i)=>{const k=P[i].m+'_'+P[i].p;gm[k]=(gm[k]||0)+v;gc[k]=(gc[k]||0)+1;wm[P[i].w]=(wm[P[i].w]||0)+v;wc[P[i].w]=(wc[P[i].w]||0)+1;});
  S[g]=z.map((v,i)=>isNum(r[i])?v-(gm[P[i].m+'_'+P[i].p]/gc[P[i].m+'_'+P[i].p]+wm[P[i].w]/wc[P[i].w]-mean):null);}
 const weeks=[...new Set(P.map(x=>x.weekEnd))],wi=new Map(weeks.map((w,i)=>[w,i]));
 return cacheRates={D,segs,raw:R,des:S,P,weeks,wkIdx:P.map(x=>wi.get(x.weekEnd))};}
function weekly(arr,wkIdx,W){const s=new Array(W).fill(null);arr.forEach((v,i)=>{if(isNum(v))s[wkIdx[i]]=(s[wkIdx[i]]||0)+v;});return s;}
function correlations(D,fl){const X=rates(D),st=Array.from(fl.stress).slice(1),W=X.weeks.length,sw=new Array(W).fill(0);st.forEach((s,i)=>{sw[X.wkIdx[i]]+=s;});const stW=sw.map(v=>v>=3),out={};
 for(const freq of ['daily','weekly'])for(const [basis,src] of [['raw',X.raw],['deseasonalised',X.des]]){const cols=X.segs.map(g=>freq==='daily'?src[g]:weekly(src[g],X.wkIdx,W)),sflag=freq==='daily'?st.map(Boolean):stW;
  for(const regime of ['all','calm','stress']){const keep=sflag.map(s=>regime==='all'||(regime==='stress'?s:!s)),c=cols.map(a=>a.filter((_,i)=>keep[i])),k=c.length,M=[...Array(k)].map(()=>Array(k).fill(null)),off=[];
   for(let i=0;i<k;i++){M[i][i]=pearson(c[i],c[i],20);for(let j=i+1;j<k;j++){const r=pearson(c[i],c[j],20);M[i][j]=M[j][i]=r;if(r!==null)off.push(r,r);}}
   out[[freq,basis,regime].join('|')]={labels:X.segs,r:M,periods:keep.filter(Boolean).length,mean_pairwise_r:off.length?off.reduce((a,b)=>a+b,0)/off.length:null,min_r:off.length?Math.min(...off):null,max_r:off.length?Math.max(...off):null};}}
 return out;}
function lift(D,fl){const d=D.daily,out={};for(const g of Object.keys(d.seg_aum)){let rs=0,as=0,rc=0,ac=0;const sm={},cm={};
 for(let i=1;i<d.dates.length;i++){const prev=d.seg_aum[g][i-1];if(!(prev>1e5))continue;const m=d.dates[i].slice(5,7),o=fl.stress[i]?sm:cm;o[m]=o[m]||[0,0];o[m][0]+=d.seg_red[g][i];o[m][1]+=prev;if(fl.stress[i]){rs+=d.seg_red[g][i];as+=prev;}else{rc+=d.seg_red[g][i];ac+=prev;}}
 let exp=0;for(const m in sm)if(cm[m]&&cm[m][1])exp+=sm[m][1]*cm[m][0]/cm[m][1];
 out[g]={stress_bp:as?1e4*rs/as:null,calm_bp:ac?1e4*rc/ac:null,lift:as&&ac&&rc?(rs/as)/(rc/ac):null,month_matched:exp?rs/exp:null};}return out;}
function replay(D,periods,aum){const d=D.daily,N=d.dates.length,net=d.fund_sub.map((s,i)=>s-d.fund_red[i]),out=[];
 for(const p of periods.filter(x=>x.on&&x.from&&x.to))for(const h of [5,20]){let worst=null,at=null;let run=0;
  for(let i=0;i<N;i++){run+=net[i];if(i>=h)run-=net[i-h];if(i<h-1)continue;const t=d.dates[i];if(t<p.from||t>p.to)continue;const r=run/d.fund_open[i-h+1];if(worst===null||r<worst){worst=r;at=t;}}
  if(worst!==null)out.push({family:'Stress period',scenario:`${p.name}: worst ${h}-day net outflow`,outflow_share:Math.max(0,-worst),outflow_gbp:Math.max(0,-worst)*aum,ended:at,payables_included_gbp:0});}
 return out;}
function tailHist(D,corrStress){const out=[];for(const h of [1,5]){const t=D.tails.filter(x=>x.horizon_days===h),lab=corrStress.labels,idx=t.map(x=>lab.indexOf(x.segment)),v=t.map(x=>x.tail_outflow_gbp);let s=0;
 for(let i=0;i<v.length;i++)for(let j=0;j<v.length;j++){const c=i===j?1:(corrStress.r[idx[i]][idx[j]]??0);s+=v[i]*c*v[j];}out.push({h,gbp:Math.sqrt(Math.max(0,s))});}return out;}
root.StressEngine={flags,correlations,lift,replay,tailHist,pearson};
})(typeof window!=='undefined'?window:globalThis);
