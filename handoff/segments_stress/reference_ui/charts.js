/* Shared SVG chart kit. Every chart: title, axis titles with units, tidy ticks, zero line,
 * legend (>=2 series), hover/focus tooltips via data-tip, and an empty state.
 * Palette: validated categorical order (dataviz reference palette, light mode). */
(function(){'use strict';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PALETTE=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948'];
const MEASURE={sub:PALETTE[0],red:PALETTE[1],net:PALETTE[2]};
const INK='#0f2a4a',MUTED='#5b6b80',GRID='#e6ebf2',AXIS='#9aa7b8';
/* Diverging blue (+1) – grey (0) – red (−1) for correlations. */
function div(r){if(r===null||r===undefined||!isFinite(r))return '#e9ecf0';const t=Math.max(-1,Math.min(1,r)),mid=[240,239,236],pos=[29,92,171],neg=[196,48,48],e=t>=0?pos:neg,a=Math.abs(t);return 'rgb('+mid.map((m,i)=>Math.round(m+(e[i]-m)*a)).join(',')+')';}
const divInk=r=>r!==null&&r!==undefined&&Math.abs(r)>.55?'#fff':INK;
/* Number formatting: one unit for the whole axis. */
function unitFor(max,kind){if(kind!=='gbp')return {div:1,suf:'',pre:''};return max>=1e6?{div:1e6,suf:'m',pre:'£'}:max>=1e3?{div:1e3,suf:'k',pre:'£'}:{div:1,suf:'',pre:'£'};}
function fmt(v,kind,u){if(v===null||v===undefined||!isFinite(v))return '—';if(kind==='r')return (v<0?'−':'')+Math.abs(v).toFixed(2);if(kind==='pct')return v.toFixed(0)+'%';if(kind==='x')return (v<0?'−':'')+(Math.abs(v)>=10?Math.abs(v).toFixed(0):Math.abs(v).toFixed(1))+'×';if(kind==='num')return Math.round(v).toLocaleString('en-GB');u=u||unitFor(Math.abs(v),kind);const x=Math.abs(v)/u.div,d=u.div===1?0:x>=100?0:x>=10?1:2;return (v<0?'−':'')+u.pre+x.toFixed(d)+u.suf;}
function money(v){if(v===null||v===undefined||!isFinite(v))return '—';return (v<0?'−':'')+'£'+Math.abs(v).toLocaleString('en-GB',{maximumFractionDigits:0});}
function nice(lo,hi,n=5){if(!(hi>lo)){hi=lo+1;}const raw=(hi-lo)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw);const a=Math.floor(lo/step)*step,b=Math.ceil(hi/step)*step,t=[];for(let v=a;v<=b+step*1e-9;v+=step)t.push(Math.abs(v)<step*1e-9?0:v);return {lo:a,hi:b,ticks:t};}
function empty(msg){return '<div class="chart-empty" role="status">'+esc(msg)+'</div>';}
function legend(items,kind='line'){if(items.length<2)return '';return '<div class="legend" role="list">'+items.map(s=>`<span role="listitem"><i class="key-${s.rect?'rect':kind}" style="--c:${s.color}"${s.dash?' data-dash="1"':''}></i>${esc(s.label)}</span>`).join('')+'</div>';}
function frame({W,H,L,R,T,B,title,yTitle,xTitle}){return {W,H,L,R,T,B,open:`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>`,
 axes:(yTitle?`<text class="axis-title" transform="translate(14 ${T+(H-T-B)/2}) rotate(-90)" text-anchor="middle">${esc(yTitle)}</text>`:'')+(xTitle?`<text class="axis-title" x="${L+(W-L-R)/2}" y="${H-4}" text-anchor="middle">${esc(xTitle)}</text>`:'')};}
function axisDigits(ticks,u){u=u||{div:1};if(ticks.length<2)return null;const st=Math.abs(ticks[1]-ticks[0])/u.div,t=String(+st.toPrecision(6));return t.includes('.')?t.split('.')[1].length:0;}
function fmtAxis(v,kind,u,d){if(kind==='r'&&d!==null)return (v<0?'−':'')+Math.abs(v).toFixed(Math.max(2,d));if(kind!=='gbp'||d===null)return fmt(v,kind,u);return (v<0?'−':'')+u.pre+(Math.abs(v)/u.div).toFixed(d)+u.suf;}
function yAxis(f,sc,ticks,kind,u){let s='';const d=axisDigits(ticks,u);for(const v of ticks){const y=sc(v);s+=`<line class="${v===0?'zeroline':'gridline'}" x1="${f.L}" x2="${f.W-f.R}" y1="${y}" y2="${y}"/><text class="tick" x="${f.L-8}" y="${y+4}" text-anchor="end">${esc(fmtAxis(v,kind,u,d))}</text>`;}return s;}
function xTicks(labels,x,f,max=8){const n=labels.length;if(!n)return '';const step=Math.max(1,Math.ceil(n/max)),idx=[];for(let i=0;i<n;i+=step)idx.push(i);if(idx.at(-1)!==n-1){if(idx.length>1&&n-1-idx.at(-1)<Math.max(2,step))idx.pop();idx.push(n-1);}return idx.map(i=>`<text class="tick" x="${x(i)}" y="${f.H-f.B+16}" text-anchor="${i===n-1&&n>2&&x(i)>f.W-f.R-30?'end':'middle'}">${esc(labels[i])}</text>`).join('');}

/* Line chart. series:[{label,color,values,dash,width}] ; band:{lo,hi,label} */
function line(o){const labels=o.labels||[],series=(o.series||[]).filter(s=>s.values&&s.values.some(v=>v!==null&&isFinite(v)));if(!labels.length||!series.length)return empty(o.empty||'No data for this selection.');
 const kind=o.kind||'gbp',vals=series.flatMap(s=>s.values).concat(o.band?[...o.band.lo,...o.band.hi]:[]).filter(v=>v!==null&&isFinite(v));if(o.refs)vals.push(...o.refs.map(r=>r.value));
 let lo=Math.min(o.zero===false?Infinity:0,...vals),hi=Math.max(o.zero===false?-Infinity:0,...vals);if(o.domain){lo=o.domain[0];hi=o.domain[1];}const nt=nice(lo,hi,o.ticks||5),u=unitFor(Math.max(Math.abs(nt.lo),Math.abs(nt.hi)),kind);
 const ST_H=18,stripsH=o.strips?o.strips.length*ST_H+8:0,f=frame({W:o.width||900,H:(o.height||300)+stripsH,L:o.strips?150:76,R:18,T:16+stripsH,B:o.xTitle?52:34,title:o.title||'Line chart',yTitle:o.yTitle,xTitle:o.xTitle}),n=labels.length,x=i=>f.L+(n===1?(f.W-f.L-f.R)/2:i*(f.W-f.L-f.R)/(n-1)),y=v=>f.T+(nt.hi-v)*(f.H-f.T-f.B)/(nt.hi-nt.lo);
 let s=f.open+yAxis(f,y,nt.ticks,kind,u)+f.axes+xTicks(labels,x,f);
 // Runs of true values -> [start,end] index pairs (single days included, min 2px wide).
 const runs=v=>{const r=[];for(let i=0;i<v.length;i++)if(v[i]){let j=i;while(j+1<v.length&&v[j+1])j++;r.push([i,j]);i=j;}return r;},half=(f.W-f.L-f.R)/Math.max(1,n-1)/2,rx=(a,b)=>{const x0=x(a)-half,x1=x(b)+half;return [x0,Math.max(2,x1-x0)];};
 if(o.shade)for(const [a,b] of runs(o.shade.values)){const [x0,w]=rx(a,b);s+=`<rect x="${x0.toFixed(1)}" y="${f.T}" width="${w.toFixed(1)}" height="${f.H-f.T-f.B}" fill="${o.shade.color}" opacity="${o.shade.opacity||.18}"/>`;}
 (o.strips||[]).forEach((st,k)=>{const y0=16+k*ST_H,rr=runs(st.values),days=rr.reduce((a,[i,j])=>a+j-i+1,0);s+=`<rect x="${f.L}" y="${y0}" width="${f.W-f.L-f.R}" height="${ST_H-6}" fill="#f1f4f8" rx="2"/><text class="tick" x="${f.L-8}" y="${y0+9}" text-anchor="end">${esc(st.label)} · ${days.toLocaleString()}</text>`;
  for(const [a,b] of rr){const [x0,w]=rx(a,b);s+=`<rect class="bar" tabindex="0" x="${x0.toFixed(1)}" y="${y0}" width="${w.toFixed(1)}" height="${ST_H-6}" fill="${st.color}" data-tip="${esc(st.label+'\n'+labels[a]+(b>a?' to '+labels[b]:'')+'\n'+(b-a+1)+' day'+(b>a?'s':''))}"/>`;}});
 if(o.band){const pts=o.band.lo.map((v,i)=>v===null?null:x(i)+','+y(v)).filter(Boolean).concat(o.band.hi.map((v,i)=>v===null?null:x(i)+','+y(v)).filter(Boolean).reverse());s+=`<polygon points="${pts.join(' ')}" fill="${o.band.color||'#cde2fb'}" opacity=".7"/>`;}
 for(const r of o.refs||[])s+=`<line x1="${f.L}" x2="${f.W-f.R}" y1="${y(r.value)}" y2="${y(r.value)}" stroke="${MUTED}" stroke-dasharray="4 4"/><text class="tick" x="${f.W-f.R}" y="${y(r.value)-5}" text-anchor="end">${esc(r.label)}</text>`;
 for(const se of series){let d='',pen=false;se.values.forEach((v,i)=>{if(v===null||!isFinite(v)){pen=false;return;}d+=(pen?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1);pen=true;});s+=`<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width||2}" stroke-linejoin="round" stroke-linecap="round"${se.dash?' stroke-dasharray="6 4"':''}/>`;
  if(n<=40&&!o.noDots)se.values.forEach((v,i)=>{if(v!==null&&isFinite(v))s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${se.color}" stroke="#fff" stroke-width="1.5"/>`;});}
 // Hover layer: one hit column per x bucket, crosshair + all-series tooltip.
 const step=Math.max(1,Math.ceil(n/300)),colW=Math.max(2,(f.W-f.L-f.R)/Math.max(1,n-1)*step);
 for(let i=0;i<n;i+=step){const tip=labels[i]+'\n'+series.map(se=>se.label+': '+(o.tipFmt?o.tipFmt(se.values[i]):tipVal(se.values[i],kind))).join('\n')+(o.band?'\n'+(o.band.label||'Range')+': '+tipVal(o.band.lo[i],kind)+' – '+tipVal(o.band.hi[i],kind):'');
  s+=`<g class="hit" tabindex="0" data-tip="${esc(tip)}"><rect x="${x(i)-colW/2}" y="${f.T}" width="${colW}" height="${f.H-f.T-f.B}" fill="transparent"/><line class="xhair" x1="${x(i)}" x2="${x(i)}" y1="${f.T}" y2="${f.H-f.B}"/></g>`;}
 return '<div class="chart-wrap">'+s+'</svg>'+legend(series.map(se=>({label:se.label,color:se.color,dash:se.dash})).concat(o.band&&o.band.label?[{label:o.band.label,color:o.band.color||'#cde2fb',rect:true}]:[]),'line')+'</div>';}
function tipVal(v,kind){return kind==='gbp'||kind==='money'?money(v):fmt(v,kind);}

/* Vertical bars, grouped. series:[{label,color,values}] ; signColors:[pos,neg] for one series. */
function bars(o){const labels=o.labels||[],series=o.series||[];if(!labels.length||!series.length)return empty(o.empty||'No data for this selection.');
 const kind=o.kind||'gbp',vals=series.flatMap(s=>s.values).filter(v=>v!==null&&isFinite(v));const nt=nice(Math.min(0,...vals),Math.max(0,...vals),o.ticks||5),u=unitFor(Math.max(Math.abs(nt.lo),Math.abs(nt.hi)),kind);
 const W0=o.width||900,n=labels.length,longest=Math.max(...labels.map(l=>String(l).length)),rot=longest*6.3>(W0-94)/n,extra=rot?Math.min(100,Math.round(longest*3.5)):0;
 const f=frame({W:W0,H:(o.height||300)+extra,L:76,R:18,T:16,B:(o.xTitle?52:34)+extra,title:o.title||'Bar chart',yTitle:o.yTitle,xTitle:o.xTitle}),slot=(f.W-f.L-f.R)/n,gw=slot*.8,bw=gw/series.length,y=v=>f.T+(nt.hi-v)*(f.H-f.T-f.B)/(nt.hi-nt.lo);
 let s=f.open+yAxis(f,y,nt.ticks,kind,u)+f.axes+(rot?labels.map((l,i)=>`<text class="tick" transform="translate(${f.L+(i+.5)*slot} ${f.H-f.B+12}) rotate(-35)" text-anchor="end">${esc(l)}</text>`).join(''):xTicks(labels,i=>f.L+(i+.5)*slot,f,o.maxTicks||12));
 series.forEach((se,k)=>se.values.forEach((v,i)=>{if(v===null||!isFinite(v))return;const x0=f.L+i*slot+(slot-gw)/2+k*bw,top=Math.min(y(v),y(0)),h=Math.max(1,Math.abs(y(v)-y(0))),c=o.signColors&&series.length===1?(v>=0?o.signColors[0]:o.signColors[1]):se.color;
  s+=`<rect class="bar" tabindex="0" x="${(x0+1).toFixed(1)}" y="${top.toFixed(1)}" width="${Math.max(1,bw-2).toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${c}" data-tip="${esc(labels[i]+'\n'+se.label+': '+tipVal(v,kind))}"/>`;}));
 if(o.notes)s+=o.notes.map((t,i)=>`<text class="tick" x="${f.W-f.R}" y="${i?f.H-f.B-6:f.T+10}" text-anchor="end">${esc(t)}</text>`).join('');
 const leg=o.signColors&&series.length===1?legend([{label:o.signLabels[0],color:o.signColors[0]},{label:o.signLabels[1],color:o.signColors[1]}],'rect'):legend(series,'rect');
 return '<div class="chart-wrap">'+s+'</svg>'+leg+'</div>';}

/* Horizontal ranked bars with names. rows:[{label,sub,value,color}] */
function hbars(o){const rows=o.rows||[];if(!rows.length)return empty(o.empty||'No data for this selection.');const kind=o.kind||'gbp',vals=rows.map(r=>r.value);const nt=nice(Math.min(0,...vals),Math.max(0,...vals),4),u=unitFor(Math.max(Math.abs(nt.lo),Math.abs(nt.hi)),kind);
 const rowH=30,L=o.labelWidth||300,R=24,T=10,B=40,W=o.width||900,H=T+B+rows.length*rowH,x=v=>L+(v-nt.lo)*(W-L-R)/(nt.hi-nt.lo);
 let s=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title||'Ranked bars')}"><title>${esc(o.title||'Ranked bars')}</title>`;
 for(const v of nt.ticks)s+=`<line class="${v===0?'zeroline':'gridline'}" x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H-B}"/><text class="tick" x="${x(v)}" y="${H-B+16}" text-anchor="middle">${esc(fmtAxis(v,kind,u,axisDigits(nt.ticks,u)))}</text>`;
 if(o.xTitle)s+=`<text class="axis-title" x="${L+(W-L-R)/2}" y="${H-4}" text-anchor="middle">${esc(o.xTitle)}</text>`;
 rows.forEach((r,i)=>{const yy=T+i*rowH,a=x(Math.min(0,r.value)),b=x(Math.max(0,r.value));s+=`<text class="rowlabel" x="${L-10}" y="${yy+rowH/2-1}" text-anchor="end">${esc(r.label.length>40?r.label.slice(0,39)+'…':r.label)}</text>${r.sub?`<text class="tick" x="${L-10}" y="${yy+rowH/2+11}" text-anchor="end">${esc(r.sub)}</text>`:''}<rect class="bar" tabindex="0" x="${a}" y="${yy+7}" width="${Math.max(1,b-a)}" height="${rowH-14}" rx="2" fill="${r.color||PALETTE[0]}" data-tip="${esc(r.label+(r.sub?' · '+r.sub:'')+'\n'+tipVal(r.value,kind)+(r.extra?'\n'+r.extra:''))}"/>`;});
 return '<div class="chart-wrap">'+s+'</svg></div>';}

/* Scatter with optional least-squares line. points:[{x,y,tip}] */
function scatter(o){const pts=(o.points||[]).filter(p=>isFinite(p.x)&&isFinite(p.y));if(pts.length<2)return empty(o.empty||'Not enough overlapping periods to plot.');const kx=o.kindX||'gbp',ky=o.kindY||'gbp';
 const nx=nice(Math.min(...pts.map(p=>p.x)),Math.max(...pts.map(p=>p.x)),5),ny=nice(Math.min(...pts.map(p=>p.y)),Math.max(...pts.map(p=>p.y)),5),ux=unitFor(Math.max(Math.abs(nx.lo),Math.abs(nx.hi)),kx),uy=unitFor(Math.max(Math.abs(ny.lo),Math.abs(ny.hi)),ky);
 const f=frame({W:o.width||560,H:o.height||380,L:76,R:18,T:16,B:52,title:o.title||'Scatter',yTitle:o.yTitle,xTitle:o.xTitle}),x=v=>f.L+(v-nx.lo)*(f.W-f.L-f.R)/(nx.hi-nx.lo),y=v=>f.T+(ny.hi-v)*(f.H-f.T-f.B)/(ny.hi-ny.lo);
 let s=f.open+yAxis(f,y,ny.ticks,ky,uy)+f.axes;for(const v of nx.ticks)s+=`<line class="${v===0?'zeroline':'gridline'}" x1="${x(v)}" x2="${x(v)}" y1="${f.T}" y2="${f.H-f.B}"/><text class="tick" x="${x(v)}" y="${f.H-f.B+16}" text-anchor="${v===nx.ticks.at(-1)?'end':'middle'}">${esc(fmtAxis(v,kx,ux,axisDigits(nx.ticks,ux)))}</text>`;
 const dense=pts.length>600;for(const p of pts)s+=`<circle class="dot" cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="${dense?2.5:4}" fill="${o.color||PALETTE[0]}" fill-opacity="${dense?.35:.65}" stroke="#fff" stroke-width="${dense?0:1}"${dense?'':` tabindex="0" data-tip="${esc(p.tip||'')}"`}/>`;
 if(o.fit!==false){const n=pts.length,mx=pts.reduce((a,p)=>a+p.x,0)/n,my=pts.reduce((a,p)=>a+p.y,0)/n;let sxy=0,sxx=0;for(const p of pts){sxy+=(p.x-mx)*(p.y-my);sxx+=(p.x-mx)**2;}if(sxx>0){const b=sxy/sxx,a=my-b*mx,x1=nx.lo,x2=nx.hi;const c=v=>Math.max(ny.lo,Math.min(ny.hi,v));s+=`<line x1="${x(x1)}" y1="${y(c(a+b*x1))}" x2="${x(x2)}" y2="${y(c(a+b*x2))}" stroke="${INK}" stroke-width="2" stroke-dasharray="6 4"/>`;}}
 return '<div class="chart-wrap">'+s+'</svg>'+(o.fit!==false?'<div class="legend"><span><i class="key-dot" style="--c:'+(o.color||PALETTE[0])+'"></i>One period</span><span><i class="key-line" data-dash="1" style="--c:'+INK+'"></i>Least-squares fit</span></div>':'')+'</div>';}

/* Histogram of correlations on a shared [-1,1] axis. dists:[{label,color,values}] ; marker:{value,label} */
function histogram(o){const dists=(o.dists||[]).filter(d=>d.values.length);if(!dists.length)return empty(o.empty||'No defined correlations to show.');const lo=o.domain?o.domain[0]:-1,hi=o.domain?o.domain[1]:1,nb=o.bins||40,w=(hi-lo)/nb;
 const counts=dists.map(d=>{const c=new Array(nb).fill(0);for(const v of d.values){const k=Math.min(nb-1,Math.max(0,Math.floor((v-lo)/w)));c[k]++;}return c.map(x=>100*x/d.values.length);});
 const top=Math.max(...counts.flat()),nt=nice(0,top,4),f=frame({W:o.width||900,H:o.height||260,L:64,R:18,T:16,B:52,title:o.title||'Distribution',yTitle:o.yTitle||'Share of pairs (%)',xTitle:o.xTitle||'Correlation coefficient (r)'}),x=v=>f.L+(v-lo)*(f.W-f.L-f.R)/(hi-lo),y=v=>f.T+(nt.hi-v)*(f.H-f.T-f.B)/(nt.hi-nt.lo);
 let s=f.open+yAxis(f,y,nt.ticks,'pct')+f.axes;for(let v=lo;v<=hi+1e-9;v+=(hi-lo)/8)s+=`<text class="tick" x="${x(v)}" y="${f.H-f.B+16}" text-anchor="middle">${esc(fmt(v,'r'))}</text>`;
 const bw=(f.W-f.L-f.R)/nb/dists.length;
 counts.forEach((c,k)=>c.forEach((p,i)=>{if(!p)return;const x0=x(lo+i*w)+k*bw;s+=`<rect class="bar" tabindex="0" x="${(x0+.5).toFixed(1)}" y="${y(p).toFixed(1)}" width="${Math.max(1,bw-1).toFixed(1)}" height="${(y(0)-y(p)).toFixed(1)}" rx="1.5" fill="${dists[k].color}" data-tip="${esc(dists[k].label+'\nr '+fmt(lo+i*w,'r')+' to '+fmt(lo+(i+1)*w,'r')+'\n'+p.toFixed(1)+'% of '+dists[k].values.length+' pairs')}"/>`;}));
 s+=`<line class="zeroline" x1="${x(0)}" x2="${x(0)}" y1="${f.T}" y2="${f.H-f.B}"/>`;
 for(const m of [].concat(o.marker||[]))s+=`<line x1="${x(m.value)}" x2="${x(m.value)}" y1="${f.T}" y2="${f.H-f.B}" stroke="${INK}" stroke-width="2.5"/><text class="marker-label" x="${Math.min(f.W-f.R-4,Math.max(f.L+4,x(m.value)))}" y="${f.T+10}" text-anchor="${x(m.value)>f.W*.75?'end':'start'}" dx="${x(m.value)>f.W*.75?-6:6}">${esc(m.label)}</text>`;
 return '<div class="chart-wrap">'+s+'</svg>'+legend(dists.map(d=>({label:d.label+' · '+d.values.length.toLocaleString()+' pairs',color:d.color})),'rect')+'</div>';}

/* Correlation heatmap with names and an optional category colour strip.
 * rows:[{label,sub,strip}] ; M[i][j]: r or null ; N overlap ; cell data-i/data-j for click. */
function heatmap(o){const rows=o.rows||[],k=rows.length;if(!k)return empty(o.empty||'Add clients to build the heatmap.');const cell=Math.max(20,Math.min(40,Math.floor(620/k))),L=o.labelWidth||280,T=o.topHeight||190,W=L+k*cell+150,H=T+k*cell+12,showVal=cell>=30&&!o.overlapMode;
 let s=`<svg class="chart heat" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title||'Correlation heatmap')}" style="max-width:${W}px"><title>${esc(o.title||'Correlation heatmap')}</title>`;
 const trunc=(t,m)=>t.length>m?t.slice(0,m-1)+'…':t;
 rows.forEach((r,i)=>{const y=T+i*cell;s+=`<text class="rowlabel" x="${L-14}" y="${y+cell/2+4}" text-anchor="end">${esc(trunc(r.label,36))}</text>`;if(r.strip)s+=`<rect x="${L-9}" y="${y+1}" width="6" height="${cell-2}" fill="${r.strip}"><title>${esc(r.sub||'')}</title></rect>`;
  const x=L+i*cell;s+=`<text class="rowlabel" transform="translate(${x+cell/2+4} ${T-14}) rotate(-55)" text-anchor="start">${esc(trunc(r.label,26))}</text>`;if(r.strip)s+=`<rect x="${x+1}" y="${T-9}" width="${cell-2}" height="6" fill="${r.strip}"/>`;});
 const maxN=Math.max(1,...o.N.flat());
 for(let i=0;i<k;i++)for(let j=0;j<k;j++){const v=o.M[i][j],n=o.N[i][j],x=L+j*cell,y=T+i*cell,fill=o.overlapMode?`rgba(42,120,214,${(.08+.85*n/maxN).toFixed(3)})`:div(i===j?null:v),tip=rows[i].label+(rows[i].sub?' ('+rows[i].sub+')':'')+'\n× '+rows[j].label+(rows[j].sub?' ('+rows[j].sub+')':'')+'\n'+(i===j?'Same client':'r = '+(v===null?'not available (too little overlap or no variation)':fmt(v,'r')))+'\nOverlap: '+n+' '+(o.periodWord||'periods');
  s+=`<rect class="cell" tabindex="0" x="${x+1}" y="${y+1}" width="${cell-2}" height="${cell-2}" rx="2" fill="${i===j?'#f4f6f9':fill}" data-i="${i}" data-j="${j}" data-tip="${esc(tip)}"/>`;
  if(showVal&&i!==j)s+=`<text class="cellval" x="${x+cell/2}" y="${y+cell/2+3.5}" text-anchor="middle" fill="${divInk(v)}">${v===null?'–':v.toFixed(2).replace('-','−')}</text>`;}
 return '<div class="chart-wrap heat-wrap">'+s+'</svg></div>';}
function divLegend(){const stops=[-1,-.5,0,.5,1];return '<div class="div-legend"><span>−1 opposite</span><i style="background:linear-gradient(90deg,'+stops.map(div).join(',')+')"></i><span>+1 move together</span><span class="muted">Grey · not available</span></div>';}

/* Global tooltip: any element with data-tip, on hover or keyboard focus. */
let tipEl=null;function tip(){if(!tipEl&&typeof document!=='undefined'&&document.body){tipEl=document.createElement('div');tipEl.className='viz-tip';tipEl.setAttribute('role','tooltip');document.body.appendChild(tipEl);}return tipEl;}
function show(t,x,y){const el=tip();if(!el)return;el.textContent='';const lines=t.split('\n');lines.forEach((l,i)=>{const d=document.createElement('div');d.textContent=l;if(i===0)d.className='tip-head';el.appendChild(d);});el.style.display='block';const r=el.getBoundingClientRect(),vw=window.innerWidth;el.style.left=Math.min(vw-r.width-8,x+14)+'px';el.style.top=Math.max(8,y-r.height-12)+'px';}
function hide(){if(tipEl)tipEl.style.display='none';}
if(typeof document!=='undefined'&&document.addEventListener){document.addEventListener('pointermove',e=>{const t=e.target&&e.target.closest&&e.target.closest('[data-tip]');if(t)show(t.getAttribute('data-tip'),e.clientX,e.clientY);else hide();});
 document.addEventListener('focusin',e=>{const t=e.target&&e.target.closest&&e.target.closest('[data-tip]');if(t){const r=t.getBoundingClientRect();show(t.getAttribute('data-tip'),r.left+r.width/2,r.top);}else hide();});document.addEventListener('scroll',hide,true);}

window.Charts={line,bars,hbars,scatter,histogram,heatmap,divLegend,div,empty,fmt,money,nice,esc,PALETTE,MEASURE,INK,MUTED};
})();
