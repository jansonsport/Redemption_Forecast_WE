/* Shared filter kit: chip groups (OR within a group, AND across groups), background filters,
 * typeahead client picker and date range with presets. Pages own their state objects. */
(function(){'use strict';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const AUM=['Under £10k','£10k–£100k','£100k–£1m','£1m–£10m','£10m and over'],STATUS=['Open','Closed'];
/* meta: {sectors,types,countries,regions,units} lookup arrays. */
const GROUPS=meta=>[['sectors','Sector',meta.sectors,true],['countries','Country',meta.countries,true],['aum','AUM at 30 Oct 2026',AUM,true],['status','Account',STATUS,true],['types','Client type',meta.types,false],['regions','Region',meta.regions,false],['units','Unit type',meta.units,false]];
function newState(){return {sectors:new Set(),countries:new Set(),aum:new Set(),status:new Set(),types:new Set(),regions:new Set(),units:new Set()};}
function chips(key,label,items,state,scope){return `<div class="fgroup" data-scope="${scope}"><b>${esc(label)}</b>${items.map((t,i)=>`<button type="button" class="fchip" data-scope="${scope}" data-group="${key}" data-i="${i}" aria-pressed="${state[key]&&state[key].has(i)?'true':'false'}">${esc(t)}</button>`).join('')}</div>`;}
function background(meta,state,scope,title='Client background',omit=[]){const g=GROUPS(meta).filter(x=>!omit.includes(x[0]));
 return `<div class="fbar" data-scope="${scope}"><h3>${esc(title)}</h3>${g.filter(x=>x[3]).map(x=>`<div class="frow">${chips(x[0],x[1],x[2],state,scope)}</div>`).join('')}
 <details class="fmore"><summary>More background filters: client type, region, unit type</summary>${g.filter(x=>!x[3]).map(x=>`<div class="frow">${chips(x[0],x[1],x[2],state,scope)}</div>`).join('')}</details>
 <div class="frow"><button type="button" class="fclear" data-scope="${scope}">Clear background filters</button><span class="fsummary" data-scope="${scope}" role="status">${esc(summary(meta,state))}</span></div></div>`;}
function summary(meta,state){const names={sectors:meta.sectors,countries:meta.countries,aum:AUM,status:STATUS,types:meta.types,regions:meta.regions,units:meta.units},parts=[];
 for(const [k,label] of [['sectors','sector'],['types','type'],['countries','country'],['regions','region'],['units','unit type'],['aum','AUM'],['status','account']])if(state[k]&&state[k].size)parts.push(label+': '+[...state[k]].map(i=>names[k][i]).join(' or '));
 return parts.length?'Clients matching '+parts.join(' AND '):'All clients (no background filter)';}
/* Bind chip toggles and clear for one scope inside root. */
function bind(root,meta,state,scope,onChange){root.querySelectorAll(`.fchip[data-scope="${scope}"]`).forEach(b=>b.onclick=()=>{const s=state[b.dataset.group],i=Number(b.dataset.i);s.has(i)?s.delete(i):s.add(i);b.setAttribute('aria-pressed',String(s.has(i)));sync(root,meta,state,scope);onChange();});
 const c=root.querySelector&&root.querySelector(`.fclear[data-scope="${scope}"]`);if(c)c.onclick=()=>{Object.values(state).forEach(s=>s.clear&&s.clear());root.querySelectorAll(`.fchip[data-scope="${scope}"]`).forEach(b=>b.setAttribute('aria-pressed','false'));sync(root,meta,state,scope);onChange();};}
function sync(root,meta,state,scope){const s=root.querySelector&&root.querySelector(`.fsummary[data-scope="${scope}"]`);if(s)s.textContent=summary(meta,state);}
const toEngine=state=>Object.fromEntries(Object.entries(state).map(([k,v])=>[k,[...v]]));
/* Rows are RANKINGS.clients-shaped: [id,name,sector,type,country,region,unit,open,close,aum]. */
function matches(row,state){const has=(k,v)=>!state[k]||!state[k].size||state[k].has(v);if(!has('sectors',row[2])||!has('types',row[3])||!has('countries',row[4])||!has('regions',row[5])||!has('units',row[6]))return false;
 if(state.aum&&state.aum.size){const a=row[9],b=a<1e4?0:a<1e5?1:a<1e6?2:a<1e7?3:4;if(!state.aum.has(b))return false;}if(state.status&&state.status.size&&!state.status.has(row[8]?1:0))return false;return true;}
/* Typeahead picker. rows: array of RANKINGS-shaped rows; onPick(index). */
function picker(root,id,meta,rows,onPick,placeholder='Search client name or ID'){const box=root.querySelector('#'+id);if(!box)return;box.classList.add('picker');box.innerHTML=`<input type="search" autocomplete="off" placeholder="${esc(placeholder)}" aria-label="${esc(placeholder)}"><div class="picker-list" role="listbox"></div>`;
 const input=box.querySelector('input'),list=box.querySelector('.picker-list');let hits=[],on=-1;
 const render=()=>{const q=input.value.toLowerCase().trim();hits=[];if(q.length>=2)for(let i=0;i<rows.length&&hits.length<40;i++){const r=rows[i];if((r[0]+' '+r[1]).toLowerCase().includes(q))hits.push(i);}
  list.innerHTML=hits.length?hits.map((i,k)=>{const r=rows[i];return `<button type="button" role="option" data-k="${k}" class="${k===on?'on':''}">${esc(r[1])} <b class="sub">${esc(r[0])}</b><small>${esc(meta.sectors[r[2]])} · ${esc(meta.types[r[3]])} · ${esc(meta.regions[r[5]])} · ${r[8]?'closed '+esc(r[8]):'open'}</small></button>`;}).join(''):(q.length>=2?'<div class="sub" style="padding:10px">No matching clients</div>':'<div class="sub" style="padding:10px">Type at least 2 characters</div>');
  list.classList.add('open');list.querySelectorAll('button').forEach(b=>b.onmousedown=e=>{e.preventDefault();pick(hits[Number(b.dataset.k)]);});};
 const pick=i=>{if(i===undefined)return;input.value='';list.classList.remove('open');onPick(i);};
 input.oninput=()=>{on=-1;render();};input.onfocus=render;input.onblur=()=>setTimeout(()=>list.classList.remove('open'),120);
 input.onkeydown=e=>{if(e.key==='ArrowDown'){on=Math.min(hits.length-1,on+1);render();e.preventDefault();}else if(e.key==='ArrowUp'){on=Math.max(0,on-1);render();e.preventDefault();}else if(e.key==='Enter'){pick(hits[Math.max(0,on)]);e.preventDefault();}else if(e.key==='Escape')list.classList.remove('open');};}
/* Date range with presets. Values are ISO dates. */
function dateRange(prefix,min,max,from,to,presets){return `<div class="frow"><label>From <input type="date" id="${prefix}-from" min="${min}" max="${max}" value="${from}"></label><label>To <input type="date" id="${prefix}-to" min="${min}" max="${max}" value="${to}"></label>${presets.map((p,i)=>`<button type="button" class="fchip" data-preset="${prefix}" data-i="${i}" aria-pressed="false">${esc(p[0])}</button>`).join('')}</div>`;}
function bindDates(root,prefix,presets,onChange){const f=root.querySelector('#'+prefix+'-from'),t=root.querySelector('#'+prefix+'-to'),btns=[...(root.querySelectorAll(`[data-preset="${prefix}"]`)||[])],mark=()=>btns.forEach(b=>{const p=presets[Number(b.dataset.i)];b.setAttribute('aria-pressed',String(p[1]===f.value&&p[2]===t.value));});
 btns.forEach(b=>b.onclick=()=>{const p=presets[Number(b.dataset.i)];f.value=p[1];t.value=p[2];mark();onChange();});f.onchange=t.onchange=()=>{mark();onChange();};mark();}
function download(name,rows){const csv=rows.map(r=>r.map(v=>'"'+String(v===null||v===undefined?'':v).replace(/"/g,'""')+'"').join(',')).join('\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
window.Filters={AUM,STATUS,newState,chips,background,summary,bind,toEngine,matches,picker,dateRange,bindDates,download,esc};
})();
