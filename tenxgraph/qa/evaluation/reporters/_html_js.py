"""JavaScript for the HTML evaluation report.

Reads the report payload from the ``application/json`` script element
(``#eval-data``) and renders the verdict, KPIs, scorecard matrix, charts and
the detail drawer in the browser. Data reaches ``innerHTML`` only through
``esc()``; free text is never concatenated unescaped. No CDN, no network.
"""

JS_CONTENT = r"""
"use strict";
const DATA=JSON.parse(document.getElementById('eval-data').textContent);
const CUR=DATA.cur, PREV=DATA.prev;
const GATE=DATA.gate.threshold, GATE_DEFAULT=DATA.gate.source!=='threshold', NEAR=0.10;
const $=s=>document.querySelector(s);
// Every data value that reaches innerHTML or an attribute goes through esc().
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const arr=x=>Array.isArray(x)?x:[];
const str=x=>typeof x==='string'?x:'';
const gp=t=>+(t*100).toFixed(1)+'%';
const pct=(x,d=1)=>(x*100).toFixed(d)+'%';
const f2=x=>x==null?'n/a':(+x).toFixed(2);
const nf=n=>Number(n||0).toLocaleString('en-US');
const dur=s=>s>=60?Math.floor(s/60)+'m '+(s%60).toFixed(0)+'s':s.toFixed(1)+'s';
const isSim=r=>r.criterion_results.some(c=>c.criterion==='simulation_goals');
const ORDER=['simulation_goals','tool_name_match_score','tool_trajectory_avg_score','node_order_score','rouge_match'];
const LABEL={simulation_goals:'Simulation goals',tool_name_match_score:'Tool names',tool_trajectory_avg_score:'Tool trajectory',node_order_score:'Node order',rouge_match:'ROUGE match'};
const lab=c=>LABEL[c]||c;
const R=CUR.results, S=CUR.summary;
const PR=PREV?Object.fromEntries(PREV.results.map(r=>[r.eval_id,r])):null;
const crit=(r,c)=>r.criterion_results.find(x=>x.criterion===c);
const critsAll=[...new Set([...ORDER.filter(c=>R.some(r=>crit(r,c))),...R.flatMap(r=>r.criterion_results.map(c=>c.criterion))])];
const caseScore=r=>r.criterion_results.length?r.criterion_results.reduce((a,c)=>a+c.score,0)/r.criterion_results.length:null;
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const errored=r=>!!r.error||r.criterion_results.some(c=>c.error);
const status=r=>errored(r)?'err':(r.passed?'pass':'fail');
function cellState(c){ if(!c)return'na'; if(c.error)return'fail'; if(!c.passed)return'fail'; return (c.score-c.threshold<NEAR)?'near':'pass'; }
function regOf(r){
  if(!PR)return null; const p=PR[r.eval_id]; if(!p)return{k:'new',t:'New case'};
  if(p.passed&&!r.passed)return{k:'newfail',t:'Newly failing'};
  if(!p.passed&&r.passed)return{k:'newpass',t:'Newly passing'};
  if(!p.passed&&!r.passed)return{k:'still',t:'Still failing'};
  return{k:'same',t:'Unchanged'};
}
function chg(r,c){ // change of one cell vs previous
  if(!PR)return null; const p=PR[r.eval_id]; const cur=crit(r,c); if(!cur)return null;
  if(!p)return{k:'nw'}; const pc=crit(p,c); if(!pc)return{k:'nw'};
  const d=cur.score-pc.score; if(Math.abs(d)<0.005&&pc.passed===cur.passed)return null;
  return{k:d>0?'up':d<0?'dn':'flip',d,prev:pc.score,flip:pc.passed!==cur.passed};
}
const prevAgg=PREV?{pr:PREV.summary.pass_rate,sc:PREV.avg_score,lat:PREV.summary.avg_duration_seconds,tok:PREV.summary.total_token_usage.total_tokens,tot:PREV.summary.total_duration_seconds}:null;
const curSc=DATA.avg_score;
// Latency is only compared when no case on either side recorded a 0.0s duration.
const zeroDur=a=>a.some(r=>!(r.duration_seconds>0));
const latOK=!!PR&&!zeroDur(R)&&!zeroDur(PREV.results);

/* header */
function renderTop(){
  $('#meta').innerHTML=`<span><b>${esc(CUR.eval_set_name||CUR.eval_set_id)}</b> <span class="mono faint">${esc(CUR.eval_set_id)}</span></span>
   <span>${esc(new Date(CUR.timestamp*1000).toLocaleString())}</span>
   <span>Duration <b>${esc(dur(S.total_duration_seconds))}</b></span>
   <span>${S.total_cases} cases, ${R.filter(isSim).length} simulation${R.filter(isSim).length===1?'':'s'}</span>`;
  const pass=S.pass_rate>=GATE;
  const gateTxt=GATE_DEFAULT?'all cases must pass':'gate '+gp(GATE);
  const gateX=Math.min(339,Math.max(1,340*GATE));
  const failN=S.failed_cases, errN=S.error_cases;
  $('#verdict').innerHTML=`<div>
    <div class="vlabel">Verdict</div>
    <div class="vmain"><span>${pass?'Gate passed':'Gate failed'}</span><span class="vbadge ${pass?'ok':'bad'}">${pass?'&#10003; PASS':'&#10005; FAIL'}</span></div>
    <div class="vsub">${pct(S.pass_rate)} of cases passed; ${GATE_DEFAULT?'the gate requires every case to pass':'the gate is '+gp(GATE)}. ${S.passed_cases} of ${S.total_cases} cases passed${failN?`, ${failN} failed`:''}${errN?`, ${errN} with errors`:''}.${PR?'':' No previous run to compare with.'}</div></div>
   <div class="gate"><svg viewBox="0 0 340 46" preserveAspectRatio="none" role="img" aria-label="Pass rate ${pct(S.pass_rate)}, ${esc(gateTxt)}">
     <rect x="0" y="20" width="340" height="10" rx="5" fill="var(--line2)"/>
     <rect x="0" y="20" width="${340*S.pass_rate}" height="10" rx="5" fill="${pass?'var(--ok)':'var(--bad)'}"/>
     ${prevAgg?`<line x1="${340*prevAgg.pr}" x2="${340*prevAgg.pr}" y1="14" y2="36" stroke="var(--ink3)" stroke-width="2" stroke-dasharray="2 2"/>`:''}
     <line x1="${gateX}" x2="${gateX}" y1="10" y2="40" stroke="var(--ink)" stroke-width="2"/></svg>
     <div class="gatecap"><span>0%</span><span>${esc(gateTxt)}</span><span>100%</span></div>
     ${prevAgg?`<div class="gatecap"><span>dashed tick: previous run ${pct(prevAgg.pr)}</span></div>`:''}</div>`;
  const T=S.total_token_usage;
  const d=(cur,prev,fmt,lowerBetter,unit)=>{ if(prev==null)return'<span class="flat">no previous run</span>';
    const x=cur-prev; if(Math.abs(x)<1e-9)return'<span class="flat">no change</span>';
    const good=lowerBetter?x<0:x>0; return `<span class="${good?'good':'badc'}">${x>0?'&#9650;':'&#9660;'} ${fmt(Math.abs(x))}${unit||''}</span> <span class="faint">from ${fmt(prev)}${unit||''}</span>`;};
  $('#kpis').innerHTML=`
   <div class="kpi"><div class="kl">Pass rate</div><div class="kv">${pct(S.pass_rate)}</div><div class="kd">${prevAgg?d(S.pass_rate*100,prevAgg.pr*100,x=>x.toFixed(1),false,' pts'):d(0,null)}</div>
     <div class="ks">${S.passed_cases} passed, ${S.failed_cases} failed, ${S.error_cases} errors</div></div>
   <div class="kpi"><div class="kl">Average score</div><div class="kv">${f2(curSc)}</div><div class="kd">${d(curSc,prevAgg&&prevAgg.sc,x=>x.toFixed(2),false)}</div><div class="ks">mean of all criterion scores</div></div>
   <div class="kpi"><div class="kl">Latency per case</div><div class="kv">${dur(S.avg_duration_seconds)}</div><div class="kd">${PR&&!latOK?'<span class="flat">not comparable</span>':d(S.avg_duration_seconds,prevAgg&&prevAgg.lat,x=>x.toFixed(1),true,'s')}</div><div class="ks">total ${dur(S.total_duration_seconds)}${PR&&!latOK?'; some cases recorded 0.0s, so latency is not compared':''}</div></div>
   <div class="kpi"><div class="kl">Tokens</div><div class="kv">${nf(T.total_tokens)}</div><div class="kd">${d(T.total_tokens,prevAgg&&prevAgg.tok,x=>nf(Math.round(x)),true)}</div>
     <div class="ks">${nf(T.input_tokens)} in, ${nf(T.output_tokens)} out, ${nf(T.cache_read_tokens)} cache</div></div>`;
  // regression summary
  if(!PR){ $('#reg').innerHTML='<span class="lead">No previous run found. The baseline is the most recent earlier report of this eval set in the same folder as this file. Run the suite again to see regressions and improvements here.</span>'; return; }
  const g={newfail:[],newpass:[],still:[],new:[]}; R.forEach(r=>{const x=regOf(r); if(g[x.k])g[x.k].push(r)});
  const removed=PREV.results.filter(p=>!R.some(r=>r.eval_id===p.eval_id));
  const grp=(arr,cls,label)=>`<span class="chip ${cls}"><b>${arr.length}</b> ${label}</span>${arr.map(r=>`<button class="link" data-open="${R.indexOf(r)}">${esc(r.eval_id)}</button>`).join(' ')}`;
  $('#reg').innerHTML=`<span class="lead">Compared with the previous report from ${esc(new Date(PREV.timestamp*1000).toLocaleString())}${PREV.filename?` <span class="mono faint">${esc(PREV.filename)}</span>`:''}</span>
   ${grp(g.newfail,g.newfail.length?'k-bad':'k-neu','newly failing')}
   ${grp(g.newpass,g.newpass.length?'k-ok':'k-neu','newly passing')}
   ${grp(g.still,g.still.length?'k-bad':'k-neu','still failing')}
   ${g.new.length?grp(g.new,'k-neu','new cases'):''}
   ${removed.length?`<span class="chip k-neu"><b>${removed.length}</b> removed</span>${removed.map(p=>`<span class="mono faint">${esc(p.eval_id)} (${p.passed?'passed':'failed'} before)</span>`).join(' ')}`:''}`;
}

/* matrix */
let flt='all',query='',sortBy='order',openId=null,view=[];
function getRows(){
  let rows=R.filter(r=>{
    const st=status(r),rg=regOf(r);
    if(flt==='pass'&&st!=='pass')return false; if(flt==='fail'&&st!=='fail')return false; if(flt==='err'&&st!=='err')return false;
    if(flt==='chg'&&!(rg&&rg.k!=='same'))return false;
    if(query){const q=query.toLowerCase(); if(!(r.eval_id+' '+r.name+' '+(r.expected&&r.expected.input||'')).toLowerCase().includes(q))return false;}
    return true;});
  const sv={order:null,status:r=>({err:0,fail:1,pass:2}[status(r)]),score:r=>caseScore(r)??9,latency:r=>-r.duration_seconds,tokens:r=>-r.token_usage.total_tokens,
    change:r=>({newfail:0,still:1,newpass:2,new:3,same:4}[(regOf(r)||{k:'same'}).k])};
  if(sv[sortBy]){const f=sv[sortBy]; rows=rows.map((r,i)=>[r,i]).sort((a,b)=>f(a[0])-f(b[0])||a[1]-b[1]).map(x=>x[0]);}
  return rows;
}
function glyph(st){return {pass:'&#10003;',near:'&#9680;',fail:'&#10005;',na:'&ndash;'}[st]}
const stLabel={pass:'Passed',near:'Passed, close to threshold',fail:'Failed',na:'Not evaluated'};
function renderMatrix(){
  view=getRows();
  const n={all:R.length,pass:0,fail:0,err:0,chg:0}; R.forEach(r=>{n[status(r)]++; const g=regOf(r); if(g&&g.k!=='same')n.chg++;});
  const defs=[['all','All'],['fail','Failed'],['pass','Passed'],['err','Errors'],...(PR?[['chg','Changed']]:[])];
  $('#filters').innerHTML=defs.map(([k,l])=>`<button type="button" data-f="${k}" aria-pressed="${flt===k}">${l}<span class="n">${n[k]}</span></button>`).join('');
  $('#mcap').textContent=`${view.length} of ${R.length} cases, ${critsAll.length} criteria`;
  const thr=c=>{const t=[...new Set(R.map(r=>crit(r,c)).filter(Boolean).map(x=>x.threshold))]; return esc(t.length===1?'min '+t[0]:'min '+t.join(' / '))};
  let h=`<thead><tr><th class="cn">Case</th>${critsAll.map(c=>`<th scope="col">${esc(lab(c))}<span class="thr">${thr(c)}</span></th>`).join('')}<th style="text-align:right;padding-right:14px">Total</th><th>vs previous</th></tr></thead><tbody>`;
  if(!view.length)h+=`<tr><td class="empty" colspan="${critsAll.length+3}">No cases match.</td></tr>`;
  view.forEach(r=>{
    const st=status(r),rg=regOf(r),cs=caseScore(r),np=r.criterion_results.filter(c=>c.passed&&!c.error).length;
    h+=`<tr class="r" tabindex="0" data-i="${R.indexOf(r)}" aria-label="Open ${esc(r.eval_id)}"><td class="cn"><div class="cname"><span class="st ${st}" title="${st==='pass'?'Passed':st==='fail'?'Failed':'Error'}">${st==='pass'?'&#10003;':st==='fail'?'&#10005;':'!'}</span><span class="t">${esc(r.name||r.eval_id)}</span><span class="kind ${isSim(r)?'sim':''}">${isSim(r)?'SIM':'EVAL'}</span></div><div class="cid mono" style="padding-left:26px">${esc(r.eval_id)}</div></td>`;
    critsAll.forEach(c=>{
      const x=crit(r,c),s=cellState(x),ch=chg(r,c);
      const tip=x?`${lab(c)}: ${f2(x.score)} vs threshold ${x.threshold}. ${stLabel[s]}.${ch&&ch.prev!=null?' Previously '+f2(ch.prev)+'.':ch&&ch.k==='nw'?' Not measured in previous run.':''}`:'Not evaluated for this case';
      h+=`<td class="c"><span class="cell ${s} ${ch?'chgd':''}" title="${esc(tip)}" aria-label="${esc(tip)}">${x?`<span class="g">${glyph(s)}</span><span class="s">${f2(x.score)}</span>`:'<span class="g">&ndash;</span>'}${ch?(ch.k==='up'?'<span class="chg up">&#9650;</span>':ch.k==='dn'?'<span class="chg dn">&#9660;</span>':ch.k==='nw'?'<span class="chg nw">NEW</span>':''):''}</span></td>`;
    });
    h+=`<td class="tot"><div class="a">${cs==null?'n/a':f2(cs)}</div><div class="b">${np}/${r.criterion_results.length} criteria</div></td>
     <td class="rg">${rg?`<span class="rgl ${{newfail:'bad',still:'bad',newpass:'ok',new:'neu',same:'flat'}[rg.k]}">${rg.k==='newfail'?'&#9660; ':rg.k==='newpass'?'&#9650; ':''}${rg.t}</span>`:'<span class="faint">no baseline</span>'}</td></tr>`;
  });
  h+='</tbody><tfoot><tr><td class="cn">Pass rate by criterion</td>';
  critsAll.forEach(c=>{const st=S.criterion_stats[c], ps=PREV&&PREV.summary.criterion_stats[c];
    h+=`<td>${st?pct(st.pass_rate,0):'n/a'}<span class="pv">${st?`${esc(st.passed)}/${esc(st.total)}`:''}${ps?` &middot; was ${pct(ps.pass_rate,0)}`:PR?' &middot; new':''}</span></td>`;});
  h+=`<td style="text-align:right;padding-right:14px">${pct(S.pass_rate,0)}<span class="pv">${S.passed_cases}/${S.total_cases} cases</span></td><td></td></tr></tfoot>`;
  $('#matrix').innerHTML=h;
}

/* run over run */
function renderRor(){
  const W=420;
  const keys=[...new Set([...critsAll,...(PREV?Object.keys(PREV.summary.criterion_stats):[])])];
  $('#ror').innerHTML='<div class="ror">'+keys.map(c=>{
    const a=S.criterion_stats[c],b=PREV&&PREV.summary.criterion_stats[c];
    const bar=(v,col,y)=>v==null?'':`<rect x="0" y="${y}" width="${Math.max(2,W*v)}" height="9" rx="2" fill="${col}"/><text x="${Math.max(2,W*v)+5}" y="${y+8}" font-size="10.5" fill="var(--ink2)">${f2(v)}</text>`;
    const svg=`<svg viewBox="0 0 ${W+40} 34" preserveAspectRatio="none" role="img" aria-label="${esc(lab(c))} average score: this run ${a?f2(a.avg_score):'n/a'}, previous ${b?f2(b.avg_score):'n/a'}">
      ${[0,.5,1].map(t=>`<line x1="${W*t}" x2="${W*t}" y1="0" y2="34" stroke="var(--line)" stroke-width="1"/>`).join('')}
      ${bar(b&&b.avg_score,'var(--ink3)',4)}${bar(a&&a.avg_score,'var(--accent)',18)}</svg>`;
    const dl=a&&b?(()=>{const x=a.avg_score-b.avg_score;return `${Math.abs(x)<.005?'score same':'score '+(x>0?'+':'')+x.toFixed(2)}<br>pass ${pct(b.pass_rate,0)} to ${pct(a.pass_rate,0)}`})():a?`pass ${pct(a.pass_rate,0)}<br><span class="faint">new this run</span>`:`<span class="faint">dropped this run</span><br>was pass ${pct(b.pass_rate,0)}`;
    return `<div class="nm">${esc(lab(c))}<small class="mono">${esc(c)}${a?`, n=${esc(a.total)}`:''}</small></div>${svg}<div class="dl">${dl}</div>`;}).join('')+'</div>'
    +(PREV?'':'<p class="faint">No previous run to compare with.</p>');
}

/* strip */
function renderStrip(){
  const mx=Math.max(1e-9,...R.map(r=>r.duration_seconds)), mt=Math.max(1,...R.map(r=>r.token_usage.total_tokens));
  let h='<div class="h">Case</div><div class="h">Duration</div><div class="h">Tokens</div>';
  R.forEach(r=>{const sim=isSim(r), p=PR&&PR[r.eval_id];
    h+=`<div class="nm" title="${esc(r.name)}">${esc(r.eval_id)}</div>
     <div class="bar"><i class="${sim?'sim':''}" style="width:${Math.max(0.5,r.duration_seconds/mx*100*.78)}%"></i>${dur(r.duration_seconds)}${p&&p.duration_seconds>0&&r.duration_seconds>0?` <span class="faint">(was ${dur(p.duration_seconds)})</span>`:''}</div>
     <div class="bar"><i class="${sim?'sim':''}" style="width:${Math.max(0.5,r.token_usage.total_tokens/mt*100*.78)}%"></i>${nf(r.token_usage.total_tokens)}${p?` <span class="faint">(was ${nf(p.token_usage.total_tokens)})</span>`:''}</div>`;});
  $('#strip').innerHTML=h;
}

/* collapsed details */
function renderMore(){
  const rows=R.map(r=>{const t=r.token_usage,a=r.agent_token_usage;return `<tr><td class="mono">${esc(r.eval_id)}</td><td class="num">${nf(t.input_tokens)}</td><td class="num">${nf(t.output_tokens)}</td><td class="num">${nf(t.cache_read_tokens)}</td><td class="num">${nf(t.cache_creation_tokens)}</td><td class="num"><b>${nf(t.total_tokens)}</b></td><td class="num">${nf(a.total_tokens)}</td></tr>`}).join('');
  const T=S.total_token_usage;
  $('#tok').innerHTML=`<table class="t"><thead><tr><th>Case</th><th class="num">Input</th><th class="num">Output</th><th class="num">Cache read</th><th class="num">Cache write</th><th class="num">Total</th><th class="num">Agent only</th></tr></thead><tbody>${rows}</tbody>
   <tfoot><tr><td><b>Run total</b></td><td class="num">${nf(T.input_tokens)}</td><td class="num">${nf(T.output_tokens)}</td><td class="num">${nf(T.cache_read_tokens)}</td><td class="num">${nf(T.cache_creation_tokens)}</td><td class="num"><b>${nf(T.total_tokens)}</b></td><td class="num faint">avg ${nf(Math.round(S.avg_tokens_per_case))} / case</td></tr></tfoot></table>`;
  const cbf=(CUR.metadata&&CUR.metadata.criteria_by_file)||{};
  const cbfE=Object.entries(cbf);
  if(!cbfE.length)$('#cbf').closest('details').hidden=true;
  $('#cbf').innerHTML=cbfE.map(([f,v])=>{v=v||{};const es=Object.entries(v.criteria||{});return `<p style="margin:6px 0"><b class="mono">${esc(f)}</b> <span class="faint">source: ${esc(v.source)}</span></p>${es.length?`<table class="t"><thead><tr><th>Criterion</th><th>Threshold</th><th>Match</th><th>Judge model</th><th>Samples</th></tr></thead><tbody>${es.map(([k,c])=>{c=c||{};return `<tr><td class="mono">${esc(k)}</td><td>${esc(c.threshold)}</td><td>${esc(c.match_type)}</td><td>${esc(c.judge_model)}</td><td>${esc(c.num_samples)}</td></tr>`}).join('')}</tbody></table>`:'<p class="faint" style="margin:0 0 10px">No explicit criteria. Goals come from the simulator scenarios.</p>'}`}).join('');
  const cu=CUR.config_used||{};
  if(!Object.keys(cu).length){$('#cfg').closest('details').hidden=true;return;}
  const crs=cu.criteria||{}, en=Object.entries(crs).filter(([k,v])=>v);
  const scal=Object.entries(cu).filter(([k,v])=>typeof v!=='object'||v===null).map(([k,v])=>`<tr><td class="mono">${esc(k)}</td><td>${esc(JSON.stringify(v))}</td></tr>`).join('');
  $('#cfg').innerHTML=`<table class="t"><tbody>${scal}<tr><td class="mono">user_simulator_config</td><td>${esc(JSON.stringify(cu.user_simulator_config))}</td></tr><tr><td class="mono">enabled criteria</td><td>${en.map(([k,v])=>esc(k)+' (min '+esc(v.threshold)+')').join(', ')||'none'}</td></tr><tr><td class="mono">not configured</td><td class="faint">${Object.entries(crs).filter(([k,v])=>!v).map(x=>esc(x[0])).join(', ')}</td></tr></tbody></table>
   <details class="sub"><summary>Reporter settings</summary><pre>${esc(JSON.stringify(cu.reporter,null,2))}</pre></details>
   <details class="sub"><summary>Full config JSON</summary><pre>${esc(JSON.stringify(cu,null,2))}</pre></details>`;
}

/* drawer */
function parseTranscript(t){
  const out=[]; (t||'').split(/\n(?=(?:USER|ASSISTANT): )/).forEach(b=>{const m=b.match(/^(USER|ASSISTANT): ([\s\S]*)$/); if(m)out.push({role:m[1]==='USER'?'user':'assistant',content:m[2].trim()});});
  return out;
}
const msgText=m=>typeof m.content==='string'?m.content:(m.content==null?'':JSON.stringify(m.content));
const msgHtml=m=>`<div class="msg ${m.role==='user'?'u':'a'}"><small>${m.role==='user'?'User':'Agent'}</small>${esc(msgText(m))||'<span class="faint">(empty)</span>'}</div>`;
function scoreBar(c,pv){
  const W=300, x=v=>W*Math.max(0,Math.min(1,v));
  return `<div class="sb"><svg viewBox="0 0 ${W} 18" preserveAspectRatio="none" role="img" aria-label="Score ${f2(c.score)}, threshold ${esc(c.threshold)}${pv!=null?', previous '+f2(pv):''}">
   <rect x="0" y="5" width="${W}" height="8" rx="4" fill="var(--line2)"/>
   <rect x="0" y="5" width="${Math.max(2,x(c.score))}" height="8" rx="4" fill="${c.passed&&!c.error?'var(--ok)':'var(--bad)'}"/>
   ${pv!=null?`<circle cx="${x(pv)}" cy="9" r="3.2" fill="var(--panel)" stroke="var(--ink3)" stroke-width="1.5"/>`:''}
   <line x1="${x(c.threshold)}" x2="${x(c.threshold)}" y1="1" y2="17" stroke="var(--ink)" stroke-width="2"/></svg></div>`;
}
function kv(o){return `<div class="kvg">${o.map(([k,v])=>`<div><b>${v}</b><span>${k}</span></div>`).join('')}</div>`}
function renderCase(r){
  try{renderCaseUnsafe(r);}catch(err){$('#drb').textContent='This case could not be displayed: '+(err&&err.message||err);}
}
function renderCaseUnsafe(r){
  const sim=isSim(r), st=status(r), rg=regOf(r), p=PR&&PR[r.eval_id], ex=r.expected||{};
  $('#drs').innerHTML=`<span class="pill ${st}">${st==='pass'?'&#10003; Passed':st==='fail'?'&#10005; Failed':'! Error'}</span> <span class="kind ${sim?'sim':''}">${sim?'SIM':'EVAL'}</span> ${rg?`<span class="rgl ${{newfail:'bad',still:'bad',newpass:'ok',new:'neu',same:'flat'}[rg.k]}" style="font-size:12px;margin-left:6px">${rg.t}</span>`:''}`;
  $('#drt').textContent=r.name||r.eval_id; $('#drid').textContent=r.eval_id+'  |  '+dur(r.duration_seconds);
  const tr=sim?parseTranscript(str(r.actual_response)):[];
  const goals=sim&&str(ex.response)?ex.response.split(/;\s*/).filter(Boolean):[];
  const cr=crit(r,'simulation_goals'), ach=new Set([...arr(cr&&cr.details&&cr.details.achieved_goals),...arr(r.metadata&&r.metadata.goals_achieved)]);
  const una=new Set(arr(cr&&cr.details&&cr.details.unachieved_goals));
  let h='';
  if(r.error)h+=`<div class="dsec"><h4>Error</h4><pre style="border-color:var(--bad)">${esc(r.error)}</pre></div>`;
  // I/E/A
  if(sim){
    h+=`<div class="dsec"><h4>Opening input</h4><div class="code">${esc(ex.input||'')}</div></div>
    <div class="dsec"><h4>Goals${r.metadata&&r.metadata.completed!=null?` <span class="faint" style="text-transform:none;letter-spacing:0">| ${esc(r.metadata.turns)} turns, ${r.metadata.completed?'completed':'not completed'}</span>`:''}</h4>
      ${goals.map(g=>`<div class="goal"><span class="gi ${ach.has(g)?'y':una.has(g)?'n':'q'}">${ach.has(g)?'&#10003;':una.has(g)?'&#10005;':'?'}</span><span>${esc(g)}${ach.has(g)||una.has(g)?'':' <span class="faint">(no verdict recorded)</span>'}</span></div>`).join('')||'<span class="faint">No goals recorded.</span>'}</div>
    <div class="dsec"><h4>Conversation (${tr.length} messages)</h4><div class="tr">${tr.map(msgHtml).join('')||'<span class="faint">No transcript.</span>'}</div></div>`;
  } else {
    h+=`<div class="dsec"><h4>Input, expected, actual</h4><div class="trio">
      <div><h5>Input</h5><div class="tx">${esc(ex.input||(arr(r.messages)[0]&&msgText(arr(r.messages)[0]))||'')}</div></div>
      <div><h5>Expected</h5><div class="tx">${esc(ex.response||'')||'<span class="faint">no expected response</span>'}</div></div>
      <div><h5>Actual</h5><div class="tx">${esc(str(r.actual_response).trim())||'<span class="faint">empty</span>'}</div></div></div></div>`;
  }
  // criteria
  h+=`<div class="dsec"><h4>Criteria (${r.criterion_results.length})</h4>`+r.criterion_results.map(c=>{
    const pc=p&&crit(p,c.criterion), s=c.error?'err':c.passed?'pass':'fail', d=c.details||{};
    const reason=str(d.reason||d.reasoning), judged=/no llm provider/i.test(reason);
    return `<div class="crit"><div class="top"><span class="nm">${esc(lab(c.criterion))}</span><span class="pill ${s}">${s==='pass'?'&#10003; pass':s==='fail'?'&#10005; fail':'! error'}</span>
      <span class="mono">${f2(c.score)} <span class="faint">vs min ${esc(c.threshold)}</span></span>
      ${pc?`<span class="tag2">previous ${f2(pc.score)} (${pc.passed?'pass':'fail'})</span>`:PR&&p?'<span class="tag2">not in previous run</span>':''}</div>
      ${scoreBar(c,pc&&pc.score)}
      ${reason?`<div class="reason">${esc(reason)}</div>`:''}
      ${judged?'<div class="note">The judge had no LLM provider, so this score is a fallback, not a real judgement.</div>':''}
      ${c.error?`<pre style="border-color:var(--bad)">${esc(c.error)}</pre>`:''}
      ${d.expected_response!=null?`<details class="sub"><summary>ROUGE detail: precision ${f2(d.precision)}, recall ${f2(d.recall)}, F1 ${f2(d.f1)}</summary><pre>expected: ${esc(d.expected_response)}\nactual:   ${esc(d.actual_response)}</pre></details>`:''}
      <details class="sub"><summary>Raw details and criterion tokens (${nf((c.token_usage||{}).total_tokens)})</summary><pre>${esc(JSON.stringify({details:d,token_usage:c.token_usage},null,2))}</pre></details></div>`;}).join('')+`</div>`;
  // tool calls
  const exTc=arr(ex.tool_calls), acTc=arr(r.actual_tool_calls);
  if(exTc.length||acTc.length)h+=`<div class="dsec"><h4>Tool calls</h4><div class="trio" style="grid-template-columns:1fr 1fr">
    <div><h5>Expected (${exTc.length})</h5>${exTc.map(t=>`<div class="code" style="margin-bottom:6px">${esc(t.name)}(${esc(JSON.stringify(t.args||{}))})</div>`).join('')||'<span class="faint">none</span>'}</div>
    <div><h5>Actual (${acTc.length})</h5>${acTc.map(t=>`<div class="code" style="margin-bottom:6px">${esc(t.name)}(${esc(JSON.stringify(t.args||{}))})${t.result!=null?`\n&rarr; ${esc(typeof t.result==='string'?t.result:JSON.stringify(t.result))}`:''}\n<span class="faint">call_id ${esc(t.call_id||'')}</span></div>`).join('')||'<span class="faint">none</span>'}</div></div></div>`;
  // node order
  const nv=arr(r.node_visits), exNo=arr(ex.node_order);
  if(nv.length||exNo.length)h+=`<div class="dsec"><h4>Node order</h4><div class="muted" style="font-size:13px">Actual: <span class="mono">${nv.map(x=>esc(x)).join(' &rarr; ')||'none'}</span></div>${exNo.length?`<div class="muted" style="font-size:13px">Expected: <span class="mono">${exNo.map(x=>esc(x)).join(' &rarr; ')}</span></div>`:''}</div>`;
  // trajectory
  const tj=arr(r.actual_trajectory);
  if(tj.length)h+=`<div class="dsec"><h4>Trajectory (${tj.length} steps)</h4><ul class="steps">${tj.map((s,i)=>`<li><span class="faint mono">${i+1}</span><span class="k">${esc(s.step_type)}</span><b class="mono">${esc(s.name)}</b>${s.args&&Object.keys(s.args).length?`<span class="mono muted">${esc(JSON.stringify(s.args))}</span>`:''}${s.metadata&&s.metadata.event_type?`<span class="faint">${esc(s.metadata.event_type)}</span>`:''}</li>`).join('')}</ul></div>`;
  // node responses
  const nr=arr(r.node_responses);
  if(nr.length)h+=`<div class="dsec"><details class="sub" style="border:0"><summary><b>Node responses (${nr.length})</b></summary>${nr.map(n=>{const u=n.token_usage||{};return `<div style="padding:8px 0;border-top:1px solid var(--line2)"><div><b class="mono">${esc(n.node_name)}</b> ${n.is_final?'<span class="tag2">final</span>':''} ${n.has_tool_calls?`<span class="tag2">tool calls: ${esc(arr(n.tool_call_names).map(x=>x||'(unnamed)').join(', '))}</span>`:''} <span class="tag2">${nf(u.input_tokens||0)} in, ${nf(u.output_tokens||0)} out</span></div>
      <div class="code" style="margin-top:4px">${esc(n.response_text)||'<span class="faint">(no text)</span>'}</div>
      <details class="sub"><summary>Input messages (${arr(n.input_messages).length})</summary><pre>${esc(JSON.stringify(n.input_messages,null,2))}</pre></details></div>`}).join('')}</details></div>`;
  if(arr(r.node_details).length)h+=`<div class="dsec"><details class="sub" style="border:0"><summary><b>Node details (${r.node_details.length})</b></summary><pre>${esc(JSON.stringify(r.node_details,null,2))}</pre></details></div>`;
  // turns / messages
  if(!sim&&arr(r.messages).length)h+=`<div class="dsec"><details class="sub" style="border:0"><summary><b>Conversation messages (${r.messages.length})</b></summary><div class="tr" style="margin-top:8px">${arr(r.messages).map(msgHtml).join('')}</div></details></div>`;
  if(arr(r.turn_results).length)h+=`<div class="dsec"><details class="sub" style="border:0"><summary><b>Turn results (${r.turn_results.length})</b></summary><pre>${esc(JSON.stringify(r.turn_results,null,2))}</pre></details></div>`;
  // tokens
  const t=r.token_usage;
  h+=`<div class="dsec"><h4>Tokens</h4>${kv([['input',nf(t.input_tokens)],['output',nf(t.output_tokens)],['cache read',nf(t.cache_read_tokens)],['cache write',nf(t.cache_creation_tokens)],['total',nf(t.total_tokens)]])}
   ${p?`<div class="tag2" style="margin-top:6px">Previous run: ${nf(p.token_usage.total_tokens)} total${p.duration_seconds>0?', '+dur(p.duration_seconds):''}</div>`:''}</div>`;
  if(r.metadata&&Object.keys(r.metadata).length)h+=`<div class="dsec"><details class="sub" style="border:0"><summary><b>Metadata</b></summary><pre>${esc(JSON.stringify(r.metadata,null,2))}</pre></details></div>`;
  $('#drb').innerHTML=h; $('#drb').scrollTop=0;
}
let lastFocus=null;
function openCase(id){
  const r=R.find(x=>x.eval_id===id); if(!r)return;
  openId=id; renderCase(r); lastFocus=document.activeElement;
  $('#dr').classList.add('on');$('#scrim').classList.add('on');$('#dr').setAttribute('aria-hidden','false');
  document.body.style.overflow='hidden'; $('#close').focus();
  try{history.replaceState(null,'','#'+encodeURIComponent(id))}catch(e){}
}
function closeCase(){
  $('#dr').classList.remove('on');$('#scrim').classList.remove('on');$('#dr').setAttribute('aria-hidden','true');
  document.body.style.overflow=''; openId=null; try{history.replaceState(null,'',location.pathname)}catch(e){}
  if(lastFocus&&lastFocus.focus)lastFocus.focus();
}
function step(d){const ids=(view.length?view:R).map(r=>r.eval_id); const i=ids.indexOf(openId); const j=(i+d+ids.length)%ids.length; openCase(ids[j]);}

/* wiring */
document.addEventListener('click',e=>{
  const f=e.target.closest('[data-f]'); if(f){flt=f.dataset.f;renderMatrix();return;}
  const o=e.target.closest('[data-open]'); if(o){const t=R[+o.dataset.open]; if(t)openCase(t.eval_id); return;}
  const row=e.target.closest('tr.r'); if(row){const t=R[+row.dataset.i]; if(t)openCase(t.eval_id);}
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&openId)closeCase();
  else if((e.key==='Enter'||e.key===' ')&&e.target.matches&&e.target.matches('tr.r')){e.preventDefault();const t=R[+e.target.dataset.i]; if(t)openCase(t.eval_id);}
  else if(openId&&(e.key==='ArrowDown'||e.key==='j')&&!/INPUT|SELECT/.test(e.target.tagName)){e.preventDefault();step(1);}
  else if(openId&&(e.key==='ArrowUp'||e.key==='k')&&!/INPUT|SELECT/.test(e.target.tagName)){e.preventDefault();step(-1);}
});
$('#scrim').onclick=closeCase;$('#close').onclick=closeCase;$('#next').onclick=()=>step(1);$('#prev').onclick=()=>step(-1);
$('#q').addEventListener('input',e=>{query=e.target.value.trim();renderMatrix();});
$('#sort').addEventListener('change',e=>{sortBy=e.target.value;renderMatrix();});
/* theme */
function setTheme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem('tg-eval-theme',t)}catch(e){}$('#theme').textContent=t==='dark'?'Light theme':'Dark theme';}
(function(){let t=null;try{t=localStorage.getItem('tg-eval-theme')}catch(e){}
  setTheme(t||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));})();
$('#theme').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
renderTop();renderMatrix();renderRor();renderStrip();renderMore();
function hashId(){ // deep link: the fragment is plain text, matched against eval ids only
  if(location.hash.length<2)return null;
  try{return decodeURIComponent(location.hash.slice(1));}catch(e){return null;}
}
const first=hashId(); if(first!=null)openCase(first);
window.addEventListener('hashchange',()=>{const id=hashId(); if(id!=null&&id!==openId)openCase(id); else if(id==null&&openId)closeCase();});
"""
