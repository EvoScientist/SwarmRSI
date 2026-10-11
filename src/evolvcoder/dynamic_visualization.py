from __future__ import annotations

from pathlib import Path


def write_dynamic_research_view(
    path: Path, *, summary_filename: str = "live_summary.json"
) -> None:
    """Write a dependency-free live view of the typed hypothesis-program graph."""
    path.write_text("""<!doctype html>
<html><head><meta charset="utf-8"><title>Evolvcoder Research Graph</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#0d1117;color:#e6edf3;font:13px system-ui}
header{height:50px;padding:0 16px;background:#161b22;display:flex;align-items:center;gap:18px;border-bottom:1px solid #30363d}
#meta{color:#9da7b3}.legend{margin-left:auto;display:flex;gap:13px;color:#aab4c0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
#wrap{overflow:auto;width:100vw;height:calc(100vh - 51px)}svg{min-width:1400px;min-height:900px}
.lane{fill:#121822;stroke:#293241;stroke-width:1}.lane-title{fill:#8b9bb0;font-size:12px}.edge{fill:none;stroke:#657285;stroke-width:1.5}.intervention{stroke:#c084fc;stroke-dasharray:6 4}.diagnostic{stroke:#f97316;stroke-dasharray:3 4}.migration{stroke:#22d3ee;stroke-dasharray:7 4;stroke-width:2}
.node{stroke:#d4dbe5;stroke-width:1.3}.valid{fill:#18794e}.failed{fill:#7f1d1d}.pending{fill:#374151}.selected{stroke:#facc15;stroke-width:3}.pending-migration{opacity:.42}.label{fill:#f8fafc;font-size:11px}.sub{fill:#bbc5d1;font-size:10px}.tooltip{position:fixed;display:none;max-width:520px;background:#202938;border:1px solid #475569;padding:10px;white-space:pre-wrap;pointer-events:none;z-index:3}
.hyp{stroke:#d4dbe5;stroke-width:1.3;fill:#1e3a8a}.hyp-deferred{fill:#334155}.hyp-archived{fill:#1f2733}.tested{stroke:#60a5fa;stroke-dasharray:5 3}.adaptation{stroke:#f472b6;stroke-width:2}.unmet{fill:#f59e0b;font-size:10px}
</style></head><body><header><b>Evolvcoder Dynamic Research Graph</b><span id="meta">waiting</span><div class="legend"><span><i class="dot" style="background:#18794e"></i>valid</span><span><i class="dot" style="background:#7f1d1d"></i>failed</span><span><i class="dot" style="background:#c084fc"></i>intervention</span><span><i class="dot" style="background:#22d3ee"></i>migration</span><span><i class="dot" style="background:#1e3a8a"></i>hypothesis</span><span><i class="dot" style="background:#f472b6"></i>adaptation</span></div></header><div id="wrap"><svg id="view"></svg></div><div id="tip" class="tooltip"></div>
<script>
const NS='http://www.w3.org/2000/svg';let prior='';
function el(n,a){const x=document.createElementNS(NS,n);for(const[k,v]of Object.entries(a))x.setAttribute(k,v);return x}
function text(svg,x,y,value,cls){const t=el('text',{x,y,class:cls});t.textContent=value;svg.appendChild(t)}
function render(summary){const graph=summary.graph||summary;const svg=document.querySelector('#view');svg.innerHTML='';const actions=graph.actions||[];const events=graph.events||[];const obs={};const selected=new Set();for(const e of events){const id=e.payload&&e.payload.action_id;if(e.event_type==='search.action_observed')obs[id]=e.payload;if(e.event_type==='search.action_selected')selected.add(id)}
const frontier=summary.frontier_report||null;const hyp=frontier?Object.entries(frontier.hypotheses||{}):[];const hypHeight=hyp.length?100+Math.ceil(hyp.length/4)*74:0;
const islandNames=Object.keys(graph.islands||{});const lanes=islandNames.length?islandNames:['program_tree'];const laneY={};lanes.forEach((n,i)=>laneY[n]=80+hypHeight+i*200);svg.setAttribute('height',Math.max(900,hypHeight+lanes.length*210));
const hypPos={};
if(hyp.length){svg.appendChild(el('rect',{x:20,y:30,width:1340,height:hypHeight-30,rx:12,class:'lane'}));
 const pressure=Object.values(frontier.pressure||{});const queued=s=>pressure.filter(p=>p.side===s&&p.status==='queued').length;
 text(svg,35,54,'hypothesis frontier \u00b7 '+hyp.length+' recorded \u00b7 unmet H '+queued('H')+' / P '+queued('P')+' \u00b7 episodes H '+((frontier.episode_counts||{}).H||0)+' / P '+((frontier.episode_counts||{}).P||0),'lane-title');
 hyp.forEach(([id,h],i)=>{const x=90+(i%4)*330,y=92+Math.floor(i/4)*74;hypPos[id]=[x,y];
  const cls='hyp '+(h.status==='active'?'':h.status==='deferred'?'hyp-deferred':'hyp-archived');
  const g=el('g',{});g.appendChild(el('rect',{x:x-26,y:y-17,width:52,height:34,rx:8,class:cls}));
  const t=el('text',{x:x+34,y:y-3,class:'label'});t.textContent=(h.family&&h.family.primary_operator||id).slice(0,30);g.appendChild(t);
  const open=(h.requirements||[]).filter(r=>r.status==='open'&&r.next_step).length;
  const m=el('text',{x:x+34,y:y+13,class:open?'unmet':'sub'});
  m.textContent=h.status+' \u00b7 tested '+h.effective_test_count+(h.unknown_test_count?' (+'+h.unknown_test_count+' unknown)':'')+(open?' \u00b7 '+open+' unmet':'');g.appendChild(m);
  g.onmousemove=ev=>{const tip=document.querySelector('#tip');tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px';
   tip.textContent=JSON.stringify({statement:h.statement,family:h.family,origin:h.origin,status:h.status,effective_test_count:h.effective_test_count,unknown_test_count:h.unknown_test_count,requirements:h.requirements,programs:h.program_ids,note:'controller bookkeeping about executed code, not a scientific verdict'},null,2)};
  g.onmouseleave=()=>document.querySelector('#tip').style.display='none';svg.appendChild(g)})}
for(const [i,n] of lanes.entries()){svg.appendChild(el('rect',{x:20,y:laneY[n]-42,width:1340,height:170,rx:12,class:'lane'}));text(svg,35,laneY[n]-18,n,'lane-title')}
const levels={root:0};let changed=true;for(let pass=0;pass<actions.length+2&&changed;pass++){changed=false;for(const e of graph.dependency_edges||[]){if(e.kind!=='code_parent'||levels[e.source_id]===undefined)continue;const v=levels[e.source_id]+1;if(levels[e.target_id]===undefined||levels[e.target_id]<v){levels[e.target_id]=v;changed=true}}}
const byLane={};for(const a of actions){const lane=a.island_id||'program_tree';(byLane[lane]??=[]).push(a)}const pos={root:[70,laneY[lanes[0]]+50]};for(const [lane,items]of Object.entries(byLane)){const rows={};for(const a of items){const level=levels[a.id]??1;const row=rows[level]||0;rows[level]=row+1;pos[a.id]=[100+level*235,laneY[lane]+row*52]}}
for(const e of graph.dependency_edges||[]){const a=pos[e.source_id],b=pos[e.target_id];if(!a||!b)continue;const cls='edge '+(e.kind==='intervention_source'?'intervention':e.kind==='diagnostic_source'?'diagnostic':'');const mid=(a[0]+b[0])/2;svg.appendChild(el('path',{d:`M${a[0]+24},${a[1]} C${mid},${a[1]} ${mid},${b[1]} ${b[0]-25},${b[1]}`,class:cls}))}
for(const e of graph.migration_edges||[]){const a=pos[e.source_id];const consumedTarget=e.target_action_id&&pos[e.target_action_id];const y=laneY[e.target_island];if(!a||(!consumedTarget&&y===undefined))continue;const b=consumedTarget||[95,y-25];const mid=(a[0]+b[0])/2;svg.appendChild(el('path',{d:`M${a[0]},${a[1]+22} C${mid},${a[1]+70} ${mid},${b[1]-35} ${b[0]},${b[1]-23}`,class:'migration '+(e.consumed?'':'pending-migration')}))}
if(hyp.length){const episodes=(frontier.episodes||[]);const adapted=new Set(episodes.filter(e=>e.mode==='gap_adaptation'||e.mode==='activate_hypothesis').map(e=>e.action_id));
 for(const [id,h] of hyp){const a=hypPos[id];if(!a)continue;for(const program of (h.program_ids||[])){const b=pos[program];if(!b)continue;
  svg.appendChild(el('path',{d:`M${a[0]},${a[1]+18} C${a[0]},${a[1]+90} ${b[0]},${b[1]-90} ${b[0]},${b[1]-24}`,class:'edge tested'+(adapted.has(program)?' adaptation':'')}))}}}
text(svg,52,pos.root[1]+4,'root','label');for(const a of actions){const [x,y]=pos[a.id];const o=obs[a.id];const cls='node '+(o?(o.artifact_valid?'valid':'failed'):'pending')+(selected.has(a.id)?' selected':'');const g=el('g',{});g.appendChild(el('circle',{cx:x,cy:y,r:23,class:cls}));const t=el('text',{x:x+31,y:y-3,class:'label'});t.textContent=a.id;g.appendChild(t);const m=el('text',{x:x+31,y:y+13,class:'sub'});m.textContent=o&&o.native_metric!==null?`score ${Number(o.native_metric).toFixed(5)}`:(o?'invalid':'pending');g.appendChild(m);g.onmousemove=ev=>{const tip=document.querySelector('#tip');tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px';const ep=(summary.frontier_report&&(summary.frontier_report.episodes||[]).find(e=>e.action_id===a.id))||null;tip.textContent=JSON.stringify({hypothesis:a.hypothesis_statement,family:a.scientific_family,island:a.island_id,episode:ep&&{arm:ep.arm,mode:ep.mode,status:ep.status,terminal_kind:ep.terminal_kind,reward:ep.reward,incumbent_pre:ep.incumbent_pre},observation:o},null,2)};g.onmouseleave=()=>document.querySelector('#tip').style.display='none';svg.appendChild(g)}
document.querySelector('#meta').textContent=`${summary.version||''} · ${actions.length} actions · ${graph.event_count||0} events · best ${summary.best_native_metric??'pending'}`
const arm=frontier?` · H/P ${(frontier.settled_counts||{}).H||0}/${(frontier.settled_counts||{}).P||0} settled`:'';document.querySelector('#meta').textContent+=arm}
async function load(){try{const r=await fetch('""" + summary_filename + """?t='+Date.now(),{cache:'no-store'});if(r.ok){const x=await r.json();const h=x.graph&&x.graph.stream_sha256;if(h!==prior){prior=h;render(x)}}}catch(e){document.querySelector('#meta').textContent='waiting for live summary'}setTimeout(load,2000)}load();
</script></body></html>""", encoding="utf-8")
