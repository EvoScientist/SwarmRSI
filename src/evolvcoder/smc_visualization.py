from __future__ import annotations

from pathlib import Path


def write_smc_graph_view(path: Path, *, graph_filename: str = "research_graph.json") -> None:
    """Write a dependency-free live SVG view for an SMC research graph."""
    path.write_text("""<!doctype html>
<html><head><meta charset="utf-8"><title>Evolvcoder SMC</title>
<style>
body{margin:0;background:#10141c;color:#e8edf5;font:14px system-ui}header{padding:12px 18px;background:#171d28;display:flex;gap:20px;align-items:center}#meta{color:#a9b5c8}svg{width:100vw;height:calc(100vh - 52px)}.edge{stroke:#637083;stroke-width:1.5}.inspiration{stroke:#bd8cff;stroke-dasharray:5 4}.node{stroke:#dfe7f3;stroke-width:1.2}.accepted{fill:#1f9d78}.rejected{fill:#5d6675}.population{stroke:#ffd166;stroke-width:3}.label{fill:white;font-size:11px;pointer-events:none}.metric{fill:#cbd5e1;font-size:10px;pointer-events:none}
</style></head><body><header><b>SMC Research Graph</b><span id="meta">loading</span><label>stage <input id="stage" type="range" min="0" value="0"></label><span id="stageValue"></span></header><svg id="view"></svg>
<script>
const NS='http://www.w3.org/2000/svg'; let graph=null;
function el(name,attrs){const x=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,v);return x}
function render(){if(!graph)return;const cutoff=+document.querySelector('#stage').value;document.querySelector('#stageValue').textContent=cutoff;const svg=document.querySelector('#view');svg.innerHTML='';const nodes=graph.particles.filter(n=>n.stage<=cutoff);const ids=new Set(nodes.map(n=>n.particle_id));const byStage={};nodes.forEach(n=>(byStage[n.stage]??=[]).push(n));const pos={};for(const [s,items] of Object.entries(byStage)){items.forEach((n,i)=>pos[n.particle_id]=[100+(+s)*240,90+i*105])}
for(const e of graph.dependency_edges){if(!ids.has(e.source)||!ids.has(e.target))continue;const a=pos[e.source],b=pos[e.target];svg.appendChild(el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],class:'edge '+(e.kind==='program_inspiration'?'inspiration':'')}))}
for(const n of nodes){const [x,y]=pos[n.particle_id];const g=el('g',{});const cls='node '+(n.accepted_in_current_population?'accepted population':n.artifact_valid?'accepted':'rejected');g.appendChild(el('circle',{cx:x,cy:y,r:28,class:cls}));const t=el('text',{x:x+36,y:y-4,class:'label'});t.textContent=n.particle_id;g.appendChild(t);const m=el('text',{x:x+36,y:y+13,class:'metric'});m.textContent='score '+(n.primary_metric??'invalid')+' · '+(n.kernel??'initial');g.appendChild(m);svg.appendChild(g)}
document.querySelector('#meta').textContent=`stage ${graph.stage} · λ ${Number(graph.annealing_lambda).toFixed(3)} · ${graph.stop_reason??'running'}`}
async function load(){try{graph=await fetch('""" + graph_filename + """?t='+Date.now()).then(r=>r.json());const slider=document.querySelector('#stage');slider.max=Math.max(0,graph.stage);if(!slider.dataset.touched)slider.value=graph.stage;render()}catch(e){document.querySelector('#meta').textContent='waiting for graph'}setTimeout(load,2000)}
document.querySelector('#stage').oninput=e=>{e.target.dataset.touched='1';render()};load();
</script></body></html>""", encoding="utf-8")
