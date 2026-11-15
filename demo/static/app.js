const $=id=>document.getElementById(id);
let episode=null, frameIndex=0, timer=null, model=null;
const names={dqn:'DQN · wind aware',dqn_no_wind:'DQN · wind unaware',random:'Random feasible allocation',greedy:'Near-fire heuristic',wind_heuristic:'Wind placement heuristic',no_intervention:'No construction'};
const fmt=(v,n=3)=>Number(v).toFixed(n);
const pct=v=>(100*v).toFixed(1)+'%';
const subset=a=>a.length?'Arms '+a.join(' + '):'Empty allocation';
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';}
async function load(){
 stop();$('error').hidden=true;$('reset').disabled=true;$('reset').textContent='Simulating…';
 try{
  if(window.DEMO_DATA){episode=window.DEMO_DATA.episodes[$('policy').value];}
  else{let response=await fetch('/api/episode?'+new URLSearchParams({policy:$('policy').value,seed:$('seed').value,budget:$('budget').value}));const candidate=await response.json();if(!response.ok)throw Error(candidate.error);episode=candidate;}
  frameIndex=0;if(episode.model)model=episode.model;
  $('timeline').max=episode.frames.length-1;
  $('arm').innerHTML=episode.arms.map(a=>`<option value="${a.arm_id}">Arm ${a.arm_id}</option>`).join('');
  if(model){$('parameters').textContent=`${model.parameter_count.toLocaleString()} learned parameters · γ = ${model.gamma} · checkpoint at ${model.training_step.toLocaleString()} training intervals`;$('architecture').textContent=model.architecture;}
  render();
 }catch(error){$('error').textContent=error.message;$('error').hidden=false;}
 finally{$('reset').disabled=false;$('reset').textContent='Load episode';}
}
function drawGrid(f){
 const c=$('grid'),ctx=c.getContext('2d'),n=episode.config.grid_size,size=c.width/n;
 ctx.clearRect(0,0,c.width,c.height);
 for(let r=0;r<n;r++)for(let col=0;col<n;col++){ctx.fillStyle=`rgb(${f.rgb[r][col].join(',')})`;ctx.fillRect(col*size,r*size,size,size);ctx.strokeStyle='rgba(255,255,255,.08)';ctx.strokeRect(col*size,r*size,size,size);}
 for(const arm of episode.arms){
  const selected=f.allocation.includes(arm.arm_id),focus=$('arm-details').open&&arm.arm_id===Number($('arm').value);
  if(!$('candidates').checked&&!selected&&!focus)continue;
  const [first,last]=[arm.cells[0],arm.cells.at(-1)];
  ctx.beginPath();ctx.moveTo((first[1]+.5)*size,(first[0]+.5)*size);ctx.lineTo((last[1]+.5)*size,(last[0]+.5)*size);
  ctx.strokeStyle=selected?'#ffe176':focus?'#fff':f.eligible[arm.arm_id]?'rgba(255,255,255,.48)':'rgba(20,40,25,.25)';
  ctx.lineWidth=selected?7:focus?4:2;ctx.setLineDash(selected||focus?[]:[5,4]);ctx.stroke();ctx.setLineDash([]);
  if(selected||focus){ctx.fillStyle='#172e22';ctx.fillRect((arm.col+.5)*size-10,(arm.row+.5)*size-28,24,20);ctx.fillStyle=selected?'#ffe176':'#fff';ctx.font='bold 14px system-ui';ctx.fillText(arm.arm_id,(arm.col+.5)*size-6,(arm.row+.5)*size-13);}
 }
}
function render(){
 if(!episode)return;const f=episode.frames[frameIndex];$('timeline').value=frameIndex;
 $('tick').textContent=f.t+' / '+episode.config.max_steps;$('materials').textContent=f.budget+' / '+episode.initial_budget;
 $('built').textContent=f.built.filter(Boolean).length+' / '+episode.arms.length;$('affected').textContent=pct(f.affected);
 $('allocation').textContent=f.finished?'Episode finished':subset(f.allocation);
 $('action-detail').textContent=f.finished?episode.ending:`Action ${f.action} · ${f.allocation.length} active arms · ${f.allocation.length*episode.config.resource_cost} material units`;
 $('feasible').textContent=f.feasible+' / '+episode.action_count;
 $('observed-wind').textContent=f.wind_observation.map(x=>fmt(x,2)).join(', ');
 $('actual-wind').textContent=episode.actual_wind.map(x=>fmt(x,2)).join(', ');
 $('ending').textContent=f.finished?episode.ending:`${f.active_fire} burning cells · asset damage ${pct(f.asset_damage)} · wind stays fixed throughout this episode`;
 $('q-note').textContent=episode.model?'Actual feasible Q values; larger is better. '+(episode.action_count-f.feasible)+' allocation entries are masked.':'This baseline uses a placement rule rather than a learned Q network.';
 $('q-values').innerHTML=f.q_values.length?f.q_values.map(q=>`<tr><td>${subset(q.arms)}</td><td>${fmt(q.q)}</td></tr>`).join(''):'<tr><td colspan="2">'+(f.finished?'No decision at the episode boundary.':'No DQN Q values for this policy.')+'</td></tr>';
 $('checkpoint').textContent=episode.model?episode.model.checkpoint+' · selected at step '+episode.model.training_step:'Observation-only baseline';
 const i=Number($('arm').value),features=f.arm_features[i];
 $('arm-state').innerHTML=[['Construction',f.built[i]?'Built · persistent':'Unbuilt'],['Eligibility',f.eligible[i]?'Available':'Unavailable'],['Local burning',pct(features[3])],['Local fuel',fmt(features[2])],['Orientation',episode.arms[i].vertical?'Vertical':'Horizontal']].map(([a,b])=>`<div class="fact"><span>${a}</span><b>${b}</b></div>`).join('');
 $('reward').textContent='Reward '+fmt(f.previous_reward);$('return').textContent=fmt(f.discounted_return);
 $('transition-text').textContent=frameIndex?`${subset(episode.frames[frameIndex-1].allocation)} executed. Fire advanced once; existing firebreaks remained.`:'Press Next step to run the decision above.';
 $('back').disabled=frameIndex===0;$('step').disabled=f.finished;$('play').disabled=f.finished;
 drawGrid(f);
}
async function results(){
 try{
  const data=window.DEMO_DATA?window.DEMO_DATA.results:await(await fetch('/api/results')).json();
  $('result-status').textContent=data.status;
  const methods=Object.keys(names),cell=(m,metric)=>data.rows.find(r=>r.method===m&&r.metric===metric);
  $('comparison').querySelector('tbody').innerHTML=data.rows.length?methods.map(m=>{const r=cell(m,'discounted_return');return `<tr><td>${names[m]}</td><td>${fmt(r.mean)} [${fmt(r.ci_low)}, ${fmt(r.ci_high)}]</td><td>${pct(cell(m,'affected_fraction').mean)}</td><td>${pct(cell(m,'settlement_survived').mean)}</td></tr>`}).join(''):'<tr><td colspan="4">Evaluation appears after the pilot finishes.</td></tr>';
  const c=$('learning'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!data.learning.length)return;
  const lo=Math.min(...data.learning.map(r=>+r.discounted_return))-.1,hi=Math.max(...data.learning.map(r=>+r.discounted_return))+.1;
  const x=s=>60+(+s/10000)*910,y=v=>270-(+v-lo)/(hi-lo)*235;
  ctx.font='13px system-ui';ctx.fillStyle='#5c7263';ctx.strokeStyle='#dce5dc';
  for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;ctx.beginPath();ctx.moveTo(55,y(v));ctx.lineTo(980,y(v));ctx.stroke();ctx.fillText(fmt(v,2),4,y(v)+4);}
  for(const step of [2500,5000,7500,10000])ctx.fillText(step.toLocaleString(),x(step)-25,300);
  for(const method of ['dqn','dqn_no_wind'])for(const seed of [11,22,33]){const rows=data.learning.filter(r=>r.method===method&&r.seed===seed);ctx.beginPath();rows.forEach((r,i)=>i?ctx.lineTo(x(r.step),y(r.discounted_return)):ctx.moveTo(x(r.step),y(r.discounted_return)));ctx.strokeStyle=method==='dqn'?'#34734d':'#d98543';ctx.globalAlpha=seed===11?1:.55;ctx.lineWidth=2.5;ctx.stroke();}ctx.globalAlpha=1;
 }catch(e){$('result-status').textContent='Could not load results: '+e.message;}
}
$('results').addEventListener('toggle',()=>{if($('results').open)results();});
$('arm-details').addEventListener('toggle',render);
$('reset').onclick=load;$('policy').onchange=load;$('arm').onchange=render;$('candidates').onchange=render;
$('step').onclick=()=>{stop();frameIndex=Math.min(frameIndex+1,episode.frames.length-1);render();};
$('back').onclick=()=>{stop();frameIndex=Math.max(0,frameIndex-1);render();};
$('timeline').oninput=()=>{stop();frameIndex=Number($('timeline').value);render();};
$('play').onclick=()=>{if(timer){stop();return;}$('play').textContent='Pause';timer=setInterval(()=>{frameIndex=Math.min(frameIndex+1,episode.frames.length-1);render();if(frameIndex===episode.frames.length-1)stop();},600);};
if(window.DEMO_DATA){$('mode').textContent='Offline replay · real saved models';$('seed').disabled=true;$('budget').disabled=true;$('scenario-note').textContent='Portable snapshot: world 42, budget 6, all six policies. Run demo/server.py to choose other worlds and budgets.';}
load();results();
