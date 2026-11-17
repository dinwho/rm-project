'use strict';
const $ = id => document.getElementById(id);
let data, learned = [], baselines = [], left, right, interval = 0, timer = null;
const number = (value, decimals = 0) => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals}) : '—';
const checkpointName = episode => typeof episode.checkpoint === 'string' ? episode.checkpoint : episode.checkpoint?.name || episode.checkpoint?.path || '';
const policyName = episode => episode.label || episode.policy || episode.id;
const learnedKey = episode => checkpointName(episode) || episode.policy || episode.id;
const isLearned = episode => Boolean(checkpointName(episode)) || /ppo|neural|learned/i.test(episode.policy || '');
const frames = episode => episode?.frames || [];
const lastStep = episode => frames(episode).at(-1)?.step ?? 0;
function option(select, value, label) { const node = document.createElement('option'); node.value = value; node.textContent = label; select.append(node); }
function stop() { clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
function start() {
  if (!left || interval >= maximum()) return;
  timer = setInterval(() => { interval = Math.min(interval + 1, maximum()); render(); if (interval >= maximum()) stop(); }, Number($('speed').value));
  $('play').textContent = 'Pause';
}
function maximum() { return Math.max(lastStep(left), lastStep(right)); }
function selectedFrame(episode) { return frames(episode).filter(frame => frame.step <= interval).at(-1) || frames(episode)[0]; }
function cell(action, episode) {
  if (action === null || action === undefined || action < 0) return 'Wait / none';
  const width = frames(episode)[0]?.rgb?.[0]?.length;
  return width ? `(${Math.floor(action / width)}, ${action % width})` : String(action);
}
function fillSeeds() {
  const previous = $('seed').value;
  $('seed').replaceChildren();
  const choices = learned.filter(ep => learnedKey(ep) === $('checkpoint').value);
  [...new Set(choices.map(ep => String(ep.seed)))].sort((a,b) => Number(a)-Number(b)).forEach(seed => option($('seed'), seed, seed));
  if ([...$('seed').options].some(o => o.value === previous)) $('seed').value = previous;
  selectEpisodes();
}
function selectEpisodes() {
  stop(); interval = 0;
  left = learned.find(ep => learnedKey(ep) === $('checkpoint').value && String(ep.seed) === $('seed').value);
  const previous = $('baseline').value;
  $('baseline').replaceChildren();
  const matches = baselines.filter(ep => String(ep.seed) === String(left?.seed) && ep.map === left?.map && ep.ignition === left?.ignition);
  for (const ep of matches) option($('baseline'), String(ep.id), policyName(ep));
  if (matches.some(ep => String(ep.id) === previous)) $('baseline').value = previous;
  if (!matches.length) option($('baseline'), '', 'No matching recording');
  right = matches.find(ep => String(ep.id) === $('baseline').value);
  updateDetails(); render();
}
function draw(id, frame) {
  const canvas = $(id), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frame?.rgb?.length) { ctx.fillStyle = '#666'; ctx.font = '18px system-ui'; ctx.fillText('No matching recording', 25, 45); return; }
  const height = frame.rgb.length, width = frame.rgb[0].length;
  const cw = canvas.width / width, ch = canvas.height / height;
  frame.rgb.forEach((row, r) => row.forEach((pixel, c) => { ctx.fillStyle = `rgb(${pixel.slice(0, 3).join(',')})`; ctx.fillRect(c*cw, r*ch, Math.ceil(cw), Math.ceil(ch)); }));
  if (Number.isInteger(frame.action) && frame.action >= 0 && frame.action < height*width) {
    const row = Math.floor(frame.action / width), col = frame.action % width;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 5; ctx.strokeRect(col*cw+3, row*ch+3, cw-6, ch-6);
    ctx.strokeStyle = '#111'; ctx.lineWidth = 2; ctx.strokeRect(col*cw+3, row*ch+3, cw-6, ch-6);
  }
}
function panel(side, episode) {
  const frame = selectedFrame(episode);
  $(side+'-title').textContent = episode ? policyName(episode) : 'Comparison unavailable';
  $(side+'-subtitle').textContent = episode ? (checkpointName(episode).split('/').at(-1) || 'Placement rule; no learned weights') : 'Choose a seed with a matching baseline recording.';
  draw(side+'-grid', frame);
  for (const [id, key, digits] of [['burning','burning',0],['harvested','harvested',0],['reward','reward',3],['return','return',3]]) $(side+'-'+id).textContent = number(frame?.[key], digits);
  const ended = episode && interval >= lastStep(episode);
  $(side+'-state').textContent = !episode ? '' : ended ? `Final recorded frame · ${episode.status || 'episode recording ended'}` : `Decision interval ${frame?.step ?? 0} of ${lastStep(episode)}`;
  $(side+'-action').textContent = frame ? `Action reaching this frame: ${cell(frame.action, episode)}${Number.isInteger(frame.action) && frame.action >= 0 ? ` · cell ID ${frame.action}` : ''}` : '';
}
function render() {
  if (!left) return;
  $('interval').textContent = `${interval} / ${maximum()}`;
  $('timeline').max = maximum(); $('timeline').value = interval;
  $('back').disabled = interval <= 0; $('next').disabled = interval >= maximum(); $('play').disabled = interval >= maximum();
  panel('left', left); panel('right', right);
  for (const row of $('trajectory').tBodies[0].rows) row.classList.toggle('current', Number(row.dataset.step) === interval);
}
function addRow(tbody, values, step) { const row = tbody.insertRow(); if (step !== undefined) row.dataset.step = step; for (const value of values) row.insertCell().textContent = value; }
function updateDetails() {
  if (!left) return;
  $('scenario').textContent = `${left.map || 'Recorded map'} · seed ${left.seed} · ${frames(left)[0]?.rgb.length || '?'} × ${frames(left)[0]?.rgb?.[0]?.length || '?'} cells. Comparisons use the same map and recorded seed.`;
  $('checkpoint-detail').textContent = checkpointName(left) || 'Checkpoint name not recorded';
  $('checkpoint-hash').textContent = left.checkpoint_sha256 ? `SHA-256: ${left.checkpoint_sha256}` : '';
  const steps = left.training_steps ?? left.trained_steps ?? left.checkpoint?.training_steps;
  $('training-detail').textContent = steps !== undefined && steps !== null ? `This saved checkpoint records ${number(steps)} training steps. This page performs playback only.` : 'The training-step count is not recorded in this episode metadata.';
  const provenance = data.provenance || {};
  $('provenance').textContent = typeof provenance === 'string' ? provenance : JSON.stringify(provenance, null, 2);
  const physics = left.physics_steps_per_action ?? left.steps_per_action ?? provenance.steps_per_action ?? provenance.physics_steps_per_action;
  const warmup = left.warmup_steps ?? provenance.warmup_steps;
  $('timing-note').textContent = physics !== undefined ? `Here, one decision interval advances ${physics} Cell2Fire simulator steps${warmup !== undefined ? `, following ${warmup} initial warm-up steps` : ''}.` : 'Exact simulator timing is recorded under Checkpoint and simulator details when available.';
  const body = $('trajectory').tBodies[0]; body.replaceChildren();
  const allSteps = [...new Set([...frames(left), ...frames(right)].map(f => f.step))].sort((a,b) => a-b);
  for (const step of allSteps) {
    const a = frames(left).find(f => f.step === step), b = frames(right).find(f => f.step === step);
    addRow(body, [step, a ? cell(a.action,left) : 'Ended', number(a?.reward,3), number(a?.burning), b ? cell(b.action,right) : 'Ended', number(b?.reward,3), number(b?.burning)], step);
  }
}
function allResults() {
  const body = $('results').tBodies[0]; body.replaceChildren();
  for (const ep of data.episodes) {
    const end = frames(ep).at(-1), metrics = ep.metrics || {};
    addRow(body,[policyName(ep), ep.seed, lastStep(ep), number(metrics.fire_affected_cells ?? end?.burning), number(metrics.harvested_cells ?? metrics.treated_cells ?? end?.harvested), number(end?.return ?? metrics.episode_return,3), ep.status || 'Recorded']);
  }
}
async function load() {
  try {
    if (window.FIREHOSE_DATA) data = window.FIREHOSE_DATA;
    else { const response = await fetch('data.json', {cache:'no-store'}); if (!response.ok) throw new Error(`data.json returned ${response.status}`); data = await response.json(); }
    if (!Array.isArray(data.episodes) || !data.episodes.length) throw new Error('No recorded episodes are available.');
    if (data.episodes.some(ep => !Array.isArray(ep.frames) || !ep.frames.length)) throw new Error('A recording has no frames. Generate complete episode data before viewing.');
    learned = data.episodes.filter(isLearned); baselines = data.episodes.filter(ep => !isLearned(ep));
    if (!learned.length) throw new Error('No saved neural-policy recording is available.');
    $('checkpoint').replaceChildren();
    const seen = new Set(); for (const ep of learned) { const key = learnedKey(ep); if (!seen.has(key)) { option($('checkpoint'),key,`${ep.map} · ${key.split('/').at(-1)}`); seen.add(key); } }
    $('viewer').hidden = false; $('message').hidden = true;
    fillSeeds(); allResults();
  } catch (error) {
    $('viewer').hidden = true; $('message').hidden = false;
    $('message').textContent = `Recorded data could not be loaded. ${error.message} Start the local server described in README.md, or use the generated offline HTML. No example simulation has been substituted.`;
  }
}
$('checkpoint').onchange = fillSeeds;
$('seed').onchange = selectEpisodes;
$('baseline').onchange = () => { stop(); interval = 0; right = baselines.find(ep => String(ep.id) === $('baseline').value); updateDetails(); render(); };
$('back').onclick = () => { stop(); interval = Math.max(0, interval-1); render(); };
$('next').onclick = () => { stop(); interval = Math.min(maximum(), interval+1); render(); };
$('play').onclick = () => { if (timer) stop(); else start(); };
$('timeline').oninput = event => { stop(); interval = Number(event.target.value); render(); };
$('speed').onchange = () => { if (timer) { stop(); start(); } };
load();
