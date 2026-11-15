const $ = id => document.getElementById(id);
let episode = null, frameIndex = 0, timer = null;
const names = {dqn: 'DQN with wind', dqn_no_wind: 'DQN without wind input', random: 'Random placement',
  greedy: 'Near-fire heuristic', wind_heuristic: 'Wind placement heuristic', no_intervention: 'No construction'};
const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const pct = value => (100 * value).toFixed(1) + '%';
const subset = cells => cells.length ? 'Clear cell ' + cells.join(', ') : 'Wait';
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $('play').textContent = 'Play';
}
async function load() {
  stop();
  $('error').hidden = true;
  $('reset').disabled = true;
  $('reset').textContent = 'Simulating…';
  try {
    let candidate;
    if (window.DEMO_DATA) {
      candidate = window.DEMO_DATA.episodes[$('policy').value];
      if (!candidate) throw Error(window.DEMO_DATA.unavailable[$('policy').value] || 'Replay unavailable');
    } else {
      const query = new URLSearchParams({policy: $('policy').value, seed: $('seed').value, budget: $('budget').value});
      const response = await fetch('/api/episode?' + query);
      candidate = await response.json();
      if (!response.ok) throw Error(candidate.error);
    }
    episode = candidate;
    frameIndex = 0;
    $('mode').textContent = (window.DEMO_DATA ? 'Offline replay: ' : 'Simulator: ') + names[episode.policy];
    $('timeline').disabled = false;
    $('timeline').max = episode.frames.length - 1;
    $('arm').innerHTML = episode.arms.map(arm => `<option value="${arm.arm_id}">Cell ${arm.arm_id} (${arm.row}, ${arm.col})</option>`).join('');
    $('dimensions').textContent = `${episode.arms.length} cell arms · ${episode.action_count} Q outputs, including wait`;
    $('capacity').textContent = episode.config.capacity + ' cell' + (episode.config.capacity === 1 ? '' : 's');
    $('parameters').textContent = episode.model ? `${episode.model.parameter_count.toLocaleString()} learned parameters · γ = ${episode.model.gamma} · checkpoint at ${episode.model.training_step.toLocaleString()} training intervals` : 'No neural network is used by this baseline.';
    $('architecture').textContent = episode.model ? episode.model.architecture : 'Choose a DQN policy with matching saved weights to inspect its network.';
    render();
  } catch (error) {
    $('error').textContent = error.message;
    $('error').hidden = false;
    $('policy').value = episode ? episode.policy : 'wind_heuristic';
    if (!episode) for (const id of ['step', 'back', 'play', 'timeline']) $(id).disabled = true;
  } finally {
    $('reset').disabled = false;
    $('reset').textContent = 'Load episode';
  }
}
function drawGrid(frame) {
  const canvas = $('grid'), ctx = canvas.getContext('2d'), n = episode.config.grid_size, size = canvas.width / n;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    ctx.fillStyle = `rgb(${frame.rgb[row][col].join(',')})`;
    ctx.fillRect(col * size, row * size, size, size);
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.strokeRect(col * size, row * size, size, size);
  }
  for (const arm of episode.arms) {
    const selected = frame.allocation.includes(arm.arm_id);
    const focus = $('arm-details').open && arm.arm_id === Number($('arm').value);
    if (!selected && !focus && (!$('candidates').checked || !frame.eligible[arm.arm_id])) continue;
    ctx.strokeStyle = selected ? '#ffe176' : focus ? '#fff' : 'rgba(255,255,255,.35)';
    ctx.lineWidth = selected ? 4 : focus ? 3 : 1;
    for (const [row, col] of arm.cells) ctx.strokeRect(col * size + 3, row * size + 3, size - 6, size - 6);
    if (selected || focus) {
      ctx.fillStyle = '#172e22';
      ctx.fillRect(arm.col * size + 5, arm.row * size + 5, 30, 21);
      ctx.fillStyle = selected ? '#ffe176' : '#fff';
      ctx.font = 'bold 13px system-ui';
      ctx.fillText(arm.arm_id, arm.col * size + 8, arm.row * size + 20);
    }
  }
}
function render() {
  if (!episode) return;
  const frame = episode.frames[frameIndex];
  $('timeline').value = frameIndex;
  $('tick').textContent = frame.t + ' / ' + episode.config.max_steps;
  $('materials').textContent = frame.budget === null ? 'No episode cap' : frame.budget + ' / ' + episode.initial_budget;
  $('built').textContent = frame.built.filter(Boolean).length + ' / ' + episode.arms.length;
  $('affected').textContent = pct(frame.affected);
  $('allocation').textContent = frame.finished ? 'Episode finished' : subset(frame.allocation);
  $('action-detail').textContent = frame.finished ? episode.ending : `Action ${frame.action} · ${frame.allocation.length} active arms · ${frame.allocation.length * episode.config.resource_cost} construction units`;
  $('feasible').textContent = frame.feasible + ' / ' + episode.action_count;
  $('observed-wind').textContent = frame.wind_observation.map(value => fmt(value, 2)).join(', ');
  $('actual-wind').textContent = episode.actual_wind.map(value => fmt(value, 2)).join(', ');
  $('ending').textContent = frame.finished ? episode.ending : `${frame.active_fire} burning cells · wind stays fixed throughout this episode`;
  $('q-note').textContent = episode.model ? 'Actual feasible Q values; larger is better. ' + (episode.action_count - frame.feasible) + ' entries are masked.' : 'This baseline follows a placement rule and has no learned Q values.';
  $('q-values').innerHTML = frame.q_values.length ? frame.q_values.map(q => `<tr><td>${subset(q.arms)}</td><td>${fmt(q.q)}</td></tr>`).join('') : '<tr><td colspan="2">' + (frame.finished ? 'No decision at the episode boundary.' : 'No DQN Q values for this policy.') + '</td></tr>';
  $('checkpoint').textContent = episode.model ? episode.model.checkpoint + ' · selected at step ' + episode.model.training_step : 'Observation-only baseline';
  const index = Number($('arm').value), features = frame.arm_features[index], arm = episode.arms[index];
  $('arm-state').innerHTML = [['Construction', frame.built[index] ? 'Built; persistent' : 'Unbuilt'], ['Eligibility', frame.eligible[index] ? 'Available' : 'Unavailable'], ['Local burning', pct(features[3])], ['Local fuel', fmt(features[2])], ['Coordinates', `(${arm.row}, ${arm.col})`]].map(([label, value]) => `<div class="fact"><span>${label}</span><b>${value}</b></div>`).join('');
  $('reward').textContent = 'Reward ' + fmt(frame.previous_reward);
  $('return').textContent = fmt(frame.discounted_return);
  $('transition-text').textContent = frameIndex ? `${subset(episode.frames[frameIndex - 1].allocation)} executed. Fire advanced once; existing firebreaks remained.` : 'Press Next step to run the decision above.';
  $('back').disabled = frameIndex === 0;
  $('step').disabled = frame.finished;
  $('play').disabled = frame.finished;
  drawGrid(frame);
}
async function results() {
  try {
    const data = window.DEMO_DATA ? window.DEMO_DATA.results : await (await fetch('/api/results')).json();
    $('result-status').textContent = data.status;
    const metric = (method, key) => data.rows.find(row => row.method === method && row.metric === key);
    const methods = Object.keys(names).filter(method => metric(method, 'discounted_return'));
    $('comparison').querySelector('tbody').innerHTML = methods.length ? methods.map(method => {
      const result = metric(method, 'discounted_return'), built = metric(method, 'budget_used');
      return `<tr><td>${names[method]}</td><td>${fmt(result.mean)} [${fmt(result.ci_low)}, ${fmt(result.ci_high)}]</td><td>${pct(metric(method, 'affected_fraction').mean)}</td><td>${built ? fmt(built.mean, 1) : '—'}</td></tr>`;
    }).join('') : '<tr><td colspan="4">New diagnostic results have not been generated.</td></tr>';
    const canvas = $('learning'), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data.learning.length) return;
    const lo = Math.min(...data.learning.map(row => +row.discounted_return)) - .05;
    const hi = Math.max(...data.learning.map(row => +row.discounted_return)) + .05;
    const maxStep = Math.max(...data.learning.map(row => +row.step));
    const x = step => 60 + (+step / maxStep) * 910, y = value => 270 - (+value - lo) / (hi - lo) * 235;
    ctx.font = '13px system-ui';
    ctx.fillStyle = '#555';
    ctx.strokeStyle = '#ddd';
    for (let i = 0; i < 5; i++) {
      const value = lo + (hi - lo) * i / 4;
      ctx.beginPath(); ctx.moveTo(55, y(value)); ctx.lineTo(980, y(value)); ctx.stroke();
      ctx.fillText(fmt(value, 2), 4, y(value) + 4);
    }
    for (let i = 1; i <= 4; i++) ctx.fillText(Math.round(maxStep * i / 4).toLocaleString(), x(maxStep * i / 4) - 25, 300);
    for (const method of ['dqn', 'dqn_no_wind']) for (const seed of new Set(data.learning.map(row => row.seed))) {
      const rows = data.learning.filter(row => row.method === method && row.seed === seed);
      ctx.beginPath();
      rows.forEach((row, index) => index ? ctx.lineTo(x(row.step), y(row.discounted_return)) : ctx.moveTo(x(row.step), y(row.discounted_return)));
      ctx.strokeStyle = method === 'dqn' ? '#34734d' : '#d98543';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  } catch (error) { $('result-status').textContent = 'Could not load results: ' + error.message; }
}
$('results').addEventListener('toggle', () => { if ($('results').open) results(); });
$('arm-details').addEventListener('toggle', render);
$('reset').onclick = load;
$('policy').onchange = load;
$('arm').onchange = render;
$('candidates').onchange = render;
$('step').onclick = () => { if (!episode) return; stop(); frameIndex = Math.min(frameIndex + 1, episode.frames.length - 1); render(); };
$('back').onclick = () => { if (!episode) return; stop(); frameIndex = Math.max(0, frameIndex - 1); render(); };
$('timeline').oninput = () => { stop(); frameIndex = Number($('timeline').value); render(); };
$('play').onclick = () => {
  if (!episode) return;
  if (timer) { stop(); return; }
  $('play').textContent = 'Pause';
  timer = setInterval(() => { frameIndex = Math.min(frameIndex + 1, episode.frames.length - 1); render(); if (frameIndex === episode.frames.length - 1) stop(); }, 600);
};
if (window.DEMO_DATA) {
  $('seed').disabled = true;
  $('budget').disabled = true;
  for (const option of $('policy').options) {
    if (!window.DEMO_DATA.episodes[option.value]) { option.disabled = true; option.textContent += ' (weights unavailable)'; }
  }
  $('scenario-note').textContent = 'Offline replay: seed 42, one construction per interval, no episode cap. Run demo/server.py to generate other episodes.';
}
load();
results();
