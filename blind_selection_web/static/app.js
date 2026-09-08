const state = { file: null, job: null, points: [], zoom: 100, activeCrop: null, waiting: false };
const $ = (id) => document.getElementById(id);

function toast(message) {
  const node = $('toast'); node.textContent = message; node.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.hidden = true, 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Fehler ${response.status}`);
  return payload;
}

function show(id, visible = true) { $(id).hidden = !visible; }
function scrollToSection(id) { $(id).scrollIntoView({ behavior: 'smooth', block: 'start' }); }

function chooseFile(file) {
  if (!file) return;
  if (!/\.tiff?$/i.test(file.name)) return toast('Bitte eine TIFF-Datei auswählen.');
  state.file = file; $('file-label').textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  $('upload-button').disabled = false;
}

async function upload() {
  if (!state.file) return;
  $('upload-button').disabled = true; $('upload-button').textContent = 'Wird geladen …';
  try {
    const job = await api('/api/jobs', { method: 'POST', headers: { 'X-Filename': encodeURIComponent(state.file.name), 'Content-Type': 'image/tiff' }, body: state.file });
    openJob(job);
  } catch (error) { toast(error.message); $('upload-button').disabled = false; $('upload-button').innerHTML = 'Rohbild öffnen <span>→</span>'; }
}

function openJob(job) {
  state.job = job; state.points = (job.selection?.points || []).map(p => ({ x: p.x, y: p.y }));
  if (job.status === 'selecting') renderSelection();
  else if (job.status === 'completed') renderReview();
  else renderProgress();
}

function renderSelection() {
  show('selection-section'); show('progress-section', false); show('review-section', false);
  const image = $('raw-image'); image.src = `${state.job.image.preview_url}?v=${state.job.version}`;
  image.onload = () => { configureOverlay(); renderMarkers(); };
  updateSelectionCount(); scrollToSection('selection-section');
}

function configureOverlay() {
  const svg = $('marker-layer'), image = state.job.image;
  svg.setAttribute('viewBox', `0 0 ${image.width} ${image.height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
}

function updateSelectionCount() {
  $('selection-count').textContent = state.points.length;
  $('selection-status').textContent = state.points.length ? `${state.points.length} Zelle${state.points.length === 1 ? '' : 'n'} ausgewählt.` : 'Noch keine Zelle ausgewählt.';
  $('lock-button').disabled = state.points.length === 0;
  $('undo-button').disabled = state.points.length === 0; $('clear-button').disabled = state.points.length === 0;
}

function renderMarkers() {
  const svg = $('marker-layer'), ns = 'http://www.w3.org/2000/svg'; svg.replaceChildren();
  const radius = Math.max(state.job.image.width, state.job.image.height) / 120;
  state.points.forEach((point, index) => {
    const group = document.createElementNS(ns, 'g'); group.classList.add('marker');
    group.innerHTML = `<circle cx="${point.x}" cy="${point.y}" r="${radius * 1.8}"></circle><circle cx="${point.x}" cy="${point.y}" r="${radius * .55}"></circle><text x="${point.x + radius}" y="${point.y - radius}">${index + 1}</text>`;
    svg.appendChild(group);
  }); updateSelectionCount();
}

function clickImage(event) {
  if (!state.job || state.job.status !== 'selecting') return;
  const rect = $('marker-layer').getBoundingClientRect();
  const x = Math.max(0, Math.min(state.job.image.width - 1, Math.round((event.clientX - rect.left) * state.job.image.width / rect.width)));
  const y = Math.max(0, Math.min(state.job.image.height - 1, Math.round((event.clientY - rect.top) * state.job.image.height / rect.height)));
  const hitRadius = 18 * state.job.image.width / rect.width;
  const found = state.points.findIndex(p => Math.hypot(p.x - x, p.y - y) <= hitRadius);
  if (found >= 0) state.points.splice(found, 1); else state.points.push({ x, y });
  renderMarkers();
}

function setZoom(value) {
  state.zoom = Number(value); $('zoom-output').textContent = `${state.zoom}%`;
  $('image-stage').style.width = `${state.zoom}%`;
}

async function lockSelection() {
  if (!state.points.length || !state.job) return;
  if (!confirm(`${state.points.length} Zellen verbindlich auswählen? Danach startet die Verarbeitung und die Klicks können nicht mehr geändert werden.`)) return;
  $('lock-button').disabled = true;
  try {
    state.job = await api(`/api/jobs/${state.job.id}/selection`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: state.points }) });
    renderProgress(); waitForJob();
  } catch (error) { toast(error.message); $('lock-button').disabled = false; }
}

const phases = ['prediction', 'hysteresis', 'selection_matching', 'instance_separation', 'evo_finalization'];
function renderProgress() {
  show('selection-section', false); show('progress-section'); show('review-section', false);
  const job = state.job; $('progress-title').textContent = job.status === 'failed' ? 'Verarbeitung fehlgeschlagen' : 'Verarbeitung läuft';
  $('status-pill').textContent = job.status; $('status-pill').classList.toggle('failed', job.status === 'failed');
  $('progress-bar').style.width = `${job.progress || 0}%`; $('progress-percent').textContent = `${job.progress || 0}%`; $('progress-message').textContent = job.message || '';
  const phaseIndex = phases.indexOf(job.phase);
  document.querySelectorAll('.phase-grid [data-phase]').forEach((node, index) => { node.classList.toggle('active', index === phaseIndex); node.classList.toggle('done', index < phaseIndex || job.status === 'completed'); });
  $('event-log').innerHTML = (job.events || []).slice(-15).map(e => `<li>${escapeHtml(e.message)}</li>`).join('');
  scrollToSection('progress-section');
}

async function waitForJob() {
  if (state.waiting || !state.job || ['completed', 'failed'].includes(state.job.status)) return;
  state.waiting = true;
  try {
    while (state.job && !['completed', 'failed'].includes(state.job.status)) {
      state.job = await api(`/api/jobs/${state.job.id}/events?version=${state.job.version || 0}`);
      renderProgress();
    }
    if (state.job.status === 'completed') renderReview(); else toast(state.job.message || 'Verarbeitung fehlgeschlagen.');
    refreshHistory();
  } catch (error) { toast(error.message); }
  finally { state.waiting = false; }
}

function pendingCrops() { return (state.job?.crops || []).filter(c => !['good', 'bad'].includes(c.review?.result)); }
function renderReview() {
  show('selection-section', false); show('progress-section', false); show('review-section');
  const summary = state.job.summary || {}, reviews = state.job.review_summary || {};
  $('review-progress').textContent = `${reviews.reviewed || 0} / ${reviews.total || 0}`;
  $('review-approved').textContent = `${reviews.approved || 0} freigegeben`;
  const failed = summary.failed_selections || 0;
  $('matching-warning').hidden = failed === 0;
  $('matching-warning').textContent = failed ? `${failed} Auswahl${failed === 1 ? '' : 'en'} konnte${failed === 1 ? '' : 'n'} nicht sicher zugeordnet oder getrennt werden und wurde${failed === 1 ? '' : 'n'} nicht als Crop angeboten.` : '';
  const pending = pendingCrops();
  if (!pending.length) { show('cell-review', false); show('review-finished'); $('finished-copy').textContent = `${reviews.approved || 0} Zellen wurden für Evo gespeichert, ${reviews.rejected || 0} verworfen.`; }
  else { show('review-finished', false); show('cell-review'); renderCrop(pending[0]); }
  scrollToSection('review-section');
}

function renderCrop(crop) {
  state.activeCrop = crop; $('cell-title').textContent = `${crop.selection_id || crop.folder} · ${crop.folder}`;
  $('cell-meta').textContent = `${crop.skeleton_pixels} Skeleton-px · ${crop.soma_pixels} Soma-px`;
  $('crop-raw').src = crop.raw_preview_url; $('crop-prediction').src = crop.prediction_preview_url;
  $('crop-hysteresis').src = crop.postprocessing_preview_url; $('crop-final').src = crop.preview_url;
  $('cell-review').focus({ preventScroll: true });
}

async function rate(result) {
  if (!state.activeCrop) return;
  const crop = state.activeCrop; state.activeCrop = null;
  $('accept-button').disabled = true; $('reject-button').disabled = true;
  try {
    const response = await api(`/api/jobs/${state.job.id}/reviews/${crop.folder}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ result }) });
    crop.review = response.review; state.job.review_summary = response.review_summary;
    renderReview();
  } catch (error) { state.activeCrop = crop; toast(error.message); }
  finally { $('accept-button').disabled = false; $('reject-button').disabled = false; }
}

async function refreshHistory() {
  try {
    const payload = await api('/api/jobs');
    $('history-list').innerHTML = payload.jobs.length ? payload.jobs.map(job => `<div class="history-row" data-job="${job.id}"><div><strong>${escapeHtml(job.filename)}</strong><small>${job.selection?.count || 0} ausgewählt · ${job.review_summary?.approved || 0} für Evo</small></div><span class="history-state">${escapeHtml(job.status)}</span><button>Öffnen</button></div>`).join('') : '<p>Noch keine Läufe.</p>';
    document.querySelectorAll('[data-job]').forEach(node => node.addEventListener('click', async () => { try { openJob(await api(`/api/jobs/${node.dataset.job}`)); if (!['selecting','completed','failed'].includes(state.job.status)) waitForJob(); } catch (error) { toast(error.message); } }));
  } catch (error) { toast(error.message); }
}

function escapeHtml(value) { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; }

$('file-input').addEventListener('change', e => chooseFile(e.target.files[0]));
$('dropzone').addEventListener('dragover', e => { e.preventDefault(); e.currentTarget.classList.add('drag'); });
$('dropzone').addEventListener('dragleave', e => e.currentTarget.classList.remove('drag'));
$('dropzone').addEventListener('drop', e => { e.preventDefault(); e.currentTarget.classList.remove('drag'); chooseFile(e.dataTransfer.files[0]); });
$('upload-button').addEventListener('click', upload); $('marker-layer').addEventListener('click', clickImage);
$('undo-button').addEventListener('click', () => { state.points.pop(); renderMarkers(); });
$('clear-button').addEventListener('click', () => { state.points = []; renderMarkers(); });
$('zoom-slider').addEventListener('input', e => setZoom(e.target.value)); $('lock-button').addEventListener('click', lockSelection);
$('accept-button').addEventListener('click', () => rate('good')); $('reject-button').addEventListener('click', () => rate('bad'));
$('refresh-button').addEventListener('click', refreshHistory);
document.addEventListener('keydown', event => { if (!$('review-section').hidden && state.activeCrop && !event.repeat) { if (event.key === 'ArrowRight') { event.preventDefault(); rate('good'); } if (event.key === 'ArrowLeft') { event.preventDefault(); rate('bad'); } } });

api('/api/config').then(config => { $('network-label').textContent = config.network; $('output-path').textContent = `Freigegebene Evo-Zellen: ${config.output_root}\\cells`; }).catch(error => toast(error.message));
refreshHistory();
