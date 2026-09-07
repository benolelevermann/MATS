const state = {
  config: null,
  case: null,
  selectedCell: null,
  tileRow: 0,
  tileColumn: 0,
  tool: "skeleton",
  radius: 3,
  view: "combined",
  drawing: false,
  points: [],
  busy: false,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const number = (value) => Number(value || 0).toLocaleString("de-DE");

async function request(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Fehler ${response.status}`);
  return payload;
}

function setSaveState(mode, text) {
  const element = $("#save-state");
  element.className = `save-state ${mode || ""}`;
  element.innerHTML = `<span></span> ${text}`;
}

function toast(message, error = false) {
  const element = $("#toast");
  clearTimeout(state.toastTimer);
  element.textContent = message;
  element.className = `toast${error ? " error" : ""}`;
  element.hidden = false;
  state.toastTimer = setTimeout(() => { element.hidden = true; }, error ? 5000 : 2800);
}

function setBusy(busy, label = "Speichert …") {
  state.busy = busy;
  if (busy) setSaveState("saving", label);
  else setSaveState("", "Automatisch gespeichert");
}

function enableWorkspace(enabled) {
  $$(".tool, #radius, #complete-button, #export-button, .tile-nav button").forEach((element) => {
    element.disabled = !enabled;
  });
}

function renderCases() {
  const list = $("#case-list");
  list.replaceChildren();
  $("#case-total").textContent = state.config.cases.length;
  for (const item of state.config.cases) {
    const button = document.createElement("button");
    button.className = `case-button${state.case?.id === item.id ? " active" : ""}`;
    const stateClass = item.completed ? "complete" : item.cells_count ? "started" : "";
    button.innerHTML = `<i class="case-state ${stateClass}"></i><span><strong>${item.name}</strong><small>${item.width.toLocaleString("de-DE")} × ${item.height.toLocaleString("de-DE")} px · ${item.cells_count || 0} Zellen</small></span>`;
    button.addEventListener("click", () => selectCase(item.id));
    list.append(button);
  }
}

async function selectCase(caseId) {
  if (state.busy) return;
  try {
    setBusy(true, "Mosaik wird geladen …");
    state.case = await request(`/api/cases/${caseId}`);
    const prior = state.case.cells.find((cell) => cell.id === state.selectedCell);
    state.selectedCell = prior?.id || state.case.cells[0]?.id || null;
    state.tileRow = 0;
    state.tileColumn = 0;
    updateCachedCase();
    renderCases();
    renderCase();
    $("#empty-state").hidden = true;
    $("#canvas-shell").hidden = false;
    $("#overview-card").hidden = false;
    enableWorkspace(true);
    await loadTile();
    $("#overview-image").src = `/api/cases/${caseId}/thumbnail.png?v=${state.case.revision}`;
    setBusy(false);
  } catch (error) {
    setSaveState("error", "Fehler beim Laden");
    toast(error.message, true);
  }
}

function updateCachedCase() {
  if (!state.case || !state.config) return;
  const index = state.config.cases.findIndex((item) => item.id === state.case.id);
  if (index >= 0) state.config.cases[index] = { ...state.config.cases[index], ...state.case };
}

function renderCase() {
  if (!state.case) return;
  const current = state.case;
  $("#case-title").textContent = current.name;
  $("#complete-button").textContent = current.completed ? "Fertig ✓" : "Als fertig markieren";
  $("#complete-button").classList.toggle("primary", current.completed);
  $("#cell-total").textContent = current.cells.length;
  const coverage = current.coverage;
  $("#soma-coverage").textContent = `${coverage.soma_percent.toLocaleString("de-DE")} %`;
  $("#skeleton-coverage").textContent = `${coverage.skeleton_percent.toLocaleString("de-DE")} %`;
  $("#soma-bar").style.width = `${coverage.soma_percent}%`;
  $("#skeleton-bar").style.width = `${coverage.skeleton_percent}%`;
  $("#unassigned-count").textContent = number(coverage.unassigned_foreground);
  $("#ambiguous-count").textContent = number(coverage.ambiguous);
  renderCells();
  updateTilePosition();
  updateOverviewMarker();
}

function renderCells() {
  const list = $("#cell-list");
  list.replaceChildren();
  $("#active-cell-empty").hidden = state.case.cells.length > 0;
  for (const cell of state.case.cells) {
    const button = document.createElement("button");
    button.className = `cell-button${state.selectedCell === cell.id ? " active" : ""}`;
    button.innerHTML = `<i class="cell-color" style="background:${cell.color}"></i><span><strong>${cell.label}</strong><small>${number(cell.skeleton_pixels)} Skeleton · ${number(cell.soma_pixels)} Soma</small></span><i class="validity ${cell.valid ? "valid" : ""}"></i>`;
    button.addEventListener("click", async () => {
      state.selectedCell = cell.id;
      renderCells();
      await loadTile();
    });
    list.append(button);
  }
  const selected = state.case.cells.find((cell) => cell.id === state.selectedCell);
  $("#selection-card").hidden = !selected;
  if (selected) {
    $("#selection-color").style.background = selected.color;
    $("#selection-label").textContent = selected.label;
    $("#selection-skeleton").textContent = number(selected.skeleton_pixels);
    $("#selection-soma").textContent = number(selected.soma_pixels);
  }
}

function tileIndex() {
  return state.tileRow * state.case.grid.columns + state.tileColumn;
}

function updateTilePosition() {
  if (!state.case) return;
  const total = state.case.grid.rows * state.case.grid.columns;
  const x = state.tileColumn * 512;
  const y = state.tileRow * 512;
  $("#tile-position").textContent = `Bereich ${state.tileColumn + 1},${state.tileRow + 1} · x ${x}–${Math.min(state.case.width, x + 512)}, y ${y}–${Math.min(state.case.height, y + 512)}`;
  $("#tile-number").textContent = `${tileIndex() + 1} / ${total}`;
}

function updateOverviewMarker() {
  if (!state.case) return;
  const marker = $("#overview-marker");
  marker.style.left = `${100 * state.tileColumn * 512 / state.case.width}%`;
  marker.style.top = `${100 * state.tileRow * 512 / state.case.height}%`;
  marker.style.width = `${100 * Math.min(512, state.case.width - state.tileColumn * 512) / state.case.width}%`;
  marker.style.height = `${100 * Math.min(512, state.case.height - state.tileRow * 512) / state.case.height}%`;
}

async function loadTile() {
  if (!state.case) return;
  const query = new URLSearchParams({
    x: state.tileColumn * 512,
    y: state.tileRow * 512,
    size: 512,
    view: state.view,
    v: state.case.revision,
  });
  if (state.selectedCell) query.set("selected", state.selectedCell);
  const image = $("#tile-image");
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("Bildausschnitt konnte nicht geladen werden"));
    image.src = `/api/cases/${state.case.id}/tile.png?${query}`;
  });
  const canvas = $("#draw-canvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  updateTilePosition();
  updateOverviewMarker();
}

function setTool(tool) {
  state.tool = tool;
  $$(".tool").forEach((button) => button.classList.toggle("active", button.dataset.tool === tool));
  if (tool === "new-soma") toast("Klicke jetzt auf das Soma der neuen Zelle.");
  if (tool === "fill") toast("Klicke eine zusammenhängende Soma- oder Skeletonkomponente an.");
  if (tool === "overlap") toast("Ordne diese Pixel zusätzlich der ausgewählten Zelle zu.");
}

async function createCell() {
  if (!state.case || state.busy) return;
  try {
    setBusy(true);
    const updated = await request(`/api/cases/${state.case.id}/cells`, {
      method: "POST",
      body: JSON.stringify({ action: "create" }),
    });
    state.case = updated;
    state.selectedCell = updated.cells.at(-1).id;
    updateCachedCase();
    renderCase();
    setTool("new-soma");
    setBusy(false);
  } catch (error) { handleError(error); }
}

function canvasPoint(event) {
  const canvas = $("#draw-canvas");
  const rect = canvas.getBoundingClientRect();
  const localX = Math.max(0, Math.min(511, (event.clientX - rect.left) * 512 / rect.width));
  const localY = Math.max(0, Math.min(511, (event.clientY - rect.top) * 512 / rect.height));
  return [localX + state.tileColumn * 512, localY + state.tileRow * 512];
}

function drawTemporary() {
  const canvas = $("#draw-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, 512, 512);
  if (!state.points.length) return;
  const selected = state.case.cells.find((cell) => cell.id === state.selectedCell);
  const color = state.tool === "ambiguous" ? "#ffc247" : state.tool === "erase" ? "#ff7777" : (selected?.color || "#ffffff");
  context.strokeStyle = color;
  context.fillStyle = color;
  context.globalAlpha = .82;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = Math.max(1, state.radius * 2 + 1);
  context.beginPath();
  state.points.forEach(([globalX, globalY], index) => {
    const x = globalX - state.tileColumn * 512;
    const y = globalY - state.tileRow * 512;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  if (state.points.length === 1) {
    const [x, y] = state.points[0];
    context.beginPath();
    context.arc(x - state.tileColumn * 512, y - state.tileRow * 512, Math.max(.75, state.radius), 0, Math.PI * 2);
    context.fill();
  }
  context.globalAlpha = 1;
}

async function handleCanvasClick(point) {
  if (!state.selectedCell && !["ambiguous", "clear-ambiguous"].includes(state.tool)) {
    toast("Lege zuerst mit N eine Zelle an.", true);
    return;
  }
  if (state.tool === "new-soma") {
    await sendFill(point, "assign", "soma");
    setTool("skeleton");
    return;
  }
  if (state.tool === "fill") {
    await sendFill(point, "assign", "auto");
  }
}

async function sendFill(point, operation, target) {
  try {
    setBusy(true);
    const payload = await request(`/api/cases/${state.case.id}/fill`, {
      method: "POST",
      body: JSON.stringify({
        x: Math.round(point[0]), y: Math.round(point[1]), operation, target,
        cell_id: state.selectedCell,
      }),
    });
    await acceptChange(payload, "Komponente zugeordnet");
  } catch (error) { handleError(error); }
}

async function sendStroke() {
  const mapping = {
    skeleton: ["assign", "skeleton"],
    soma: ["assign", "soma"],
    overlap: ["overlap", "foreground"],
    ambiguous: ["ambiguous", "foreground"],
    "clear-ambiguous": ["clear_ambiguous", "foreground"],
    erase: ["erase", "foreground"],
  };
  if (!mapping[state.tool]) return;
  if (!state.selectedCell && !["ambiguous", "clear-ambiguous", "erase"].includes(state.tool)) {
    toast("Lege zuerst mit N eine Zelle an.", true);
    return;
  }
  const [operation, target] = mapping[state.tool];
  try {
    setBusy(true);
    const payload = await request(`/api/cases/${state.case.id}/stroke`, {
      method: "POST",
      body: JSON.stringify({
        points: state.points,
        radius: state.radius,
        operation,
        target,
        cell_id: state.selectedCell,
      }),
    });
    const skipped = payload.change.skipped ? ` · ${number(payload.change.skipped)} bereits anders zugeordnet` : "";
    await acceptChange(payload, `${number(payload.change.changed)} Pixel gespeichert${skipped}`);
  } catch (error) { handleError(error); }
}

async function acceptChange(payload, message) {
  state.case = payload.case;
  updateCachedCase();
  renderCase();
  renderCases();
  await loadTile();
  setBusy(false);
  toast(message);
}

function handleError(error) {
  setSaveState("error", "Nicht gespeichert");
  state.busy = false;
  toast(error.message, true);
  loadTile().catch(() => {});
}

async function moveTile(dx, dy) {
  if (!state.case || state.busy) return;
  state.tileColumn = Math.max(0, Math.min(state.case.grid.columns - 1, state.tileColumn + dx));
  state.tileRow = Math.max(0, Math.min(state.case.grid.rows - 1, state.tileRow + dy));
  await loadTile();
}

async function moveLinear(delta) {
  if (!state.case || state.busy) return;
  const total = state.case.grid.rows * state.case.grid.columns;
  const next = Math.max(0, Math.min(total - 1, tileIndex() + delta));
  state.tileRow = Math.floor(next / state.case.grid.columns);
  state.tileColumn = next % state.case.grid.columns;
  await loadTile();
}

async function nextOpenTile() {
  if (!state.case || state.busy) return;
  const ordered = [...state.case.tiles].sort((a, b) => (a.row * state.case.grid.columns + a.column) - (b.row * state.case.grid.columns + b.column));
  const currentIndex = tileIndex();
  const next = ordered.find((tile) => tile.unassigned > 0 && tile.row * state.case.grid.columns + tile.column > currentIndex)
    || ordered.find((tile) => tile.unassigned > 0);
  if (!next) return toast("Alle vorhandenen Vordergrundpixel sind zugeordnet.");
  state.tileRow = next.row;
  state.tileColumn = next.column;
  await loadTile();
}

async function toggleComplete() {
  if (!state.case || state.busy) return;
  try {
    setBusy(true);
    state.case = await request(`/api/cases/${state.case.id}/complete`, {
      method: "POST", body: JSON.stringify({ completed: !state.case.completed }),
    });
    updateCachedCase(); renderCase(); renderCases(); setBusy(false);
  } catch (error) { handleError(error); }
}

async function exportCase() {
  if (!state.case || state.busy) return;
  try {
    setBusy(true, "Trainingsdaten werden erzeugt …");
    const validation = await request(`/api/cases/${state.case.id}/validate`);
    if (validation.warnings.length) {
      const proceed = confirm(`Es gibt noch Hinweise:\n\n${validation.warnings.join("\n")}\n\nTrotzdem exportieren?`);
      if (!proceed) { setBusy(false); return; }
    }
    const result = await request(`/api/cases/${state.case.id}/export`, {
      method: "POST", body: JSON.stringify({}),
    });
    state.case = await request(`/api/cases/${state.case.id}`);
    updateCachedCase(); renderCase(); renderCases(); setBusy(false);
    $("#result-title").textContent = `${result.valid_cells} Zellen exportiert`;
    $("#result-content").innerHTML = `<strong>${result.case_id}</strong><p>Revision ${result.revision} wurde gespeichert. Unsichere Pixel bleiben als Ignore-Maske erhalten.</p>`;
    $("#result-dialog").showModal();
  } catch (error) { handleError(error); }
}

async function renameCell() {
  const selected = state.case?.cells.find((cell) => cell.id === state.selectedCell);
  if (!selected || state.busy) return;
  const label = prompt("Name der Zelle:", selected.label);
  if (label === null) return;
  try {
    setBusy(true);
    state.case = await request(`/api/cases/${state.case.id}/cells`, {
      method: "POST", body: JSON.stringify({ action: "rename", cell_id: selected.id, label }),
    });
    updateCachedCase(); renderCase(); setBusy(false);
  } catch (error) { handleError(error); }
}

async function deleteCell() {
  const selected = state.case?.cells.find((cell) => cell.id === state.selectedCell);
  if (!selected || state.busy) return;
  if (!confirm(`${selected.label} und alle ihre Zuordnungen wirklich löschen?`)) return;
  try {
    setBusy(true);
    state.case = await request(`/api/cases/${state.case.id}/cells`, {
      method: "POST", body: JSON.stringify({ action: "delete", cell_id: selected.id }),
    });
    state.selectedCell = state.case.cells[0]?.id || null;
    updateCachedCase(); renderCase(); renderCases(); await loadTile(); setBusy(false);
  } catch (error) { handleError(error); }
}

function bindEvents() {
  $$(".tool[data-tool]").forEach((button) => {
    button.addEventListener("click", () => button.dataset.tool === "new" ? createCell() : setTool(button.dataset.tool));
  });
  $("#radius").addEventListener("input", (event) => {
    state.radius = Number(event.target.value);
    $("#radius-value").textContent = `${state.radius} px`;
  });
  $("#view-select").addEventListener("change", async (event) => {
    state.view = event.target.value;
    if (state.case) await loadTile();
  });
  $("#complete-button").addEventListener("click", toggleComplete);
  $("#export-button").addEventListener("click", exportCase);
  $("#rename-cell").addEventListener("click", renameCell);
  $("#delete-cell").addEventListener("click", deleteCell);
  $("#previous-cell-tile").addEventListener("click", () => moveLinear(-1));
  $("#next-open-tile").addEventListener("click", nextOpenTile);
  $$('[data-move="up"]').forEach((button) => button.addEventListener("click", () => moveTile(0, -1)));
  $$('[data-move="down"]').forEach((button) => button.addEventListener("click", () => moveTile(0, 1)));
  $$('[data-move="left"]').forEach((button) => button.addEventListener("click", () => moveTile(-1, 0)));
  $$('[data-move="right"]').forEach((button) => button.addEventListener("click", () => moveTile(1, 0)));

  const overview = $("#overview-image");
  overview.addEventListener("click", async (event) => {
    const rect = overview.getBoundingClientRect();
    state.tileColumn = Math.min(state.case.grid.columns - 1, Math.floor((event.clientX - rect.left) / rect.width * state.case.width / 512));
    state.tileRow = Math.min(state.case.grid.rows - 1, Math.floor((event.clientY - rect.top) / rect.height * state.case.height / 512));
    await loadTile();
  });

  const canvas = $("#draw-canvas");
  canvas.addEventListener("pointerdown", async (event) => {
    if (!state.case || state.busy) return;
    const point = canvasPoint(event);
    if (["fill", "new-soma"].includes(state.tool)) return handleCanvasClick(point);
    state.drawing = true;
    state.points = [point];
    canvas.setPointerCapture(event.pointerId);
    drawTemporary();
  });
  canvas.addEventListener("pointermove", (event) => {
    const cursor = $("#cursor");
    cursor.style.display = "block";
    cursor.style.left = `${event.clientX}px`;
    cursor.style.top = `${event.clientY}px`;
    const rect = canvas.getBoundingClientRect();
    const diameter = Math.max(4, (state.radius * 2 + 1) * rect.width / 512);
    cursor.style.width = `${diameter}px`;
    cursor.style.height = `${diameter}px`;
    if (!state.drawing) return;
    const point = canvasPoint(event);
    const last = state.points.at(-1);
    if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) >= .8) {
      state.points.push(point);
      drawTemporary();
    }
  });
  canvas.addEventListener("pointerleave", () => { $("#cursor").style.display = "none"; });
  canvas.addEventListener("pointerup", async (event) => {
    if (!state.drawing) return;
    state.drawing = false;
    canvas.releasePointerCapture(event.pointerId);
    await sendStroke();
    state.points = [];
  });

  window.addEventListener("keydown", async (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName) || $("#result-dialog").open) return;
    const key = event.key.toLowerCase();
    if (key === "n") { event.preventDefault(); await createCell(); }
    else if (key === "1") setTool("skeleton");
    else if (key === "2") setTool("soma");
    else if (key === "f") setTool("fill");
    else if (key === "v") setTool("overlap");
    else if (key === "u") setTool(event.shiftKey ? "clear-ambiguous" : "ambiguous");
    else if (key === "e") setTool("erase");
    else if (event.key === "ArrowLeft") { event.preventDefault(); await moveTile(-1, 0); }
    else if (event.key === "ArrowRight") { event.preventDefault(); await moveTile(1, 0); }
    else if (event.key === "ArrowUp") { event.preventDefault(); await moveTile(0, -1); }
    else if (event.key === "ArrowDown") { event.preventDefault(); await moveTile(0, 1); }
  });
}

async function initialize() {
  bindEvents();
  try {
    state.config = await request("/api/config");
    renderCases();
    setSaveState("", "Automatisch gespeichert");
  } catch (error) {
    setSaveState("error", "Tool konnte nicht geladen werden");
    $("#case-list").textContent = error.message;
  }
}

initialize();
