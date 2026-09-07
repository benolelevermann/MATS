const $ = (selector) => document.querySelector(selector);
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const startButton = $("#start-button");
const runPanel = $("#run-panel");
const results = $("#results");
let selectedFile = null;
let activeJob = null;
let eventTimer = null;
let currentJob = null;
let reviewFilter = "all";
let fovRecords = [];
let fovSelection = new Set();
let cropEditor = null;
let advancingFov = false;
let activeReviewFolder = null;
let savingReview = false;
let activeDatasetId = "…";
let hysteresisProfiles = [];
let activeReviewMode = "training";

const phaseOrder = ["prediction", "hysteresis", "cell_classification", "neurotreetracer", "evo_finalization"];
const phaseAliases = {
  starting: "prediction",
  queued: "prediction",
  cell_export: "neurotreetracer",
  evo_validation: "evo_finalization",
  archive: "evo_finalization",
  complete: "evo_finalization",
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

async function loadPipelineConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    const config = await response.json();
    if (!response.ok) throw new Error(config.error || "Konfiguration konnte nicht gelesen werden.");
    activeDatasetId = String(config.dataset_id ?? "…");
    activeReviewMode = config.review_mode === "evo" ? "evo" : "training";
    document.querySelectorAll("[data-network-id]").forEach((node) => {
      node.textContent = activeDatasetId;
    });
    hysteresisProfiles = Array.isArray(config.hysteresis_profiles) ? config.hysteresis_profiles : [];
    const profileSelect = $("#hysteresis-profile");
    if (profileSelect && hysteresisProfiles.length) {
      profileSelect.replaceChildren(...hysteresisProfiles.map((profile) => {
        const option = document.createElement("option");
        option.value = profile.id;
        option.textContent = profile.label;
        return option;
      }));
      profileSelect.value = config.default_hysteresis_profile || "adaptive";
      updateHysteresisHelp();
    }
    applyReviewModeCopy();
  } catch (_) {}
}

function applyReviewModeCopy() {
  const evoMode = activeReviewMode === "evo";
  document.title = evoMode
    ? `Netz ${activeDatasetId} · Evo-Zellauswahl`
    : `Netz ${activeDatasetId} · Trainingsdaten erzeugen`;
  $("#hero-title").innerHTML = evoMode
    ? "Vom TIFF zu<br><em>ausgewählten Evo-Zellen.</em>"
    : "Vom TIFF zu<br><em>neuen Trainingsdaten.</em>";
  $("#hero-copy").innerHTML = evoMode
    ? `Netz <span data-network-id>${activeDatasetId}</span> segmentiert das Übersichtsbild. Nach Hysterese und sicherer Einzelzell-Extraktion entscheidest du nur noch, welche vollständigen Zellen in die Evo-Pipeline übernommen werden.`
    : `Netz <span data-network-id>${activeDatasetId}</span> segmentiert das Übersichtsbild. Nach Hysterese und Einzelzell-Extraktion übernimmst du gute Bild-Masken-Paare in den getrennten Trainingsdatensatz.`;
  $("#dataset-inspector-link").hidden = evoMode;
  $("#new-run-kicker").textContent = evoMode ? "NEUE EVO-AUSWAHL" : "NEUE TRAININGSDATEN";
  $("#review-download-button").innerHTML = evoMode
    ? "Für Evo ausgewählte Zellen <span>↓</span>"
    : "Freigegebene Trainingszellen <span>↓</span>";
  $("#workflow-footer").innerHTML = evoMode
    ? `Dataset <b data-network-id>${activeDatasetId}</b> final · Evo-Auswahl · keine Trainingsdaten`
    : `Dataset <b data-network-id>${activeDatasetId}</b> final · Trainingsdaten-Auswahl`;
}

function selectedHysteresisProfile() {
  return $("#hysteresis-profile")?.value || "adaptive";
}

function updateHysteresisHelp() {
  const profile = hysteresisProfiles.find((item) => item.id === selectedHysteresisProfile());
  $("#hysteresis-profile-help").textContent = profile?.description || "Gewählte Hysterese-Einstellung.";
}

$("#hysteresis-profile")?.addEventListener("change", updateHysteresisHelp);

function chooseFile(file) {
  if (!file) return;
  if (!/\.tiff?$/i.test(file.name)) {
    alert("Bitte eine TIFF-Datei (.tif oder .tiff) auswählen.");
    return;
  }
  selectedFile = file;
  $("#drop-title").textContent = file.name;
  $("#drop-copy").textContent = `${formatBytes(file.size)} · bereit zum Start`;
  $("#selected-file").textContent = file.name;
  startButton.disabled = false;
}

function loadStoredFovSelection() {
  try {
    const values = JSON.parse(localStorage.getItem("mica-fov-selection") || "[]");
    if (Array.isArray(values)) fovSelection = new Set(values.filter((value) => typeof value === "string"));
  } catch (_) { fovSelection = new Set(); }
}

function persistFovSelection() {
  localStorage.setItem("mica-fov-selection", JSON.stringify([...fovSelection]));
}

function selectedFilterValue(selector) {
  return $(selector)?.value || "all";
}

function visibleFovs() {
  const div = selectedFilterValue("#fov-div-filter");
  const plate = selectedFilterValue("#fov-plate-filter");
  const well = selectedFilterValue("#fov-well-filter");
  return fovRecords.filter((record) => (
    (div === "all" || String(record.div) === div)
    && (plate === "all" || String(record.plate) === plate)
    && (well === "all" || record.well === well)
  ));
}

function updateFovSelectionState() {
  const count = fovSelection.size;
  $("#fov-selected-count").textContent = String(count);
  $("#fov-queue-title").textContent = count
    ? `${count} FOV${count === 1 ? "" : "s"} für die Analyse vorgemerkt`
    : "Noch kein FOV ausgewählt";
  $("#fov-queue-button").disabled = count === 0;
  $("#fov-expand-button").disabled = count === 0;
}

function renderFovGallery() {
  const records = visibleFovs();
  $("#fov-visible-count").textContent = `${records.length} Bilder`;
  const gallery = $("#fov-gallery");
  gallery.replaceChildren(...records.map((record) => {
    const label = document.createElement("label");
    label.className = "fov-card";
    label.classList.toggle("expanded", record.generation === "expanded");
    label.classList.toggle("selected", fovSelection.has(record.id));
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = fovSelection.has(record.id);
    input.setAttribute("aria-label", `${record.filename} auswählen`);
    input.addEventListener("change", () => {
      if (input.checked) fovSelection.add(record.id);
      else fovSelection.delete(record.id);
      label.classList.toggle("selected", input.checked);
      persistFovSelection();
      updateFovSelectionState();
    });
    const image = document.createElement("img");
    image.src = record.preview_url;
    image.alt = `Vorschau ${record.filename}`;
    image.loading = "lazy";
    const check = document.createElement("span");
    check.className = "fov-check";
    check.textContent = "✓";
    const meta = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `DIV${String(record.div).padStart(2, "0")} · Platte ${record.plate} · ${record.well} · FOV ${String(record.fov).padStart(2, "0")}`;
    const details = document.createElement("small");
    details.textContent = `x ${record.x} · y ${record.y}${record.std == null ? "" : ` · Kontrast ${Number(record.std).toFixed(1)}`}`;
    meta.append(title, details);
    label.append(input, image, check, meta);
    return label;
  }));
  updateFovSelectionState();
}

function fillFovFilter(selector, values, formatter = (value) => value) {
  const select = $(selector);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = formatter(value);
    select.append(option);
  });
}

function installFovLibrary(payload, resetFilters = false) {
    if (!payload?.available) return;
    fovRecords = payload.fovs || [];
    const validIds = new Set(fovRecords.map((record) => record.id));
    fovSelection = new Set([...fovSelection].filter((identifier) => validIds.has(identifier)));
    persistFovSelection();
    if (resetFilters) {
      ["#fov-div-filter", "#fov-plate-filter", "#fov-well-filter"].forEach((selector) => {
        const select = $(selector);
        select.replaceChildren(select.querySelector('option[value="all"]') || new Option("Alle", "all"));
      });
      fillFovFilter("#fov-div-filter", [...new Set(fovRecords.map((item) => item.div))].sort((a, b) => a - b), (value) => `DIV${String(value).padStart(2, "0")}`);
      fillFovFilter("#fov-plate-filter", [...new Set(fovRecords.map((item) => item.plate))].sort((a, b) => a - b), (value) => `Platte ${value}`);
      fillFovFilter("#fov-well-filter", [...new Set(fovRecords.map((item) => item.well))].sort());
    }
    $("#fov-library").hidden = false;
    renderFovGallery();
}

async function loadFovLibrary() {
  loadStoredFovSelection();
  try {
    const response = await fetch("/api/fovs", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || !payload.available) return;
    installFovLibrary(payload, true);
  } catch (_) {}
}

["#fov-div-filter", "#fov-plate-filter", "#fov-well-filter"].forEach((selector) => {
  $(selector).addEventListener("change", renderFovGallery);
});
$("#fov-select-visible").addEventListener("click", () => {
  visibleFovs().forEach((record) => fovSelection.add(record.id));
  persistFovSelection();
  renderFovGallery();
});
$("#fov-clear-selection").addEventListener("click", () => {
  fovSelection.clear();
  persistFovSelection();
  renderFovGallery();
});
$("#fov-expand-button").addEventListener("click", async () => {
  if (!fovSelection.size) return;
  const button = $("#fov-expand-button");
  const selectedCount = fovSelection.size;
  button.disabled = true;
  button.textContent = "Neue FOVs werden ausgeschnitten …";
  try {
    const response = await fetch("/api/fovs/expand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fov_ids: [...fovSelection], count_per_overview: 6 }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const restartHint = response.status === 404
        ? " Der lokale Server läuft noch mit der alten Version; bitte ihn nach Abschluss laufender Analysen einmal neu starten."
        : "";
      throw new Error((payload.error || "Zusätzliche FOVs konnten nicht erzeugt werden.") + restartHint);
    }
    fovSelection.clear();
    persistFovSelection();
    installFovLibrary(payload.library, false);
    alert(`${payload.created} neue FOVs aus ${payload.overview_count} Übersichtsbild${payload.overview_count === 1 ? "" : "ern"} erzeugt. Sie sind mit „NEU“ markiert und können jetzt als Netz-Eingaben ausgewählt werden.`);
  } catch (error) {
    alert(error.message);
  } finally {
    button.innerHTML = "6 weitere je Übersicht <span>＋</span>";
    updateFovSelectionState();
  }
});
$("#fov-queue-button").addEventListener("click", async () => {
  if (!fovSelection.size) return;
  const button = $("#fov-queue-button");
  button.disabled = true;
  button.textContent = "Wird vorbereitet …";
  try {
    const response = await fetch("/api/fov-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fov_ids: [...fovSelection],
        hysteresis_profile: selectedHysteresisProfile(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "FOV-Auswahl konnte nicht gestartet werden.");
    const jobs = payload.jobs || [];
    localStorage.setItem("mica-fov-batch-jobs", JSON.stringify(jobs.map((job) => job.id)));
    fovSelection.clear();
    persistFovSelection();
    renderFovGallery();
    if (jobs.length) {
      activeJob = jobs[0].id;
      renderJob(jobs[0]);
      watchJob(jobs[0]);
      runPanel.hidden = false;
      runPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      await loadHistory();
      if (jobs.length > 1) alert(`${jobs.length} FOVs wurden eingereiht und werden nacheinander verarbeitet.`);
    }
  } catch (error) {
    alert(error.message);
  } finally {
    button.innerHTML = "Auswahl analysieren <span>→</span>";
    updateFovSelectionState();
  }
});

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add("is-dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-dragging");
}));
dropzone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

function uploadJob(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    request.setRequestHeader("X-Hysteresis-Profile", encodeURIComponent(selectedHysteresisProfile()));
    request.setRequestHeader("Content-Type", "image/tiff");
    $("#upload-progress").hidden = false;
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) $("#upload-progress-bar").style.width = `${100 * event.loaded / event.total}%`;
    });
    request.addEventListener("load", () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText); } catch (_) {}
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.error || `Upload fehlgeschlagen (${request.status})`));
    });
    request.addEventListener("error", () => reject(new Error("Der lokale Server ist nicht erreichbar.")));
    request.send(file);
  });
}

startButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  startButton.disabled = true;
  startButton.textContent = "Wird hochgeladen …";
  results.hidden = true;
  try {
    const job = await uploadJob(selectedFile);
    activeJob = job.id;
    renderJob(job);
    runPanel.hidden = false;
    runPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    watchJob(job);
    loadHistory();
  } catch (error) {
    alert(error.message);
  } finally {
    startButton.disabled = false;
    startButton.innerHTML = "Analyse starten <span>→</span>";
  }
});

function statusLabel(status) {
  return ({ queued: "Warteschlange", running: "Läuft", completed: "Fertig", failed: "Fehlgeschlagen", cancelled: "Abgebrochen" })[status] || status;
}

function renderStages(job) {
  const canonical = phaseAliases[job.phase] || job.phase;
  const activeIndex = phaseOrder.indexOf(canonical);
  document.querySelectorAll("#stage-grid > div").forEach((element, index) => {
    element.classList.toggle("active", job.status === "running" && index === activeIndex);
    element.classList.toggle("done", job.status === "completed" || (activeIndex >= 0 && index < activeIndex));
  });
}

function renderEvents(events = []) {
  const log = $("#event-log");
  log.replaceChildren(...events.slice(-30).reverse().map((event) => {
    const item = document.createElement("li");
    const time = new Date(event.time).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    item.textContent = `${time}  ${event.message}`;
    return item;
  }));
}

function renderJob(job) {
  runPanel.hidden = false;
  $("#run-title").textContent = job.filename || job.case || "Verarbeitung";
  const pill = $("#status-pill");
  pill.textContent = statusLabel(job.status);
  pill.classList.toggle("failed", job.status === "failed");
  pill.classList.toggle("cancelled", job.status === "cancelled");
  $("#job-progress-bar").style.width = `${job.progress || 0}%`;
  $("#progress-percent").textContent = `${job.progress || 0}%`;
  $("#progress-message").textContent = job.message || "";
  $("#active-hysteresis-label").textContent = job.hysteresis?.label || "gewählte Schwelle";
  renderStages(job);
  renderEvents(job.events);
  if (job.status === "completed") renderResults(job);
  if (job.status === "failed" || job.status === "cancelled") results.hidden = true;
}

async function watchJob(job) {
  if (eventTimer) clearTimeout(eventTimer);
  if (!activeJob || job.status === "completed" || job.status === "failed" || job.status === "cancelled") return;
  try {
    const response = await fetch(`/api/jobs/${activeJob}/events?version=${job.version || 0}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Status konnte nicht gelesen werden.");
    const text = await response.text();
    const dataLine = text.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) throw new Error("Ungültige Statusantwort.");
    const updated = JSON.parse(dataLine.slice(6));
    renderJob(updated);
    watchJob(updated);
  } catch (_) {
    eventTimer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/jobs/${activeJob}`, { cache: "no-store" });
        const updated = await response.json();
        renderJob(updated);
        watchJob(updated);
      } catch (_) { watchJob(job); }
    }, 1700);
  }
}

function cropReviewState(crop) {
  const review = crop.review || {};
  const result = review.result
    || (review.training && review.postprocessing
      ? (review.training === "good" && review.postprocessing === "good" ? "good" : "bad")
      : null);
  if (!result) return "pending";
  return result === "good" ? "approved" : "rejected";
}

function reviewSummary(crops) {
  const complete = crops.filter((crop) => cropReviewState(crop) !== "pending").length;
  const approved = crops.filter((crop) => cropReviewState(crop) === "approved").length;
  return { complete, approved, total: crops.length };
}

function updateReviewOverview(crops) {
  const summary = reviewSummary(crops);
  $("#review-progress").textContent = `${summary.complete} / ${summary.total} geprüft`;
  $("#review-progress-bar").style.width = `${summary.total ? 100 * summary.complete / summary.total : 0}%`;
  $("#review-approved").textContent = activeReviewMode === "evo"
    ? `${summary.approved} für Evo übernommen`
    : `${summary.approved} für Training freigegeben`;
}

function comparisonTile(label, url, crop) {
  const figure = document.createElement("figure");
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  const image = document.createElement("img");
  image.src = url;
  image.alt = `${label} für ${crop.folder}`;
  image.loading = "lazy";
  const caption = document.createElement("figcaption");
  caption.textContent = label;
  link.append(image);
  figure.append(link, caption);
  return figure;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function cropSelectionValid(editor) {
  const selection = editor.selection;
  const bounds = editor.info.label_bounds;
  const clearance = editor.info.clearance || 0;
  return selection.width >= editor.info.minimum_side
    && selection.height >= editor.info.minimum_side
    && selection.x >= 0
    && selection.y >= 0
    && selection.x + selection.width <= editor.info.source_width
    && selection.y + selection.height <= editor.info.source_height
    && selection.x <= bounds.x_min - clearance
    && selection.y <= bounds.y_min - clearance
    && selection.x + selection.width >= bounds.x_max_exclusive + clearance
    && selection.y + selection.height >= bounds.y_max_exclusive + clearance;
}

function syncCropFields() {
  if (!cropEditor) return;
  const selection = cropEditor.selection;
  [["#crop-x", "x"], ["#crop-y", "y"], ["#crop-width", "width"], ["#crop-height", "height"]].forEach(([selector, key]) => {
    $(selector).value = Math.round(selection[key]);
  });
  const valid = cropSelectionValid(cropEditor);
  $("#crop-validation").textContent = valid
    ? "Alle Zellpixel liegen sicher im Crop."
    : `Der Rahmen muss alle Zellpixel mit ${cropEditor.info.clearance} px Abstand enthalten.`;
  $("#crop-validation").classList.toggle("invalid", !valid);
  $("#crop-save").disabled = !valid;
}

function drawCropEditor() {
  if (!cropEditor) return;
  const { canvas, context, image, info, selection } = cropEditor;
  selection.width = Math.round(clamp(selection.width, info.minimum_side, info.source_width));
  selection.height = Math.round(clamp(selection.height, info.minimum_side, info.source_height));
  selection.x = Math.round(clamp(selection.x, 0, info.source_width - selection.width));
  selection.y = Math.round(clamp(selection.y, 0, info.source_height - selection.height));
  const scaleX = canvas.width / info.source_width;
  const scaleY = canvas.height / info.source_height;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  const x = selection.x * scaleX;
  const y = selection.y * scaleY;
  const width = selection.width * scaleX;
  const height = selection.height * scaleY;
  context.fillStyle = "rgba(5, 13, 10, .58)";
  context.fillRect(0, 0, canvas.width, y);
  context.fillRect(0, y + height, canvas.width, canvas.height - y - height);
  context.fillRect(0, y, x, height);
  context.fillRect(x + width, y, canvas.width - x - width, height);

  const label = info.label_bounds;
  context.save();
  context.strokeStyle = "#72e0bd";
  context.lineWidth = 2;
  context.setLineDash([7, 5]);
  context.strokeRect(
    (label.x_min - info.clearance) * scaleX,
    (label.y_min - info.clearance) * scaleY,
    (label.x_max_exclusive - label.x_min + 2 * info.clearance) * scaleX,
    (label.y_max_exclusive - label.y_min + 2 * info.clearance) * scaleY,
  );
  context.restore();
  context.strokeStyle = "#ffcf57";
  context.lineWidth = 2.5;
  context.strokeRect(x, y, width, height);
  [[x, y], [x + width, y], [x, y + height], [x + width, y + height]].forEach(([handleX, handleY]) => {
    context.fillStyle = "#ffcf57";
    context.fillRect(handleX - 6, handleY - 6, 12, 12);
    context.strokeStyle = "#18201d";
    context.lineWidth = 1;
    context.strokeRect(handleX - 6, handleY - 6, 12, 12);
  });
  syncCropFields();
}

function cropPointerPosition(event, editor) {
  const rectangle = editor.canvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rectangle.left) * editor.info.source_width / rectangle.width, 0, editor.info.source_width),
    y: clamp((event.clientY - rectangle.top) * editor.info.source_height / rectangle.height, 0, editor.info.source_height),
  };
}

function cropDragMode(position, editor) {
  const selection = editor.selection;
  const threshold = 14 * editor.info.source_width / editor.canvas.getBoundingClientRect().width;
  const corners = {
    nw: [selection.x, selection.y],
    ne: [selection.x + selection.width, selection.y],
    sw: [selection.x, selection.y + selection.height],
    se: [selection.x + selection.width, selection.y + selection.height],
  };
  for (const [mode, [x, y]] of Object.entries(corners)) {
    if (Math.hypot(position.x - x, position.y - y) <= threshold) return mode;
  }
  if (
    position.x >= selection.x && position.x <= selection.x + selection.width
    && position.y >= selection.y && position.y <= selection.y + selection.height
  ) return "move";
  return null;
}

function updateCropFromPointer(position) {
  if (!cropEditor?.drag) return;
  const editor = cropEditor;
  const { mode, start, initial } = editor.drag;
  const minimum = editor.info.minimum_side;
  const maxX = editor.info.source_width;
  const maxY = editor.info.source_height;
  let left = initial.x;
  let top = initial.y;
  let right = initial.x + initial.width;
  let bottom = initial.y + initial.height;
  if (mode === "move") {
    const dx = position.x - start.x;
    const dy = position.y - start.y;
    left = clamp(initial.x + dx, 0, maxX - initial.width);
    top = clamp(initial.y + dy, 0, maxY - initial.height);
    right = left + initial.width;
    bottom = top + initial.height;
  } else {
    if (mode.includes("w")) left = clamp(position.x, 0, right - minimum);
    if (mode.includes("e")) right = clamp(position.x, left + minimum, maxX);
    if (mode.includes("n")) top = clamp(position.y, 0, bottom - minimum);
    if (mode.includes("s")) bottom = clamp(position.y, top + minimum, maxY);
  }
  editor.selection = {
    x: Math.round(left),
    y: Math.round(top),
    width: Math.round(right - left),
    height: Math.round(bottom - top),
  };
  drawCropEditor();
}

async function openCropEditor(crop) {
  if (!activeJob) return;
  try {
    const response = await fetch(`/api/jobs/${activeJob}/crops/${crop.folder}`, { cache: "no-store" });
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || "Crop-Informationen konnten nicht geladen werden.");
    const image = new Image();
    image.src = `${info.preview_url}?v=${Date.now()}`;
    await image.decode();
    const canvas = $("#crop-canvas");
    const maximum = 760;
    const scale = Math.min(1, maximum / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    cropEditor = {
      crop,
      info,
      image,
      canvas,
      context: canvas.getContext("2d"),
      selection: { ...info.selection },
      drag: null,
    };
    $("#crop-dialog-title").textContent = `${crop.folder} zuschneiden`;
    drawCropEditor();
    $("#crop-dialog").showModal();
  } catch (error) {
    alert(error.message);
  }
}

$("#crop-canvas").addEventListener("pointerdown", (event) => {
  if (!cropEditor) return;
  const position = cropPointerPosition(event, cropEditor);
  const mode = cropDragMode(position, cropEditor);
  if (!mode) return;
  cropEditor.drag = { mode, start: position, initial: { ...cropEditor.selection } };
  cropEditor.canvas.setPointerCapture(event.pointerId);
});
$("#crop-canvas").addEventListener("pointermove", (event) => {
  if (!cropEditor?.drag) return;
  updateCropFromPointer(cropPointerPosition(event, cropEditor));
});
["pointerup", "pointercancel"].forEach((eventName) => $("#crop-canvas").addEventListener(eventName, () => {
  if (cropEditor) cropEditor.drag = null;
}));

[["#crop-x", "x"], ["#crop-y", "y"], ["#crop-width", "width"], ["#crop-height", "height"]].forEach(([selector, key]) => {
  $(selector).addEventListener("change", () => {
    if (!cropEditor) return;
    const value = Number.parseInt($(selector).value, 10);
    if (Number.isFinite(value)) cropEditor.selection[key] = value;
    drawCropEditor();
  });
});
$("#crop-reset").addEventListener("click", () => {
  if (!cropEditor) return;
  cropEditor.selection = {
    x: 0,
    y: 0,
    width: cropEditor.info.source_width,
    height: cropEditor.info.source_height,
  };
  drawCropEditor();
});
$("#crop-save").addEventListener("click", async () => {
  if (!cropEditor || !activeJob || !cropSelectionValid(cropEditor)) return;
  const button = $("#crop-save");
  button.disabled = true;
  button.textContent = "Wird gespeichert …";
  try {
    const response = await fetch(`/api/jobs/${activeJob}/crops/${cropEditor.crop.folder}/recrop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cropEditor.selection),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Crop konnte nicht gespeichert werden.");
    const index = currentJob.crops.findIndex((item) => item.folder === cropEditor.crop.folder);
    if (index >= 0) currentJob.crops[index] = payload.crop;
    $("#crop-dialog").close();
    cropEditor = null;
    renderCropGallery();
  } catch (error) {
    alert(error.message);
    button.disabled = false;
  } finally {
    button.innerHTML = "Crop speichern <span>✓</span>";
  }
});
$("#crop-dialog").addEventListener("close", () => { cropEditor = null; });

function renderCropGallery() {
  const crops = currentJob?.crops || [];
  const visible = crops.filter((crop) => reviewFilter === "all" || cropReviewState(crop) === reviewFilter);
  const gallery = $("#crop-gallery");
  let crop = visible.find((item) => item.folder === activeReviewFolder);
  if (!crop) {
    crop = visible.find((item) => cropReviewState(item) === "pending") || visible[0];
    activeReviewFolder = crop?.folder || null;
  }
  if (!crop) {
    $("#gallery-count").textContent = `0 von ${crops.length} Zellen`;
    const empty = document.createElement("div");
    empty.className = "review-empty";
    empty.textContent = reviewFilter === "pending"
      ? "Alle Zellen dieses FOVs sind bewertet."
      : "Keine Zelle passt zu diesem Filter.";
    gallery.replaceChildren(empty);
    updateReviewOverview(crops);
    return;
  }

  const visibleIndex = visible.findIndex((item) => item.folder === crop.folder);
  $("#gallery-count").textContent = `Zelle ${visibleIndex + 1} von ${visible.length} · insgesamt ${crops.length}`;
  const state = cropReviewState(crop);
  const card = document.createElement("article");
  card.className = `crop-card review-${state}`;
  card.dataset.folder = crop.folder;
  card.dataset.reviewState = state;

  const meta = document.createElement("div");
  meta.className = "crop-meta";
  const name = document.createElement("strong");
  name.textContent = crop.display_name || crop.case_id || crop.folder;
  const tags = document.createElement("div");
  tags.className = "crop-tags";
  const tag = document.createElement("span");
  const manualTrainingReview = currentJob?.review_kind === "manual_training_data";
  tag.className = `source-tag ${(crop.source === "neurotreetracer" || manualTrainingReview) ? "traced" : ""}`;
  tag.textContent = manualTrainingReview
    ? (crop.source_collection_label || "manuelles Training")
    : crop.source === "neurotreetracer" ? "getract" : "isoliert";
  const stateTag = document.createElement("span");
  stateTag.className = `review-state ${state}`;
  stateTag.textContent = ({ approved: "freigegeben", rejected: "abgelehnt", pending: "offen" })[state];
  tags.append(tag, stateTag);
  meta.append(name, tags);

  const comparison = document.createElement("div");
  comparison.className = "comparison-grid";
  if (manualTrainingReview) {
    comparison.append(
      comparisonTile("Rohbild", crop.raw_preview_url || crop.preview_url, crop),
      comparisonTile("Manuelle Annotation", crop.prediction_preview_url || crop.preview_url, crop),
      comparisonTile("Trainings-Crop", crop.preview_url, crop),
    );
  } else {
    comparison.append(
      comparisonTile(`Netz ${activeDatasetId}`, crop.prediction_preview_url || crop.preview_url, crop),
      comparisonTile("Hysterese", crop.postprocessing_preview_url || crop.preview_url, crop),
      comparisonTile("Einzelzelle", crop.preview_url, crop),
    );
  }

  const details = document.createElement("div");
  details.className = "crop-details";
  const detailText = document.createElement("span");
  const cropSize = crop.manual_crop
    ? ` · Crop ${crop.manual_crop.width}×${crop.manual_crop.height} px`
    : crop.source_width && crop.source_height
      ? ` · Auto-Crop ${crop.source_width}×${crop.source_height} px`
      : "";
  detailText.textContent = `${crop.skeleton_pixels.toLocaleString("de-DE")} Skelettpixel · ${crop.soma_pixels.toLocaleString("de-DE")} Somapixel${cropSize}`;
  const cropButton = document.createElement("button");
  cropButton.type = "button";
  cropButton.className = "crop-edit-button";
  cropButton.textContent = crop.manual_crop ? "Crop erneut anpassen" : "Crop anpassen";
  cropButton.addEventListener("click", () => openCropEditor(crop));
  details.append(detailText);
  if (activeReviewMode !== "evo") details.append(cropButton);

  const controls = document.createElement("div");
  controls.className = "review-controls";
  const reject = document.createElement("button");
  reject.type = "button";
  reject.className = "review-decision bad";
  reject.classList.toggle("selected", state === "rejected");
  reject.innerHTML = activeReviewMode === "evo"
    ? "<strong>←</strong> Nicht für Evo"
    : "<strong>←</strong> Nicht gut";
  reject.addEventListener("click", () => saveReview(crop, "bad", card));
  const accept = document.createElement("button");
  accept.type = "button";
  accept.className = "review-decision good";
  accept.classList.toggle("selected", state === "approved");
  accept.innerHTML = activeReviewMode === "evo"
    ? "Für Evo übernehmen <strong>→</strong>"
    : "Als Training übernehmen <strong>→</strong>";
  accept.addEventListener("click", () => saveReview(crop, "good", card));
  controls.append(reject, accept);

  const saveState = document.createElement("div");
  saveState.className = "review-save-state";
  saveState.textContent = state === "approved"
    ? (activeReviewMode === "evo" ? "Als vollständiger Evo-Zellordner gespeichert" : "Als Trainingspaar gespeichert")
    : state === "rejected"
      ? (activeReviewMode === "evo" ? "Nicht in die Evo-Auswahl übernommen" : "Nicht in den Trainingsdatensatz übernommen")
      : (activeReviewMode === "evo" ? "Einmal entscheiden: ← verwerfen · → für Evo" : "Einmal entscheiden: ← nicht gut · → Training");
  card.append(meta, comparison, details, controls, saveState);
  gallery.replaceChildren(card);
  updateReviewOverview(crops);

  const nextPending = crops.find((item) => item.folder !== crop.folder && cropReviewState(item) === "pending");
  if (nextPending) {
    [nextPending.raw_preview_url, nextPending.prediction_preview_url, nextPending.postprocessing_preview_url, nextPending.preview_url]
      .filter(Boolean)
      .forEach((url) => { const image = new Image(); image.src = url; });
  }
}

async function saveReview(crop, value, card) {
  if (!activeJob || savingReview || card?.classList.contains("saving")) return;
  savingReview = true;
  card?.classList.add("saving");
  try {
    const response = await fetch(`/api/jobs/${activeJob}/reviews/${crop.folder}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Bewertung konnte nicht gespeichert werden.");
    crop.review = payload.review;
    if (currentJob) currentJob.review_summary = payload.review_summary;
    const crops = currentJob?.crops || [];
    const currentIndex = crops.findIndex((item) => item.folder === crop.folder);
    const ordered = currentIndex >= 0
      ? [...crops.slice(currentIndex + 1), ...crops.slice(0, currentIndex)]
      : crops;
    activeReviewFolder = ordered.find((item) => cropReviewState(item) === "pending")?.folder || crop.folder;
    renderCropGallery();
    if (payload.review_summary.total > 0 && payload.review_summary.pending === 0) {
      window.setTimeout(() => advanceToNextFov(true), 350);
    }
  } catch (error) {
    card?.classList.remove("saving");
    alert(error.message);
  } finally {
    savingReview = false;
  }
}

function storedBatchJobIds() {
  try {
    const values = JSON.parse(localStorage.getItem("mica-fov-batch-jobs") || "[]");
    return Array.isArray(values) ? values.filter((value) => typeof value === "string") : [];
  } catch (_) { return []; }
}

async function advanceToNextFov(automatic = false) {
  if (advancingFov) return;
  advancingFov = true;
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Jobliste konnte nicht gelesen werden.");
    let jobs = (payload.jobs || []).filter((job) => job.source?.kind === "fov_library");
    const batchId = currentJob?.source?.batch_id;
    if (batchId) {
      jobs = jobs.filter((job) => job.source?.batch_id === batchId);
    } else {
      const stored = new Set(storedBatchJobIds());
      if (stored.size && stored.has(activeJob)) jobs = jobs.filter((job) => stored.has(job.id));
    }
    jobs.sort((left, right) => new Date(left.created_at) - new Date(right.created_at));
    const currentIndex = jobs.findIndex((job) => job.id === activeJob);
    const ordered = currentIndex >= 0
      ? [...jobs.slice(currentIndex + 1), ...jobs.slice(0, currentIndex)]
      : jobs;
    const completed = ordered.find((job) => (
      job.status === "completed"
      && (job.review_summary?.total || 0) > 0
      && (job.review_summary?.pending ?? job.review_summary?.total ?? 0) > 0
    ));
    const inProgress = ordered.find((job) => job.status === "running");
    const queued = ordered.find((job) => job.status === "queued");
    const next = completed || inProgress || queued;
    if (!next) {
      if (!automatic) alert("Alle verfügbaren FOVs dieses Laufs sind bereits geprüft.");
      return;
    }
    await loadJob(next.id);
  } catch (error) {
    if (!automatic) alert(error.message);
  } finally {
    advancingFov = false;
  }
}

function renderResults(job) {
  if (currentJob?.id !== job.id) activeReviewFolder = null;
  currentJob = job;
  const summary = job.summary || {};
  const crops = job.crops || [];
  results.hidden = false;
  const manualTrainingReview = job.review_kind === "manual_training_data";
  const evoMode = (job.review_mode || activeReviewMode) === "evo";
  $("#stat-exported").textContent = summary.exported_cell_count ?? crops.length;
  $("#stat-isolated").textContent = manualTrainingReview
    ? (job.source?.collections?.length || 0)
    : (summary.isolated_exported ?? 0);
  $("#stat-traced").textContent = summary.neurotreetracer_cells_exported ?? 0;
  $("#stat-rejected").textContent = summary.neurotreetracer_groups_rejected ?? 0;
  const statCopy = manualTrainingReview
    ? [
        ["#stat-exported-label", "IMPORTIERT"], ["#stat-exported-copy", "vollständige Zellen"],
        ["#stat-isolated-label", "QUELLEN"], ["#stat-isolated-copy", "Tracing-Sammlungen"],
        ["#stat-traced-label", "ANNOTIERT"], ["#stat-traced-copy", "manuelle Labels"],
        ["#stat-rejected-label", "AUSGELASSEN"], ["#stat-rejected-copy", "unvollständige Ordner"],
      ]
    : [
        ["#stat-exported-label", "EXPORTIERT"], ["#stat-exported-copy", "Einzelzellen"],
        ["#stat-isolated-label", "DIREKT"], ["#stat-isolated-copy", "isolierte Zellen"],
        ["#stat-traced-label", "GETRACED"], ["#stat-traced-copy", "aus Konflikten"],
        ["#stat-rejected-label", "VERWORFEN"], ["#stat-rejected-copy", "Konfliktgruppen"],
      ];
  statCopy.forEach(([selector, value]) => { $(selector).textContent = value; });
  $("#result-copy").textContent = manualTrainingReview
    ? `${crops.length} manuell getracte Trainings-Crops. Schneide eine zweite Zelle bei Bedarf heraus und entscheide dann einmal: links verwerfen, rechts übernehmen.`
    : evoMode
      ? `${crops.length} sicher extrahierte Zell-Crops. Übernimm nur vollständige Zellen; verworfene Zellen gelangen nicht in den Evo-Ordner.`
      : `${crops.length} Masken mit jeweils genau einem Soma. Entscheide pro Zelle, ob das Bild-Masken-Paar als Trainingsdatum übernommen wird.`;
  $("#review-heading").textContent = manualTrainingReview
    ? "Eine Zielzelle pro Crop behalten."
    : evoMode ? "Nur vollständige Evo-Zellen übernehmen." : "Neue Trainingsdaten auswählen.";
  $("#review-instructions").innerHTML = manualTrainingReview
    ? "Prüfe Rohbild und manuelle Annotation. Enthält der Crop eine zweite Zelle, nutze <strong>Crop anpassen</strong>. Danach: <strong>→ übernehmen</strong> oder <strong>← verwerfen</strong>."
    : evoMode
      ? `Du siehst Netz ${activeDatasetId}, Hysterese und die isolierte Einzelzelle. Drücke <strong>→ für Evo übernehmen</strong> oder <strong>← verwerfen</strong>; danach erscheint sofort die nächste Zelle.`
      : `Du siehst Netz ${activeDatasetId}, Hysterese und die isolierte Einzelzelle. Drücke <strong>→ als Training übernehmen</strong> oder <strong>← verwerfen</strong>.`;
  $("#download-button").hidden = evoMode || !job.download_url;
  $("#download-button").href = job.download_url || "#";
  $("#next-ready-run").hidden = manualTrainingReview;
  renderCropGallery();
}

async function loadJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Der angeforderte Review-Lauf wurde nicht gefunden.");
  const job = await response.json();
  activeJob = job.id;
  const url = new URL(window.location.href);
  url.searchParams.set("job", job.id);
  window.history.replaceState({}, "", url);
  renderJob(job);
  runPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  watchJob(job);
}

async function loadHistory() {
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    const payload = await response.json();
    const list = $("#history-list");
    if (!payload.jobs.length) {
      list.innerHTML = '<p class="empty-state">Noch keine Läufe vorhanden.</p>';
      return;
    }
    list.replaceChildren(...payload.jobs.map((job) => {
      const row = document.createElement("div");
      row.className = "history-row";
      row.tabIndex = 0;
      const title = document.createElement("div");
      const titleName = document.createElement("strong");
      titleName.textContent = job.filename || job.case || "Unbenannter Lauf";
      const titleTime = document.createElement("small");
      titleTime.textContent = new Date(job.created_at).toLocaleString("de-DE");
      title.append(titleName, titleTime);
      const state = document.createElement("span");
      state.className = `history-state ${job.status === "failed" ? "failed" : job.status === "cancelled" ? "cancelled" : ""}`;
      state.textContent = statusLabel(job.status);
      const progress = document.createElement("span");
      progress.className = "history-progress";
      progress.textContent = `${job.progress || 0}%`;
      const arrow = document.createElement("span");
      arrow.textContent = "→";
      row.append(title, state, progress, arrow);
      row.addEventListener("click", () => loadJob(job.id));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter") loadJob(job.id); });
      return row;
    }));
  } catch (_) {}
}

$("#refresh-history").addEventListener("click", loadHistory);
document.querySelectorAll("#review-filters button").forEach((button) => {
  button.addEventListener("click", () => {
    reviewFilter = button.dataset.filter || "all";
    activeReviewFolder = null;
    document.querySelectorAll("#review-filters button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderCropGallery();
  });
});
$("#next-pending").addEventListener("click", () => {
  const pending = (currentJob?.crops || []).find((crop) => cropReviewState(crop) === "pending");
  if (!pending) return;
  if (reviewFilter !== "all" && reviewFilter !== "pending") {
    reviewFilter = "pending";
    document.querySelectorAll("#review-filters button").forEach((button) => {
      button.classList.toggle("active", button.dataset.filter === "pending");
    });
  }
  activeReviewFolder = pending.folder;
  renderCropGallery();
  requestAnimationFrame(() => $("#crop-gallery")?.scrollIntoView({ behavior: "smooth", block: "center" }));
});
$("#next-ready-run").addEventListener("click", () => advanceToNextFov(false));
document.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
  if (!currentJob || currentJob.status !== "completed" || results.hidden || cropEditor || savingReview) return;
  const target = event.target;
  if (target instanceof HTMLElement && (
    target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  )) return;
  const crop = (currentJob.crops || []).find((item) => item.folder === activeReviewFolder);
  if (!crop) return;
  event.preventDefault();
  saveReview(crop, event.key === "ArrowRight" ? "good" : "bad", document.querySelector(".crop-card"));
});
async function initializePage() {
  await Promise.all([loadPipelineConfig(), loadFovLibrary(), loadHistory()]);
  const requestedJob = new URLSearchParams(window.location.search).get("job");
  if (!requestedJob) return;
  try {
    await loadJob(requestedJob);
    requestAnimationFrame(() => results.scrollIntoView({ behavior: "smooth", block: "start" }));
  } catch (error) {
    alert(error.message);
  }
}

initializePage();
