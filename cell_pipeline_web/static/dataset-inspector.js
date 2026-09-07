const state = {
  payload: null,
  filtered: [],
  activeId: null,
  seed: 1,
  visibleLimit: 150,
};

const $ = (selector) => document.querySelector(selector);
const number = new Intl.NumberFormat("de-DE");
const decimal = new Intl.NumberFormat("de-DE", { maximumFractionDigits: 1 });

function pct(value) {
  return `${decimal.format(Number(value) || 0)} %`;
}

function shape(value) {
  return Array.isArray(value) ? value.join(" × ") : "–";
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function showError(message) {
  $("#loading-banner").hidden = true;
  const banner = $("#error-banner");
  banner.hidden = false;
  banner.textContent = message;
}

function createBarRow(item, maximum) {
  const row = document.createElement("div");
  row.className = "bar-row";
  const label = document.createElement("span");
  label.textContent = item.label;
  const track = document.createElement("div");
  track.className = "bar-track";
  const fill = document.createElement("i");
  fill.style.width = `${maximum ? (100 * item.count) / maximum : 0}%`;
  track.append(fill);
  const count = document.createElement("b");
  count.textContent = number.format(item.count);
  row.append(label, track, count);
  return row;
}

function renderBars(selector, items) {
  const container = $(selector);
  container.replaceChildren();
  const maximum = Math.max(1, ...items.map((item) => item.count));
  items.forEach((item) => container.append(createBarRow(item, maximum)));
}

function renderSummary() {
  const { summary } = state.payload;
  setText("#summary-total", number.format(summary.total_cases));
  setText("#summary-split", `${number.format(summary.training_cases)} Training · ${number.format(summary.validation_cases)} Validierung`);
  setText("#summary-size", `${decimal.format(summary.width.median)} × ${decimal.format(summary.height.median)} px`);
  setText("#summary-range", `Breite ${summary.width.min}–${summary.width.max} · Höhe ${summary.height.min}–${summary.height.max}`);
  setText("#summary-padded", number.format(summary.padding.cases_with_padding));
  setText("#summary-padding-mean", `im Mittel ${pct(summary.padding.mean_percent)} Hintergrund`);
  setText("#summary-empty", `${number.format(summary.skeleton_pixels.empty_cases)} / ${number.format(summary.soma_pixels.empty_cases)}`);
  renderBars("#size-distribution", summary.size_distribution);
  renderBars("#padding-distribution", summary.padding_distribution);
}

function humanNormalization(value) {
  const values = Array.isArray(value) ? value : [value];
  return values.map((item) => item === "ZScoreNormalization" ? "bildweiser Z-Score" : item).join(", ");
}

function renderPreprocessing() {
  const { preprocessing } = state.payload;
  const flow = $("#pipeline-flow");
  flow.replaceChildren();
  preprocessing.steps.forEach((step, index) => {
    const item = document.createElement("li");
    const indexLabel = document.createElement("b");
    indexLabel.textContent = String(index + 1).padStart(2, "0");
    const copy = document.createElement("span");
    copy.textContent = step;
    item.append(indexLabel, copy);
    flow.append(item);
  });

  const details = $("#preprocessing-details");
  const rows = [
    ["Konfiguration", preprocessing.configuration],
    ["Ziel-Pixelabstand", shape(preprocessing.target_spacing)],
    ["Normalisierung", humanNormalization(preprocessing.normalization)],
    ["Maskierte Normalisierung", preprocessing.use_mask_for_normalization.some(Boolean) ? "ja" : "nein"],
    ["Netz-Patch", `${shape(preprocessing.patch_size)} px`],
    ["Batch-Größe", preprocessing.batch_size],
    ["Trainer", preprocessing.training_trainer],
    ["Augmentation", preprocessing.augmentation],
    ["Netzarchitektur", preprocessing.network],
    ["Preprocessor", preprocessing.preprocessor],
  ];
  const list = document.createElement("dl");
  rows.forEach(([term, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = String(value);
    list.append(dt, dd);
  });
  details.replaceChildren(list);
}

function populateCollections() {
  const select = $("#collection-filter");
  state.payload.summary.collections.forEach((collection) => {
    const option = document.createElement("option");
    option.value = collection.label;
    option.textContent = `${collection.label} (${number.format(collection.count)})`;
    select.append(option);
  });
}

function matchesPadding(item, filter) {
  const value = item.padding_fraction;
  if (filter === "none") return value === 0;
  if (filter === "low") return value > 0 && value <= .25;
  if (filter === "medium") return value > .25 && value <= .5;
  if (filter === "high") return value > .5 && value <= .75;
  if (filter === "very-high") return value > .75;
  return true;
}

function applyFilters({ preserveSelection = true } = {}) {
  const query = $("#search-input").value.trim().toLocaleLowerCase("de");
  const collection = $("#collection-filter").value;
  const split = $("#split-filter").value;
  const padding = $("#padding-filter").value;
  const sort = $("#sort-select").value;
  state.filtered = state.payload.cases.filter((item) => {
    const searchText = `${item.case_id} ${item.source_case_id} ${item.collection} ${item.origin}`.toLocaleLowerCase("de");
    return (!query || searchText.includes(query))
      && (collection === "all" || item.collection === collection)
      && (split === "all" || item.split === split)
      && matchesPadding(item, padding);
  });
  const sorters = {
    case: (a, b) => a.case_id.localeCompare(b.case_id, "de", { numeric: true }),
    "size-desc": (a, b) => b.area - a.area,
    "size-asc": (a, b) => a.area - b.area,
    "padding-desc": (a, b) => b.padding_fraction - a.padding_fraction,
    "skeleton-desc": (a, b) => b.skeleton_pixels - a.skeleton_pixels,
  };
  state.filtered.sort(sorters[sort]);
  state.visibleLimit = 150;
  renderCaseList();

  const selectionExists = state.filtered.some((item) => item.case_id === state.activeId);
  if (!preserveSelection || !selectionExists) {
    if (state.filtered.length) selectCase(state.filtered[0].case_id);
    else clearCaseDetail();
  }
}

function renderCaseList() {
  const list = $("#case-list");
  list.replaceChildren();
  const visible = state.filtered.slice(0, state.visibleLimit);
  setText("#visible-count", `${number.format(state.filtered.length)} Fälle`);
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Für diesen Filter wurden keine Fälle gefunden.";
    list.append(empty);
  }
  visible.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `case-row${item.case_id === state.activeId ? " active" : ""}`;
    row.dataset.caseId = item.case_id;
    const title = document.createElement("strong");
    title.textContent = item.case_id;
    const padding = document.createElement("b");
    padding.textContent = `≈ ${pct(item.padding_percent)}`;
    padding.title = "Aus der Originalgröße geschätztes Padding vor der zufälligen Augmentation";
    const source = document.createElement("span");
    source.textContent = item.collection;
    const dimensions = document.createElement("small");
    dimensions.textContent = `${item.width} × ${item.height} px`;
    const labels = document.createElement("small");
    labels.textContent = `Sk ${number.format(item.skeleton_pixels)} · So ${number.format(item.soma_pixels)}`;
    row.append(title, padding, source, dimensions, labels);
    row.addEventListener("click", () => selectCase(item.case_id));
    list.append(row);
  });
  $("#load-more").hidden = state.filtered.length <= state.visibleLimit;
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const title = document.createElement("span");
  title.textContent = label;
  const copy = document.createElement("strong");
  copy.textContent = value;
  item.append(title, copy);
  return item;
}

function statsText(stats) {
  return `Werte ${decimal.format(stats.min)} bis ${decimal.format(stats.max)} · Mittel ${decimal.format(stats.mean)} · Std. ${decimal.format(stats.std)}`;
}

function setImage(selector, url) {
  const image = $(selector);
  image.removeAttribute("src");
  image.src = url;
}

function renderDetail(detail) {
  const { case: item, raw, preprocessed, training_sample: training, views } = detail;
  setText("#detail-origin", `${item.collection} · ${item.split === "training" ? "TRAINING" : "VALIDIERUNG"}`);
  setText("#detail-case", item.case_id);
  setText("#detail-source", item.source_case_id);
  const metrics = $("#case-metrics");
  metrics.replaceChildren(
    metric("Originalgröße", `${item.width} × ${item.height} px`),
    metric("Padding in diesem Sample", pct(training.padding_percent)),
    metric("Skeleton im Original", `${number.format(item.skeleton_pixels)} px`),
    metric("Soma im Original", `${number.format(item.soma_pixels)} px`),
    metric("Trainingsauswahl", training.force_foreground ? "Vordergrund erzwungen" : "zufälliger Ausschnitt"),
  );
  setImage("#view-raw", views.raw);
  setImage("#view-raw-label", views.raw_label);
  setImage("#view-preprocessed", views.preprocessed);
  setImage("#view-training", `${views.training}&v=${Date.now()}`);
  setText("#raw-stats", `${shape(raw.shape)} px · ${raw.dtype} · ${statsText(raw.stats)}`);
  setText("#raw-label-stats", `Skeleton ${number.format(raw.label_counts.skeleton)} px · Soma ${number.format(raw.label_counts.soma)} px`);
  setText("#preprocessed-stats", `${shape(preprocessed.shape)} px · ${statsText(preprocessed.stats)} · Crop-BBox ${JSON.stringify(preprocessed.bbox_used_for_cropping)}`);
  setText("#training-stats", `${shape(training.shape)} px · Seed ${training.seed} · Padding ${pct(training.padding_percent)} · Skeleton ${number.format(training.label_counts.skeleton)} px · Soma ${number.format(training.label_counts.soma)} px`);

  const provenance = $("#provenance-list");
  provenance.replaceChildren();
  [
    ["Quelle", item.collection],
    ["Ursprünglicher Fall", item.source_case_id],
    ["Bild", item.source_image],
    ["Label", item.source_label],
    ["Dataset139-ID", item.case_id],
  ].forEach(([term, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value || "–";
    provenance.append(dt, dd);
  });
}

function clearCaseDetail() {
  state.activeId = null;
  setText("#detail-case", "Kein Fall im aktuellen Filter");
  setText("#detail-source", "Filter ändern, um Fälle anzuzeigen.");
  $("#case-metrics").replaceChildren();
  ["#view-raw", "#view-raw-label", "#view-preprocessed", "#view-training"].forEach((selector) => $(selector).removeAttribute("src"));
}

async function selectCase(caseId, { keepSeed = false } = {}) {
  if (!keepSeed) state.seed = 1;
  state.activeId = caseId;
  renderCaseList();
  setText("#detail-case", `${caseId} wird geladen …`);
  try {
    const response = await fetch(`/api/training-dataset/139/cases/${encodeURIComponent(caseId)}?seed=${state.seed}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    if (state.activeId !== caseId) return;
    renderDetail(detail);
    const url = new URL(window.location.href);
    url.searchParams.set("case", caseId);
    history.replaceState(null, "", url);
  } catch (error) {
    setText("#detail-case", caseId);
    setText("#detail-source", `Dieser Fall konnte nicht geladen werden: ${error.message}`);
  }
}

function moveSelection(direction) {
  if (!state.filtered.length) return;
  const current = state.filtered.findIndex((item) => item.case_id === state.activeId);
  const next = Math.max(0, Math.min(state.filtered.length - 1, current + direction));
  selectCase(state.filtered[next].case_id);
  document.querySelector(`.case-row[data-case-id="${CSS.escape(state.filtered[next].case_id)}"]`)?.scrollIntoView({ block: "nearest" });
}

function newAugmentation() {
  if (!state.activeId) return;
  state.seed += 1;
  selectCase(state.activeId, { keepSeed: true });
}

function csvCell(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function exportCsv() {
  const header = ["case_id", "collection", "source_case_id", "split", "width", "height", "padding_percent", "skeleton_pixels", "soma_pixels"];
  const rows = state.filtered.map((item) => header.map((key) => csvCell(item[key])).join(";"));
  const blob = new Blob(["\ufeff", header.join(";"), "\r\n", rows.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "Dataset139_gefilterte_Uebersicht.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function bindEvents() {
  ["#collection-filter", "#split-filter", "#padding-filter", "#sort-select"].forEach((selector) => {
    $(selector).addEventListener("change", () => applyFilters());
  });
  $("#search-input").addEventListener("input", () => applyFilters());
  $("#load-more").addEventListener("click", () => {
    state.visibleLimit += 150;
    renderCaseList();
  });
  $("#previous-case").addEventListener("click", () => moveSelection(-1));
  $("#next-case").addEventListener("click", () => moveSelection(1));
  $("#new-augmentation").addEventListener("click", newAugmentation);
  $("#csv-button").addEventListener("click", exportCsv);
  $("#toggle-details").addEventListener("click", () => {
    const details = $("#preprocessing-details");
    details.hidden = !details.hidden;
    $("#toggle-details").textContent = details.hidden ? "Details anzeigen" : "Details ausblenden";
    $("#toggle-details").setAttribute("aria-expanded", String(!details.hidden));
  });
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowLeft") moveSelection(-1);
    if (event.key === "ArrowRight") moveSelection(1);
    if (event.key.toLocaleLowerCase("de") === "r") newAugmentation();
  });
}

async function initialize() {
  bindEvents();
  try {
    const response = await fetch("/api/training-dataset/139");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.payload = await response.json();
    renderSummary();
    renderPreprocessing();
    populateCollections();
    $("#loading-banner").hidden = true;
    ["#summary-section", "#flow-section", "#explorer"].forEach((selector) => $(selector).hidden = false);
    const requestedCase = new URL(window.location.href).searchParams.get("case");
    state.activeId = state.payload.cases.some((item) => item.case_id === requestedCase) ? requestedCase : null;
    applyFilters({ preserveSelection: true });
    if (requestedCase && state.activeId === requestedCase) selectCase(state.activeId);
  } catch (error) {
    showError(`Dataset139 konnte nicht geöffnet werden. ${error.message}`);
  }
}

initialize();
