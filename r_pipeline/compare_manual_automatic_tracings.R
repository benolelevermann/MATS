# Build a self-contained tracing QC report for manual and automatic SWC trees.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop(paste(
    "Aufruf: Rscript compare_manual_automatic_tracings.R",
    "<manual_extraction_dir> <treatment_csv> <automatic_cell_dir> <output_dir>",
    "[automatic_pixel_size_um]"
  ))
}

manual_extraction_dir <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
treatment_csv <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
automatic_cell_dir <- normalizePath(args[[3]], winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(args[[4]], winslash = "/", mustWork = FALSE)
automatic_pixel_size_um <- if (length(args) >= 5L) as.numeric(args[[5]]) else 0.2875008

if (!is.finite(automatic_pixel_size_um) || automatic_pixel_size_um <= 0) {
  stop("automatic_pixel_size_um muss eine positive Zahl sein.")
}
required_packages <- c("ggplot2", "patchwork", "jsonlite")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(sprintf("Fehlende R-Pakete: %s", paste(missing_packages, collapse = ", ")))
}

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grepl("^--file=", script_arg)][1])
script_dir <- dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE))
source(file.path(script_dir, "plot_analyzed_skeletons.R"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

htmlEscape <- function(value) {
  value <- gsub("&", "&amp;", as.character(value), fixed = TRUE)
  value <- gsub("<", "&lt;", value, fixed = TRUE)
  value <- gsub(">", "&gt;", value, fixed = TRUE)
  value <- gsub('"', "&quot;", value, fixed = TRUE)
  value
}

slugify <- function(value) {
  value <- iconv(value, to = "ASCII//TRANSLIT")
  value <- gsub("[^A-Za-z0-9]+", "_", value)
  tolower(gsub("(^_+|_+$)", "", value))
}

safeMedian <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::median(x, na.rm = TRUE)
}

safeIqr <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::IQR(x, na.rm = TRUE)
}

manual_assignments <- utils::read.csv(
  treatment_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8-BOM"
)
required_manual_columns <- c("id", "treatment")
if (!all(required_manual_columns %in% colnames(manual_assignments))) {
  stop("Treatment-Tabelle muss die Spalten 'id' und 'treatment' enthalten.")
}
manual_swc_dir <- file.path(manual_extraction_dir, "swc_final")
if (!dir.exists(manual_swc_dir)) {
  stop(sprintf("Manueller swc_final-Ordner fehlt: %s", manual_swc_dir))
}
manual_meta <- data.frame(
  raw_id = as.character(manual_assignments$id),
  group = as.character(manual_assignments$treatment),
  swc_file = file.path(manual_swc_dir, paste0(manual_assignments$id, ".swc")),
  stringsAsFactors = FALSE
)
manual_meta <- manual_meta[file.exists(manual_meta$swc_file), , drop = FALSE]
manual_meta$id <- paste0(manual_meta$raw_id, " | ", manual_meta$group)

automatic_dirs <- list.dirs(automatic_cell_dir, full.names = TRUE, recursive = FALSE)
automatic_rows <- lapply(automatic_dirs, function(cell_dir) {
  swc_file <- file.path(cell_dir, "seg-000.swc")
  review_file <- file.path(cell_dir, "review.json")
  if (!file.exists(swc_file)) return(NULL)
  review <- if (file.exists(review_file)) {
    jsonlite::fromJSON(review_file, simplifyVector = TRUE)
  } else {
    list()
  }
  cell_id <- basename(cell_dir)
  case_label <- if (!is.null(review$case) && nzchar(review$case)) review$case else "unbekannt"
  data.frame(
    raw_id = cell_id,
    group = as.character(case_label),
    job_id = if (!is.null(review$job_id)) as.character(review$job_id) else NA_character_,
    swc_file = swc_file,
    id = paste0(cell_id, " | ", case_label),
    stringsAsFactors = FALSE
  )
})
automatic_meta <- do.call(rbind, Filter(Negate(is.null), automatic_rows))
if (is.null(automatic_meta) || !nrow(automatic_meta)) {
  stop(sprintf("Keine automatischen seg-000.swc gefunden: %s", automatic_cell_dir))
}

galleries <- list()
qc_tables <- list()

renderGroup <- function(meta, source_kind, group_name, coordinate_scale, source_description) {
  group_slug <- paste(slugify(source_kind), slugify(group_name), sep = "_")
  group_root <- file.path(output_dir, "galleries", group_slug)
  result <- renderAnalyzedSkeletons(
    meta_data = meta[, c("id", "swc_file"), drop = FALSE],
    output_dir = group_root,
    pixel_unit = "um",
    skeletons_per_page = 12L,
    print_pages = FALSE,
    max_inline_pages = 0L,
    coordinate_scale = coordinate_scale,
    report_title = sprintf("%s | %s", source_kind, group_name),
    source_description = source_description
  )
  qc <- result$qc
  qc$raw_id <- meta$raw_id[match(qc$id, meta$id)]
  qc$source_kind <- source_kind
  qc$group <- group_name
  qc$display_group <- paste(source_kind, group_name, sep = " | ")
  qc_tables[[length(qc_tables) + 1L]] <<- qc
  galleries[[length(galleries) + 1L]] <<- data.frame(
    source_kind = source_kind,
    group = group_name,
    count = nrow(qc),
    index = normalizePath(result$index, winslash = "/", mustWork = TRUE),
    first_page = normalizePath(result$pages[[1]], winslash = "/", mustWork = TRUE),
    stringsAsFactors = FALSE
  )
}

manual_group_order <- c("DMSO", "dilutedDMSO", "Blebbistatin", "Fasudil", "Y-27632")
manual_groups <- unique(manual_meta$group)
manual_groups <- c(
  manual_group_order[manual_group_order %in% manual_groups],
  sort(setdiff(manual_groups, manual_group_order))
)
for (group_name in manual_groups) {
  group_meta <- manual_meta[manual_meta$group == group_name, , drop = FALSE]
  renderGroup(
    group_meta,
    "Manuell",
    group_name,
    coordinate_scale = 1,
    source_description = "finale, von scEvoView analysierte swc_final-Dateien"
  )
}

for (group_name in sort(unique(automatic_meta$group))) {
  group_meta <- automatic_meta[automatic_meta$group == group_name, , drop = FALSE]
  renderGroup(
    group_meta,
    "Automatisch",
    group_name,
    coordinate_scale = automatic_pixel_size_um,
    source_description = sprintf(
      "exportierte seg-000.swc; fuer diesen Bericht mit %.7f um/px skaliert",
      automatic_pixel_size_um
    )
  )
}

all_qc <- do.call(rbind, qc_tables)
all_qc$display_group <- factor(
  all_qc$display_group,
  levels = unique(all_qc$display_group)
)
utils::write.csv(all_qc, file.path(output_dir, "trace_qc_all.csv"), row.names = FALSE)

summary_groups <- split(all_qc, all_qc$display_group)
group_summary <- do.call(rbind, lapply(summary_groups, function(rows) {
  data.frame(
    source = rows$source_kind[[1]],
    group = rows$group[[1]],
    cells = nrow(rows),
    median_length_um = safeMedian(rows$total_length_2d),
    iqr_length_um = safeIqr(rows$total_length_2d),
    median_branch_points = safeMedian(rows$branch_points),
    iqr_branch_points = safeIqr(rows$branch_points),
    median_tips = safeMedian(rows$tips),
    iqr_tips = safeIqr(rows$tips),
    cells_with_root_problem = sum(rows$roots != 1),
    orphan_parent_links = sum(rows$orphan_parent_links),
    stringsAsFactors = FALSE
  )
}))
rownames(group_summary) <- NULL
utils::write.csv(group_summary, file.path(output_dir, "group_summary.csv"), row.names = FALSE)

metric_data <- rbind(
  data.frame(all_qc[, c("display_group", "source_kind")], metric = "Gesamtlaenge (um)", value = all_qc$total_length_2d),
  data.frame(all_qc[, c("display_group", "source_kind")], metric = "Verzweigungen", value = all_qc$branch_points),
  data.frame(all_qc[, c("display_group", "source_kind")], metric = "Endpunkte", value = all_qc$tips)
)
comparison_plot <- ggplot2::ggplot(
  metric_data,
  ggplot2::aes(x = display_group, y = value, fill = source_kind)
) +
  ggplot2::geom_violin(alpha = 0.40, color = NA, trim = TRUE) +
  ggplot2::geom_boxplot(width = 0.16, outlier.alpha = 0.28, linewidth = 0.3) +
  ggplot2::facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  ggplot2::scale_fill_manual(values = c("Manuell" = "#178f82", "Automatisch" = "#db4c91")) +
  ggplot2::labs(
    title = "Tracing-Strukturen nach Quelle und Metadaten-Gruppe",
    subtitle = "Gruppenverteilungen; keine Zell-zu-Zell-Paarung",
    x = NULL,
    y = NULL,
    fill = "Quelle"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 28, hjust = 1),
    panel.grid.minor = ggplot2::element_blank(),
    strip.text = ggplot2::element_text(face = "bold"),
    plot.title = ggplot2::element_text(face = "bold")
  )
comparison_plot_path <- file.path(output_dir, "tracing_comparison.png")
ggplot2::ggsave(
  comparison_plot_path,
  comparison_plot,
  width = 13,
  height = 11,
  dpi = 180,
  bg = "white"
)

formatNumber <- function(value, digits = 1L) {
  format(round(value, digits), nsmall = digits, decimal.mark = ",", big.mark = ".")
}

summary_rows <- vapply(seq_len(nrow(group_summary)), function(index) {
  row <- group_summary[index, ]
  sprintf(
    paste0(
      "<tr><td><span class=\"pill %s\">%s</span></td><td>%s</td><td>%d</td>",
      "<td>%s ± %s</td><td>%s ± %s</td><td>%s ± %s</td><td>%d</td><td>%d</td></tr>"
    ),
    ifelse(row$source == "Manuell", "manual", "automatic"),
    htmlEscape(row$source),
    htmlEscape(row$group),
    row$cells,
    formatNumber(row$median_length_um),
    formatNumber(row$iqr_length_um),
    formatNumber(row$median_branch_points),
    formatNumber(row$iqr_branch_points),
    formatNumber(row$median_tips),
    formatNumber(row$iqr_tips),
    row$cells_with_root_problem,
    row$orphan_parent_links
  )
}, character(1))

gallery_table <- do.call(rbind, galleries)
gallery_cards <- vapply(seq_len(nrow(gallery_table)), function(index) {
  row <- gallery_table[index, ]
  gallery_index_rel <- gsub("\\\\", "/", substring(row$index, nchar(output_dir) + 2L))
  first_page_rel <- gsub("\\\\", "/", substring(row$first_page, nchar(output_dir) + 2L))
  sprintf(
    paste0(
      "<article class=\"gallery\"><div><span class=\"eyebrow\">%s</span>",
      "<h3>%s</h3><p>%d Zellen</p></div>",
      "<a href=\"%s\"><img loading=\"lazy\" src=\"%s\" alt=\"%s %s\"></a>",
      "<a class=\"button\" href=\"%s\">Alle Skeletons ansehen</a></article>"
    ),
    htmlEscape(row$source_kind),
    htmlEscape(row$group),
    row$count,
    htmlEscape(gallery_index_rel),
    htmlEscape(first_page_rel),
    htmlEscape(row$source_kind),
    htmlEscape(row$group),
    htmlEscape(gallery_index_rel)
  )
}, character(1))

automatic_cases <- sort(unique(automatic_meta$group))
automatic_has_dmso <- any(tolower(automatic_cases) == "dmso")
mismatch_note <- if (!automatic_has_dmso) {
  sprintf(
    paste0(
      "<div class=\"warning\"><strong>Wichtiger Metadaten-Hinweis:</strong> ",
      "Der Ordner heißt <code>%s</code>, aber die gespeicherten Review-Daten enthalten ",
      "keine DMSO-Zelle. Sie sind als <strong>%s</strong> beschriftet. Deshalb ist dieser ",
      "Bericht ein technischer Tracing-Vergleich und noch kein belastbarer biologischer ",
      "DMSO-gegen-DMSO-Vergleich.</div>"
    ),
    htmlEscape(basename(automatic_cell_dir)),
    htmlEscape(paste(sprintf("%s (%d)", names(table(automatic_meta$group)), as.integer(table(automatic_meta$group))), collapse = " und "))
  )
} else {
  ""
}

index_path <- file.path(output_dir, "index.html")
writeLines(c(
  "<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\">",
  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
  "<title>Manuelle und automatische Tracings</title>",
  paste0(
    "<style>:root{--ink:#17211d;--muted:#637069;--paper:#f1f4ef;--card:#fff;--manual:#178f82;--auto:#db4c91}",
    "*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}",
    "header,main{width:min(1500px,calc(100% - 40px));margin:auto}header{padding:46px 0 22px}",
    "h1{font:clamp(34px,5vw,60px)/1.05 Georgia,serif;margin:0 0 12px;max-width:950px}h2{font:32px Georgia,serif;margin:44px 0 14px}",
    "h3{font-size:24px;margin:4px 0}.lead{font-size:18px;color:var(--muted);max-width:1000px}",
    ".warning{background:#fff3cf;border-left:6px solid #d59b12;padding:16px 18px;margin:22px 0;border-radius:4px}",
    ".facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:22px 0}",
    ".fact,.gallery,.plot-card,.table-card{background:var(--card);padding:18px;box-shadow:0 12px 34px #17211d12}",
    ".fact strong{font:34px Georgia,serif;display:block}.fact span{color:var(--muted)}",
    ".plot-card img,.gallery img{width:100%;height:auto;display:block}.table-card{overflow:auto}",
    "table{border-collapse:collapse;width:100%;min-width:940px}th,td{text-align:right;padding:10px;border-bottom:1px solid #dfe5e0}",
    "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.pill{display:inline-block;color:white;padding:3px 9px;border-radius:999px;font-size:12px}",
    ".pill.manual{background:var(--manual)}.pill.automatic{background:var(--auto)}",
    ".gallery-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}.gallery{display:grid;gap:12px}.eyebrow{color:var(--muted);text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:bold}",
    ".button{display:inline-block;background:var(--ink);color:white;text-decoration:none;padding:10px 14px;border-radius:4px;width:max-content}",
    "code{background:#e5e9e4;padding:2px 5px;border-radius:3px}footer{color:var(--muted);padding:45px 0 70px}</style></head><body>"
  ),
  sprintf(
    paste0(
      "<header><h1>Manuelle und automatische Skeleton-Tracings</h1>",
      "<p class=\"lead\">Vollständige Sichtkontrolle der SWC-Bäume und ein technischer Gruppenvergleich. ",
      "Cyan zeigt den Baum, Magenta die Wurzel beziehungsweise das Soma und Gelb Verzweigungspunkte.</p>%s",
      "<div class=\"facts\"><div class=\"fact\"><strong>%d</strong><span>manuelle Zellen</span></div>",
      "<div class=\"fact\"><strong>%d</strong><span>automatische, manuell freigegebene Zellen</span></div>",
      "<div class=\"fact\"><strong>%d</strong><span>Galerien</span></div></div></header><main>"
    ),
    mismatch_note,
    nrow(manual_meta),
    nrow(automatic_meta),
    nrow(gallery_table)
  ),
  "<h2>Strukturvergleich</h2>",
  paste0(
    "<p>Die Verteilungen vergleichen Gruppen, nicht dieselben Einzelzellen. Die automatische Länge wurde für diese Darstellung ",
    sprintf("mit %.7f µm/px skaliert. Knotenanzahlen werden bewusst nicht verglichen, weil beide Tracer unterschiedlich dicht samplen.</p>", automatic_pixel_size_um)
  ),
  "<div class=\"plot-card\"><img src=\"tracing_comparison.png\" alt=\"Verteilungen von Länge, Verzweigungen und Endpunkten\"></div>",
  "<h2>Kennzahlen</h2>",
  paste0(
    "<div class=\"table-card\"><table><thead><tr><th>Quelle</th><th>Gruppe</th><th>Zellen</th>",
    "<th>Länge Median ± IQR (µm)</th><th>Verzweigungen Median ± IQR</th><th>Endpunkte Median ± IQR</th>",
    "<th>Wurzelproblem</th><th>verwaiste Links</th></tr></thead><tbody>"
  ),
  summary_rows,
  "</tbody></table><p><a href=\"group_summary.csv\">Gruppentabelle</a> · <a href=\"trace_qc_all.csv\">Einzelzell-QC</a></p></div>",
  "<h2>Alle Skeletons</h2><p>Ein Klick öffnet jede Gruppe über alle Seiten.</p><div class=\"gallery-grid\">",
  gallery_cards,
  "</div>",
  sprintf(
    paste0(
      "<footer><p><strong>Manuelle Quelle:</strong> %s<br>",
      "<strong>Automatische Quelle:</strong> %s</p>",
      "<p>Manuell: finale <code>swc_final</code>-Dateien. Automatisch: aktuelle exportierte <code>seg-000.swc</code>; ",
      "sie wurden im Bericht kalibriert, aber noch nicht als neuer scEvoView-Feature-Extraktionslauf geschrieben.</p></footer></main></body></html>"
    ),
    htmlEscape(manual_swc_dir),
    htmlEscape(automatic_cell_dir)
  )
), index_path, useBytes = TRUE)

message(sprintf("CHECK: Vergleichsbericht erzeugt: %s", index_path))
message(sprintf("CHECK: %d manuelle und %d automatische SWCs dargestellt.", nrow(manual_meta), nrow(automatic_meta)))
