# Plot the exact final SWC files used by scEvoView feature extraction.

readAnalyzedSwc <- function(path, cell_id) {
  if (is.na(path) || !nzchar(path) || !file.exists(path)) {
    warning(sprintf("SWC fehlt fuer %s: %s", cell_id, path), call. = FALSE)
    return(NULL)
  }

  swc <- tryCatch(
    utils::read.table(
      path,
      comment.char = "#",
      header = FALSE,
      fill = TRUE,
      stringsAsFactors = FALSE
    ),
    error = function(error) {
      warning(
        sprintf("SWC fuer %s konnte nicht gelesen werden: %s", cell_id, error$message),
        call. = FALSE
      )
      NULL
    }
  )
  if (is.null(swc) || nrow(swc) == 0 || ncol(swc) < 7) return(NULL)

  swc <- swc[, seq_len(7), drop = FALSE]
  colnames(swc) <- c("node", "type", "x", "y", "z", "radius", "parent")
  numeric_columns <- c("node", "type", "x", "y", "z", "radius", "parent")
  swc[numeric_columns] <- lapply(swc[numeric_columns], as.numeric)
  swc <- swc[
    stats::complete.cases(swc[, c("node", "x", "y", "parent")]),
    ,
    drop = FALSE
  ]
  if (!nrow(swc)) return(NULL)

  parent_row <- match(swc$parent, swc$node)
  edge_rows <- which(swc$parent >= 0 & !is.na(parent_row))
  edges <- data.frame(
    cell_id = cell_id,
    node = swc$node[edge_rows],
    parent = swc$parent[edge_rows],
    x = swc$x[edge_rows],
    y = swc$y[edge_rows],
    xend = swc$x[parent_row[edge_rows]],
    yend = swc$y[parent_row[edge_rows]],
    stringsAsFactors = FALSE
  )
  if (nrow(edges)) {
    edges$length_2d <- sqrt((edges$x - edges$xend)^2 + (edges$y - edges$yend)^2)
  } else {
    edges$length_2d <- numeric(0)
  }

  child_count <- table(swc$parent[swc$parent >= 0])
  branch_nodes <- as.numeric(names(child_count)[child_count > 1])
  tip_nodes <- setdiff(swc$node, swc$parent[swc$parent >= 0])
  nodes <- data.frame(
    cell_id = cell_id,
    node = swc$node,
    type = swc$type,
    x = swc$x,
    y = swc$y,
    role = ifelse(
      swc$type == 1 | swc$parent < 0,
      "soma",
      ifelse(
        swc$node %in% branch_nodes,
        "branch",
        ifelse(swc$node %in% tip_nodes, "tip", "path")
      )
    ),
    stringsAsFactors = FALSE
  )

  list(
    nodes = nodes,
    edges = edges,
    qc = data.frame(
      id = cell_id,
      swc_file = normalizePath(path, winslash = "/", mustWork = FALSE),
      nodes = nrow(nodes),
      edges = nrow(edges),
      roots = sum(nodes$role == "soma"),
      branch_points = sum(nodes$role == "branch"),
      tips = sum(nodes$role == "tip"),
      orphan_parent_links = sum(swc$parent >= 0 & is.na(parent_row)),
      total_length_2d = sum(edges$length_2d),
      stringsAsFactors = FALSE
    )
  )
}

plotAnalyzedSwc <- function(trace, pixel_unit) {
  x_range <- range(trace$nodes$x, finite = TRUE)
  y_range <- range(trace$nodes$y, finite = TRUE)
  square_side <- max(diff(x_range), diff(y_range), 1) * 1.10
  x_center <- mean(x_range)
  y_center <- mean(y_range)
  x_limits <- x_center + c(-0.5, 0.5) * square_side
  y_limits <- y_center + c(-0.5, 0.5) * square_side

  ggplot2::ggplot() +
    ggplot2::geom_segment(
      data = trace$edges,
      ggplot2::aes(x = x, y = y, xend = xend, yend = yend),
      color = "#00c6d8",
      linewidth = 0.48,
      lineend = "round"
    ) +
    ggplot2::geom_point(
      data = trace$nodes[trace$nodes$role == "branch", , drop = FALSE],
      ggplot2::aes(x = x, y = y),
      color = "#ffd166",
      size = 1.15
    ) +
    ggplot2::geom_point(
      data = trace$nodes[trace$nodes$role == "soma", , drop = FALSE],
      ggplot2::aes(x = x, y = y),
      color = "#ed5c9e",
      size = 3.2
    ) +
    ggplot2::scale_y_reverse() +
    ggplot2::coord_equal(xlim = x_limits, ylim = y_limits, expand = FALSE) +
    ggplot2::labs(
      title = trace$qc$id,
      subtitle = sprintf(
        "%d Knoten | %d Verzweigungen | %.1f %s",
        trace$qc$nodes,
        trace$qc$branch_points,
        trace$qc$total_length_2d,
        pixel_unit
      )
    ) +
    ggplot2::theme_void(base_size = 10) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "#111917", color = NA),
      panel.background = ggplot2::element_rect(fill = "#111917", color = NA),
      plot.title = ggplot2::element_text(color = "white", face = "bold", size = 10),
      plot.subtitle = ggplot2::element_text(color = "#aebbb6", size = 8),
      aspect.ratio = 1,
      plot.margin = ggplot2::margin(8, 8, 8, 8)
    )
}

renderAnalyzedSkeletons <- function(
  meta_data,
  output_dir,
  pixel_unit,
  skeletons_per_page = 12L,
  skeleton_preview_ids = character(0),
  print_pages = TRUE,
  max_inline_pages = 2L
) {
  if (!("swc_file" %in% colnames(meta_data))) {
    stop("FEHLER: meta_data besitzt nach dem SWC-Parsing keine Spalte 'swc_file'.")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("FEHLER: ggplot2 wird fuer die Skeleton-Vorschau benoetigt.")
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("FEHLER: patchwork wird fuer die Skeleton-Vorschau benoetigt.")
  }
  if (!is.numeric(skeletons_per_page) || length(skeletons_per_page) != 1 ||
      is.na(skeletons_per_page) || skeletons_per_page < 1) {
    stop("FEHLER: skeletons_per_page muss eine positive ganze Zahl sein.")
  }
  skeletons_per_page <- as.integer(skeletons_per_page)

  preview_rows <- meta_data
  if (length(skeleton_preview_ids)) {
    missing_ids <- setdiff(skeleton_preview_ids, as.character(meta_data$id))
    if (length(missing_ids)) {
      warning(
        sprintf("Nicht gefundene skeleton_preview_ids: %s", paste(missing_ids, collapse = ", ")),
        call. = FALSE
      )
    }
    preview_rows <- preview_rows[
      as.character(preview_rows$id) %in% skeleton_preview_ids,
      ,
      drop = FALSE
    ]
  }

  analyzed_traces <- lapply(seq_len(nrow(preview_rows)), function(index) {
    readAnalyzedSwc(
      preview_rows$swc_file[[index]],
      as.character(preview_rows$id[[index]])
    )
  })
  analyzed_traces <- Filter(Negate(is.null), analyzed_traces)
  if (!length(analyzed_traces)) {
    stop("FEHLER: Keine analysierbare SWC-Datei fuer die Skeleton-Vorschau gefunden.")
  }

  skeleton_qc <- do.call(rbind, lapply(analyzed_traces, `[[`, "qc"))
  skeleton_preview_dir <- file.path(output_dir, "analyzed_skeletons")
  dir.create(skeleton_preview_dir, showWarnings = FALSE, recursive = TRUE)
  old_pages <- list.files(
    skeleton_preview_dir,
    pattern = "^analyzed_skeletons_page_[0-9]+\\.png$",
    full.names = TRUE
  )
  if (length(old_pages)) unlink(old_pages)
  utils::write.csv(
    skeleton_qc,
    file.path(skeleton_preview_dir, "analyzed_skeleton_qc.csv"),
    row.names = FALSE
  )

  page_number <- ceiling(seq_along(analyzed_traces) / skeletons_per_page)
  trace_pages <- split(analyzed_traces, page_number)
  page_paths <- character(length(trace_pages))
  for (page_index in seq_along(trace_pages)) {
    page_plots <- lapply(
      trace_pages[[page_index]],
      plotAnalyzedSwc,
      pixel_unit = pixel_unit
    )
    page_columns <- min(4L, length(page_plots))
    page <- patchwork::wrap_plots(
      page_plots,
      ncol = page_columns,
      widths = rep(1, page_columns)
    ) +
      patchwork::plot_annotation(
        title = sprintf(
          "Analysierte SWC-Skeletons - Seite %d/%d",
          page_index,
          length(trace_pages)
        ),
        subtitle = paste(
          "Quelle: meta_data$swc_file (swc_final);",
          "Cyan = Baum, Magenta = Soma/Wurzel, Gelb = Verzweigung"
        )
      )
    page_paths[[page_index]] <- file.path(
      skeleton_preview_dir,
      sprintf("analyzed_skeletons_page_%03d.png", page_index)
    )
    ggplot2::ggsave(
      page_paths[[page_index]],
      page,
      width = 16,
      height = max(4.2, 4.0 * ceiling(length(page_plots) / 4)),
      dpi = 180,
      bg = "white"
    )
    if (isTRUE(print_pages) && page_index <= max_inline_pages) print(page)
  }

  html_cards <- vapply(
    seq_along(page_paths),
    function(index) sprintf(
      paste0(
        "<section><h2>Seite %d/%d</h2>",
        "<a href=\"%s\"><img loading=\"lazy\" src=\"%s\" alt=\"Skeleton-Seite %d\"></a></section>"
      ),
      index,
      length(page_paths),
      basename(page_paths[[index]]),
      basename(page_paths[[index]]),
      index
    ),
    character(1)
  )
  index_path <- file.path(skeleton_preview_dir, "index.html")
  writeLines(
    c(
      "<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\">",
      "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
      "<title>Analysierte SWC-Skeletons</title>",
      paste0(
        "<style>body{margin:0;background:#eef1ec;color:#17211d;font:15px system-ui,sans-serif}",
        "header,main{width:min(1500px,calc(100% - 40px));margin:auto}",
        "header{padding:34px 0 20px}h1{font:44px Georgia,serif;margin:0 0 8px}",
        "p{color:#66716c}a{color:#087b65}section{margin:0 0 28px;background:#fff;padding:14px;box-shadow:0 15px 40px #17211d12}",
        "h2{font-size:14px;margin:0 0 10px}img{display:block;width:100%;height:auto}</style></head><body>"
      ),
      sprintf(
        paste0(
          "<header><h1>Analysierte SWC-Skeletons</h1>",
          "<p>%d Zellen · Quelle: <code>meta_data$swc_file</code> aus <code>swc_final</code>. ",
          "Dies sind die Baeume, die scEvoView analysiert. ",
          "<a href=\"analyzed_skeleton_qc.csv\">QC-Tabelle herunterladen</a>.</p></header><main>"
        ),
        length(analyzed_traces)
      ),
      html_cards,
      "</main></body></html>"
    ),
    index_path,
    useBytes = TRUE
  )

  message(sprintf(
    "CHECK: %d tatsaechlich analysierte SWC-Skeletons auf %d Seite(n) dargestellt: %s",
    length(analyzed_traces),
    length(trace_pages),
    skeleton_preview_dir
  ))
  invisible(list(
    qc = skeleton_qc,
    pages = page_paths,
    index = index_path,
    traces = analyzed_traces
  ))
}
