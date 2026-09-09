# Preserve MATS soma-connector node types after the legacy scEvoView parser.
# The parser intentionally converts every non-root node to type 2. MATS v3 uses
# type 1 for pixel-wise paths inside the soma so those edges can be excluded
# from neurite visualization and length measurements.

readMatsSwcTypeMap <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  header <- lines[grepl("^#", trimws(lines))]
  if (!any(grepl("canonical-1px", header, fixed = TRUE))) {
    return(NULL)
  }
  data_lines <- lines[nzchar(trimws(lines)) & !grepl("^#", trimws(lines))]
  fields <- strsplit(trimws(data_lines), "[[:space:]]+")
  valid <- lengths(fields) >= 7L
  fields <- fields[valid]
  if (!length(fields)) return(NULL)
  node_ids <- vapply(fields, `[[`, character(1), 1L)
  node_types <- vapply(fields, `[[`, character(1), 2L)
  stats::setNames(node_types, node_ids)
}

restoreMatsSwcFileTypes <- function(source_path, final_path) {
  type_map <- readMatsSwcTypeMap(source_path)
  if (is.null(type_map)) return(FALSE)
  lines <- readLines(final_path, warn = FALSE, encoding = "UTF-8")
  data_indices <- which(nzchar(trimws(lines)) & !grepl("^#", trimws(lines)))
  changed <- FALSE
  for (line_index in data_indices) {
    fields <- strsplit(trimws(lines[[line_index]]), "[[:space:]]+")[[1]]
    if (length(fields) < 7L) next
    source_type <- unname(type_map[fields[[1]]])
    if (!length(source_type) || is.na(source_type)) next
    if (!identical(fields[[2]], source_type)) {
      fields[[2]] <- source_type
      lines[[line_index]] <- paste(fields, collapse = " ")
      changed <- TRUE
    }
  }
  if (changed) {
    writeLines(lines, final_path, useBytes = TRUE)
  }
  changed
}

restoreMatsGeneratedSwcTypes <- function(meta_data) {
  if (!("swc" %in% names(meta_data))) {
    warning("MATS-Typwiederherstellung uebersprungen: Spalte 'swc' fehlt.", call. = FALSE)
    return(meta_data)
  }
  final_column <- if ("swc_final" %in% names(meta_data)) {
    "swc_final"
  } else if ("swc_file" %in% names(meta_data)) {
    "swc_file"
  } else {
    warning(
      "MATS-Typwiederherstellung uebersprungen: swc_final/swc_file fehlt.",
      call. = FALSE
    )
    return(meta_data)
  }
  restored <- mapply(
    function(source_path, final_path) {
      if (is.na(source_path) || is.na(final_path) ||
          !file.exists(source_path) || !file.exists(final_path)) {
        return(FALSE)
      }
      restoreMatsSwcFileTypes(source_path, final_path)
    },
    as.character(meta_data$swc),
    as.character(meta_data[[final_column]]),
    USE.NAMES = FALSE
  )
  message(sprintf(
    "CHECK: MATS-Somaverbindertypen in %d/%d finalen SWCs wiederhergestellt.",
    sum(restored),
    length(restored)
  ))
  meta_data
}
