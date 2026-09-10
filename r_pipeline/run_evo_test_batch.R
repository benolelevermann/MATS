# Batch wrapper around the unchanged scEvoView analysis stages used by
# EvoTest/scEvoView_pipeline_EvoCheck.Rmd.

.count_direct_cell_folders <- function(path) {
  children <- list.dirs(path, recursive = FALSE, full.names = TRUE)
  if (!length(children)) return(0L)
  sum(vapply(children, function(child) {
    length(list.files(child, pattern = "\\.swc$", full.names = TRUE,
                      ignore.case = TRUE)) > 0L
  }, logical(1)))
}

.cell_folder_candidates <- function(dataset_dir) {
  candidates <- unique(c(
    dataset_dir,
    list.dirs(dataset_dir, recursive = TRUE, full.names = TRUE)
  ))
  ignored <- grepl(
    "(^|[/\\\\])(Extraction|scEvoView_analysis|comparison_overview|plots|swc_cc|swc_final)([/\\\\]|$)",
    candidates,
    ignore.case = TRUE
  )
  candidates <- candidates[!ignored]
  counts <- vapply(candidates, .count_direct_cell_folders, integer(1))
  candidates <- candidates[counts > 0L]
  counts <- counts[counts > 0L]
  if (!length(candidates)) {
    return(data.frame(path = character(), cells = integer()))
  }
  data.frame(path = candidates, cells = unname(counts), stringsAsFactors = FALSE)
}

.find_cell_folder <- function(dataset_dir) {
  candidates <- .cell_folder_candidates(dataset_dir)
  if (!nrow(candidates)) return(NA_character_)

  # Prefer the directory containing the most per-cell folders. This selects,
  # for example, div10_CC/tracings_M237_totrace_Encrypted rather than one cell.
  candidates$path[[which.max(candidates$cells)]]
}

.find_existing_evo <- function(dataset_dir, analysis_dir) {
  preferred <- c(
    file.path(analysis_dir, "evoObject.rds"),
    file.path(dataset_dir, "Extraction", "evoObject.rds")
  )
  preferred <- preferred[file.exists(preferred)]
  if (length(preferred)) return(preferred[[1]])

  found <- list.files(
    dataset_dir,
    pattern = "^evoObject\\.rds$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  if (!length(found)) return(NA_character_)
  found[[which.max(file.info(found)$mtime)]]
}

.short_analysis_id <- function(dataset, max_length = 32L) {
  clean <- gsub("[^A-Za-z0-9_-]+", "_", dataset)
  clean <- gsub("_+", "_", clean)
  clean <- gsub("^_+|_+$", "", clean)
  if (!nzchar(clean)) clean <- "dataset"
  if (nchar(clean, type = "chars") <= max_length) return(clean)

  # Keep feature-extraction filenames below the legacy Windows MAX_PATH
  # boundary used by LMeasure. The weighted checksum keeps truncated names
  # deterministic and distinct without adding another R dependency.
  codes <- utf8ToInt(enc2utf8(dataset))
  checksum <- sum((as.double(codes) * seq_along(codes)) %% 1000000007) %%
    1000000007
  suffix <- sprintf("%08x", as.integer(checksum))
  prefix_length <- max(1L, max_length - nchar(suffix) - 1L)
  paste0(substr(clean, 1L, prefix_length), "_", suffix)
}

.replace_percell_dataset_id <- function(meta_data, analysis_id) {
  required <- c("cellnr", "cellframe")
  missing <- setdiff(required, names(meta_data))
  if (length(missing)) {
    stop(sprintf(
      "FEHLER: Per-cell-Metadaten enthalten nicht: %s",
      paste(missing, collapse = ", ")
    ))
  }
  meta_data$mousename <- analysis_id
  meta_data$cellid <- paste0(analysis_id, "_", meta_data$cellnr)
  meta_data$id <- paste0(meta_data$cellid, "_", meta_data$cellframe)
  meta_data
}

discoverEvoTestDatasets <- function(evo_test_root, analysis_id_max_length = 32L) {
  dataset_dirs <- list.dirs(evo_test_root, recursive = FALSE, full.names = TRUE)
  dataset_dirs <- dataset_dirs[
    !startsWith(basename(dataset_dirs), ".") &
      !startsWith(basename(dataset_dirs), "_")
  ]

  make_row <- function(dataset, dataset_dir, cell_folder) {
    analysis_dir <- file.path(dataset_dir, "scEvoView_analysis")
    existing_evo <- .find_existing_evo(dataset_dir, analysis_dir)
    if (is.na(cell_folder) && is.na(existing_evo)) return(NULL)
    data.frame(
      dataset = dataset,
      analysis_id = .short_analysis_id(dataset, analysis_id_max_length),
      dataset_dir = normalizePath(dataset_dir, winslash = "/", mustWork = TRUE),
      cell_folder = if (is.na(cell_folder)) NA_character_ else
        normalizePath(cell_folder, winslash = "/", mustWork = TRUE),
      analysis_dir = normalizePath(analysis_dir, winslash = "/", mustWork = FALSE),
      existing_evo = if (is.na(existing_evo)) NA_character_ else
        normalizePath(existing_evo, winslash = "/", mustWork = TRUE),
      stringsAsFactors = FALSE
    )
  }

  rows <- lapply(dataset_dirs, function(dataset_dir) {
    analysis_dir <- file.path(dataset_dir, "scEvoView_analysis")
    existing_evo <- .find_existing_evo(dataset_dir, analysis_dir)
    candidates <- .cell_folder_candidates(dataset_dir)

    # A grouped Evo export has one child directory per input image. Expose each
    # of those image folders as a separate PCA dataset. Ordinary experiment
    # folders (for example div10_CC) have one cell root and remain one dataset.
    if (is.na(existing_evo) && nrow(candidates) > 1L) {
      direct_candidates <- candidates[
        dirname(normalizePath(candidates$path, winslash = "/", mustWork = TRUE)) ==
          normalizePath(dataset_dir, winslash = "/", mustWork = TRUE),
        ,
        drop = FALSE
      ]
      if (nrow(direct_candidates) > 1L) {
        return(lapply(seq_len(nrow(direct_candidates)), function(index) {
          image_dir <- direct_candidates$path[[index]]
          make_row(
            paste(basename(dataset_dir), basename(image_dir), sep = " / "),
            image_dir,
            image_dir
          )
        }))
      }
    }

    list(make_row(basename(dataset_dir), dataset_dir, .find_cell_folder(dataset_dir)))
  })
  rows <- Filter(Negate(is.null), unlist(rows, recursive = FALSE))
  if (!length(rows)) {
    stop(sprintf("FEHLER: Keine auswertbaren Datensaetze in %s gefunden.", evo_test_root))
  }
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

.latest_extraction_dir <- function(output_dir) {
  extraction_dirs <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
  extraction_dirs <- extraction_dirs[grepl("^Extraction_", basename(extraction_dirs))]
  if (!length(extraction_dirs)) {
    stop(sprintf("FEHLER: Kein Extraction_* Ordner in %s erzeugt.", output_dir))
  }
  extraction_dirs[[which.max(file.info(extraction_dirs)$mtime)]]
}

.prepare_evo_metadata <- function(
  cell_folder,
  output_dir,
  dataset_name,
  cellline,
  acquisition_mode,
  pixel_width,
  pixel_height,
  pixel_depth,
  pixel_unit,
  is_swc_scaled,
  use_speedfiles,
  labels
) {
  stack_directory <- cell_folder
  if (acquisition_mode == "multiframe") {
    meta_data <- getMetaData_multiframe_v2(
      dataset_name, cellline, cell_folder,
      pixel_width, pixel_height, pixel_depth,
      pixel_unit, is_swc_scaled, stack_directory,
      use_speedfiles = use_speedfiles
    )
  } else if (acquisition_mode == "singlecell") {
    meta_data <- getMetaData_singlecell_v2(
      cellline, cell_folder, dataset_name,
      pixel_width, pixel_height, pixel_depth,
      pixel_unit, is_swc_scaled, stack_directory
    )
  } else if (acquisition_mode == "singlecellBATCH") {
    meta_data <- getMetaData_singlecellBATCH_v2(
      cell_folder, stack_directory, pixel_unit, is_swc_scaled
    )
  } else if (acquisition_mode == "singlecellBATCH_percell") {
    meta_data <- getMetaData_singlecellBATCH_percell_v2(
      cell_folder, pixel_unit, is_swc_scaled
    )
    # This metadata reader derives IDs from the (possibly very long) parent
    # folder name and has no dataset-name argument. Replace only its IDs before
    # parseSWCFilesStepOne writes LMeasure input/output filenames.
    meta_data <- .replace_percell_dataset_id(meta_data, dataset_name)
  } else {
    stop(sprintf("FEHLER: Unbekannter acquisition_mode '%s'.", acquisition_mode))
  }

  meta_data$cellline <- cellline
  if (!is_swc_scaled) {
    meta_data$pixel_width <- pixel_width
    meta_data$pixel_height <- pixel_height
    meta_data$pixel_depth <- pixel_depth
  }
  meta_data$human <- labels$human
  meta_data$em <- labels$em
  meta_data$batch <- labels$batch
  meta_data$rt_experiment <- labels$rt_experiment
  meta_data$is_surviving_rt <- labels$is_surviving_rt
  meta_data$post_rt <- labels$post_rt

  meta_data <- parseSWCFilesStepOne(meta_data, output_dir)
  if (anyNA(meta_data$pixel_width) || anyNA(meta_data$pixel_height) ||
      anyNA(meta_data$pixel_depth)) {
    stop(sprintf("FEHLER: Pixelgroessen enthalten NAs (%s).", dataset_name))
  }
  meta_data <- parseSWCFilesStepTwo(meta_data, output_dir)
  meta_data <- prepareColumns(meta_data)
  utils::write.csv(meta_data, file.path(output_dir, "meta_data.csv"), row.names = FALSE)
  meta_data
}

.build_mapped_evo <- function(output_dir, base_evo, inference_path) {
  input_dir <- .latest_extraction_dir(output_dir)
  data <- loadExtractionData(directory = input_dir)
  evo <- setupEvoObject("inVitro")
  evo <- loadMetadata(evo, data@meta_data_dir)
  evo <- loadMorphData(evo, data@features_dir)
  if ("custom.higherorder_length_min" %in% colnames(evo@morph_full)) {
    evo@morph_full$custom.higherorder_length_min <- NULL
  }
  evo <- loadTracesAndShapes(
    evo,
    tm_data_path = data@tm_coordinates_dir,
    soma_data_path = data@soma_coordinates_dir
  )

  feature_ids <- rownames(evo@morph_full)
  metadata_ids <- as.character(evo@meta.data$id)
  if (anyNA(feature_ids) || any(feature_ids == "") || anyDuplicated(feature_ids)) {
    stop("FEHLER: morph_full besitzt keine eindeutigen Zell-IDs.")
  }
  keep_ids <- feature_ids[feature_ids %in% metadata_ids]
  if (!length(keep_ids)) {
    stop("FEHLER: Keine gemeinsamen IDs zwischen morph_full und meta.data.")
  }
  excluded_cells <- evo@meta.data[!(metadata_ids %in% keep_ids), , drop = FALSE]
  excluded_cells$exclusion_reason <-
    "No usable morphology feature row after feature extraction"
  utils::write.csv(
    excluded_cells,
    file.path(output_dir, "excluded_unextractable_cells.csv"),
    row.names = FALSE
  )
  evo@morph_full <- evo@morph_full[keep_ids, , drop = FALSE]
  evo@meta.data <- evo@meta.data[match(keep_ids, metadata_ids), , drop = FALSE]
  stopifnot(identical(rownames(evo@morph_full), as.character(evo@meta.data$id)))

  evo <- runInferenceMapping(evo, base_evo, inference_path)
  morph_data <- evo@morph_mtx
  morph_data$id <- rownames(morph_data)
  evo@meta.data <- merge_meta_sort(evo@meta.data, morph_data, by = "id")
  evo
}

analyzeOneEvoTestDataset <- function(
  dataset_row,
  base_evo,
  inference_path,
  code_dir,
  cellline,
  acquisition_mode,
  pixel_width,
  pixel_height,
  pixel_depth,
  pixel_unit,
  is_swc_scaled,
  use_speedfiles,
  labels
) {
  if (is.na(dataset_row$cell_folder)) {
    stop(sprintf("FEHLER: Kein Zellordner fuer %s gefunden.", dataset_row$dataset))
  }
  output_dir <- dataset_row$analysis_dir
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("\n===== %s =====", dataset_row$dataset))
  message(sprintf("Interne Analyse-ID: %s", dataset_row$analysis_id))
  message(sprintf("Zellordner: %s", dataset_row$cell_folder))
  message(sprintf("Ausgabe:     %s", output_dir))

  meta_data <- .prepare_evo_metadata(
    dataset_row$cell_folder, output_dir, dataset_row$analysis_id, cellline,
    acquisition_mode, pixel_width, pixel_height, pixel_depth, pixel_unit,
    is_swc_scaled, use_speedfiles, labels
  )
  featEx <- newFeatureExtraction(name = paste0("FeatureExtraction_", dataset_row$dataset), code_dir)
  featEx <- addSaveDir(featEx, output_dir)
  featEx <- addMetaData(featEx, meta_data)
  checkFilePaths(featEx)
  checkMetaData(featEx)
  options(warn = -1)
  featEx <- runFeatureExtraction_NEW(featEx)
  featEx <- saveObjectWithoutMerging(featEx, output_dir)

  evo <- .build_mapped_evo(output_dir, base_evo, inference_path)
  saveRDS(evo, file.path(output_dir, "evoObject.rds"))
  evo
}

.metadata_for_comparison <- function(evo, dataset_name, source_rds) {
  metadata <- as.data.frame(evo@meta.data)
  if (!"id" %in% names(metadata)) metadata$id <- rownames(metadata)
  metadata$dataset <- dataset_name
  metadata$evo_source <- source_rds
  metadata
}

runEvoTestBatch <- function(
  evo_test_root,
  comparison_dir,
  comparison_metric,
  reuse_existing_evo,
  force_reanalyze,
  base_evo_path,
  inference_path,
  code_dir,
  cellline,
  acquisition_mode,
  pixel_width,
  pixel_height,
  pixel_depth,
  pixel_unit,
  is_swc_scaled,
  use_speedfiles,
  labels,
  analysis_id_max_length = 32L
) {
  datasets <- discoverEvoTestDatasets(evo_test_root, analysis_id_max_length)
  dir.create(comparison_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(datasets, file.path(comparison_dir, "detected_datasets.csv"), row.names = FALSE)
  print(datasets[, c("dataset", "cell_folder", "existing_evo"), drop = FALSE])

  base_evo <- readRDS(base_evo_path)
  results <- vector("list", nrow(datasets))
  status_rows <- vector("list", nrow(datasets))
  for (index in seq_len(nrow(datasets))) {
    row <- datasets[index, , drop = FALSE]
    can_reuse <- isTRUE(reuse_existing_evo) &&
      !(row$dataset %in% force_reanalyze) &&
      !is.na(row$existing_evo) && file.exists(row$existing_evo)
    if (can_reuse) {
      message(sprintf("CHECK: %s verwendet vorhandenes %s", row$dataset, row$existing_evo))
      evo <- readRDS(row$existing_evo)
      source_rds <- row$existing_evo
      mode <- "reused"
    } else {
      evo <- analyzeOneEvoTestDataset(
        row, base_evo, inference_path, code_dir, cellline, acquisition_mode,
        pixel_width, pixel_height, pixel_depth, pixel_unit, is_swc_scaled,
        use_speedfiles, labels
      )
      source_rds <- file.path(row$analysis_dir, "evoObject.rds")
      mode <- "analyzed"
    }
    results[[index]] <- .metadata_for_comparison(evo, row$dataset, source_rds)
    status_rows[[index]] <- data.frame(
      dataset = row$dataset,
      mode = mode,
      cells = nrow(results[[index]]),
      evo_source = source_rds,
      stringsAsFactors = FALSE
    )
  }

  combined <- dplyr::bind_rows(results)
  status <- dplyr::bind_rows(status_rows)
  combined$dataset <- factor(combined$dataset, levels = datasets$dataset)
  utils::write.csv(combined, file.path(comparison_dir, "all_cells_mapped.csv"), row.names = FALSE)
  utils::write.csv(status, file.path(comparison_dir, "analysis_status.csv"), row.names = FALSE)

  if (!comparison_metric %in% names(combined)) {
    stop(sprintf(
      "FEHLER: Vergleichsmetrik '%s' fehlt. Verfuegbar: %s",
      comparison_metric,
      paste(names(combined), collapse = ", ")
    ))
  }
  if (!is.numeric(combined[[comparison_metric]])) {
    stop(sprintf("FEHLER: Vergleichsmetrik '%s' ist nicht numerisch.", comparison_metric))
  }
  metric_data <- combined[is.finite(combined[[comparison_metric]]), , drop = FALSE]
  if (!nrow(metric_data)) stop("FEHLER: Keine endlichen Werte fuer den Vergleichsplot.")

  summary <- metric_data |>
    dplyr::group_by(dataset) |>
    dplyr::summarise(
      n = dplyr::n(),
      mean = mean(.data[[comparison_metric]]),
      median = stats::median(.data[[comparison_metric]]),
      sd = stats::sd(.data[[comparison_metric]]),
      q25 = stats::quantile(.data[[comparison_metric]], 0.25),
      q75 = stats::quantile(.data[[comparison_metric]], 0.75),
      .groups = "drop"
    )
  utils::write.csv(summary, file.path(comparison_dir, "metric_summary.csv"), row.names = FALSE)

  metric_plot <- ggplot2::ggplot(
    metric_data,
    ggplot2::aes(x = dataset, y = .data[[comparison_metric]], fill = dataset)
  ) +
    ggplot2::geom_violin(trim = FALSE, alpha = 0.35, color = NA) +
    ggplot2::geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.75) +
    ggplot2::geom_jitter(width = 0.12, size = 1.0, alpha = 0.35) +
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(legend.position = "none", axis.text.x = ggplot2::element_text(angle = 25, hjust = 1)) +
    ggplot2::labs(
      title = sprintf("EvoTest: %s aller Datensaetze", comparison_metric),
      x = NULL,
      y = comparison_metric
    )

  width <- max(7, 2.4 * length(unique(metric_data$dataset)))
  ggplot2::ggsave(file.path(comparison_dir, paste0(comparison_metric, "_side_by_side.png")),
                  metric_plot, width = width, height = 6, dpi = 300)
  ggplot2::ggsave(file.path(comparison_dir, paste0(comparison_metric, "_side_by_side.svg")),
                  metric_plot, width = width, height = 6)

  pca_plot <- NULL
  if (all(c("PC1", "PC2") %in% names(combined))) {
    pca_data <- combined[is.finite(combined$PC1) & is.finite(combined$PC2), , drop = FALSE]
    reference <- as.data.frame(base_evo@meta.data)
    pca_plot <- ggplot2::ggplot() +
      ggplot2::geom_point(
        data = reference,
        ggplot2::aes(PC1, PC2),
        color = "grey82", size = 0.35, alpha = 0.45
      ) +
      ggplot2::geom_point(
        data = pca_data,
        ggplot2::aes(PC1, PC2),
        color = "#E83E8C", size = 1.1, alpha = 0.75
      ) +
      ggplot2::facet_wrap(~dataset) +
      ggplot2::coord_fixed() +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::labs(title = "Alle EvoTest-Datensaetze auf derselben Referenz-PCA")
    ggplot2::ggsave(file.path(comparison_dir, "PCA_side_by_side.png"),
                    pca_plot, width = width, height = 6, dpi = 300)
    ggplot2::ggsave(file.path(comparison_dir, "PCA_side_by_side.svg"),
                    pca_plot, width = width, height = 6)
  }

  list(
    datasets = datasets,
    status = status,
    summary = summary,
    cells = combined,
    metric_plot = metric_plot,
    pca_plot = pca_plot
  )
}
