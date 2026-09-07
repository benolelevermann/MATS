options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(devtools))

required_packages <- c("strex", "stringr", "RImageJROI")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing R packages: ",
    paste(missing_packages, collapse = ", ")
  )
}
invisible(lapply(
  required_packages,
  function(package_name) {
    suppressPackageStartupMessages(
      library(package_name, character.only = TRUE)
    )
  }
))

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
PROJECT <- "C:/Ole/20260721_CellClassification_v2"
RUN_ROOT <- file.path(PROJECT, "20260728_TestBleb_v3")
OUTPUT_DIR <- file.path(RUN_ROOT, "05_r_pipeline", "metadata")

# Acquisition calibration used for these TestBleb images.
PIXEL_WIDTH <- 0.40625
PIXEL_HEIGHT <- 0.40625
PIXEL_DEPTH <- 1.0
PIXEL_UNIT <- "microns"
CELLLINE <- "S24 GFP"

case_table <- data.frame(
  case_id = c("Bleb", "DMSO"),
  condition = c("Blebbistatin", "DMSO"),
  cell_root = c(
    file.path(RUN_ROOT, "04_exported_cells_Bleb_safe"),
    file.path(RUN_ROOT, "04_exported_cells_DMSO_safe")
  ),
  original_image = c(
    file.path(RUN_ROOT, "01_inputimages", "Bleb_0000.tif"),
    file.path(RUN_ROOT, "01_inputimages", "DMSO_0000.tif")
  ),
  stringsAsFactors = FALSE
)

assert_all_exist <- function(paths, label) {
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    stop(
      label,
      " missing:\n",
      paste(missing, collapse = "\n")
    )
  }
}

find_cell_directories <- function(root) {
  if (!dir.exists(root)) {
    stop("Cell root does not exist: ", root)
  }
  directories <- list.dirs(
    root,
    recursive = FALSE,
    full.names = TRUE
  )
  directories <- directories[
    grepl("^cell[0-9]+$", basename(directories))
  ]
  directories[
    order(as.integer(sub("^cell", "", basename(directories))))
  ]
}

load_all(PACKAGE_DIR)

metadata_parts <- vector("list", nrow(case_table))

for (case_index in seq_len(nrow(case_table))) {
  case_id <- case_table$case_id[[case_index]]
  condition <- case_table$condition[[case_index]]
  cell_root <- case_table$cell_root[[case_index]]
  original_image <- case_table$original_image[[case_index]]

  assert_all_exist(original_image, paste0(case_id, " original TIFF"))

  cell_dirs <- find_cell_directories(cell_root)
  if (length(cell_dirs) == 0) {
    stop("No cell#### folders found in: ", cell_root)
  }

  cell_numbers <- sub("^cell", "", basename(cell_dirs))
  unique_cell_ids <- paste0(case_id, "_", cell_numbers)

  swc_paths <- file.path(cell_dirs, "seg-000.swc")
  traces_paths <- file.path(cell_dirs, "seg.traces")
  soma_paths <- file.path(cell_dirs, "soma.zip")
  raw_paths <- file.path(cell_dirs, "raw.tif")
  bounds_paths <- file.path(cell_dirs, "bounds.zip")
  location_paths <- file.path(cell_dirs, "location.zip")

  assert_all_exist(swc_paths, paste0(case_id, " SWC"))
  assert_all_exist(traces_paths, paste0(case_id, " TRACES"))
  assert_all_exist(soma_paths, paste0(case_id, " soma ROI"))
  assert_all_exist(raw_paths, paste0(case_id, " raw TIFF"))
  assert_all_exist(bounds_paths, paste0(case_id, " bounds ROI"))
  assert_all_exist(location_paths, paste0(case_id, " location ROI"))

  metadata_parts[[case_index]] <- data.frame(
    id = paste0(unique_cell_ids, "_0"),
    mousename = case_id,
    swc = swc_paths,
    traces = traces_paths,
    cellline = CELLLINE,
    pixel_width = PIXEL_WIDTH,
    pixel_height = PIXEL_HEIGHT,
    pixel_depth = PIXEL_DEPTH,
    pixel_unit = PIXEL_UNIT,
    is_swc_scaled = FALSE,
    cellfolder = cell_dirs,
    cellnr = cell_numbers,
    cellframe = 0,
    cellid = unique_cell_ids,
    somamask_tif = NA_character_,
    somaraw_tif = NA_character_,
    roi_groupz = soma_paths,
    crop_raw = raw_paths,
    acquisition = "singleframe_new",
    boundingbox_roi = bounds_paths,
    hyperstack_directory = original_image,
    human = 0,
    em = 0,
    batch = "DIV07_TestBleb_v3",
    condition = condition,
    treatment = condition,
    source_case = case_id,
    source_image = original_image,
    rt_experiment = 0,
    is_surviving_rt = 0,
    post_rt = 0,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

source_rows <- do.call(rbind, metadata_parts)
if (anyDuplicated(source_rows$id)) {
  stop("Metadata IDs are not unique.")
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

meta_data <- parseSWCFilesStepOne(source_rows, OUTPUT_DIR)

if ("pixel_width_real" %in% names(meta_data)) {
  meta_data$pixel_width_real <- PIXEL_WIDTH
}
if ("pixel_height_real" %in% names(meta_data)) {
  meta_data$pixel_height_real <- PIXEL_HEIGHT
}
if ("pixel_depth_real" %in% names(meta_data)) {
  meta_data$pixel_depth_real <- PIXEL_DEPTH
}
if ("pixel_unit_real" %in% names(meta_data)) {
  meta_data$pixel_unit_real <- PIXEL_UNIT
}

meta_data <- parseSWCFilesStepTwo(meta_data, OUTPUT_DIR)
meta_data <- prepareColumns(meta_data)

# Restore fields that the legacy helpers can reorder or overwrite.
meta_data$id <- source_rows$id
meta_data$mousename <- source_rows$mousename
meta_data$cellid <- source_rows$cellid
meta_data$cellnr <- source_rows$cellnr
meta_data$is_swc_scaled <- FALSE
meta_data$boundingbox_roi <- source_rows$boundingbox_roi
meta_data$hyperstack_directory <- source_rows$hyperstack_directory
meta_data$condition <- source_rows$condition
meta_data$treatment <- source_rows$treatment
meta_data$source_case <- source_rows$source_case
meta_data$source_image <- source_rows$source_image

metadata_path <- file.path(OUTPUT_DIR, "meta_data.csv")
write.csv(meta_data, metadata_path, row.names = FALSE)

cat("\n============================================================\n")
cat("TESTBLEB V3 METADATA READY\n")
cat("============================================================\n")
cat("Metadata: ", metadata_path, "\n", sep = "")
cat("Cells: ", nrow(meta_data), "\n", sep = "")
print(table(meta_data$condition, useNA = "ifany"))

