options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(devtools))

required_packages <- c("strex", "stringr", "RImageJROI")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
}
invisible(lapply(required_packages, function(package_name) {
  suppressPackageStartupMessages(library(package_name, character.only = TRUE))
}))

arguments <- commandArgs(trailingOnly = TRUE)
METHOD <- if (length(arguments) >= 1) arguments[[1]] else "gcut_inspired"
allowed_methods <- c("topology_baseline", "matrix_forest_inspired", "gcut_inspired")
if (!METHOD %in% allowed_methods) {
  stop("Unknown method: ", METHOD)
}

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
PROJECT <- "C:/Ole/20260721_CellClassification_v2"
TEST_ROOT <- file.path(PROJECT, "20280812_newTest2")
CELL_ROOT <- file.path(TEST_ROOT, paste0("06_evo_cells_", METHOD))
ANALYSIS_ROOT <- file.path(TEST_ROOT, paste0("07_evo_pipeline_", METHOD))
OUTPUT_DIR <- file.path(ANALYSIS_ROOT, "05_r_pipeline", "metadata")
ORIGINAL_IMAGE <- file.path(TEST_ROOT, "01_inputimages", "NewImage2_0000.tif")

# Keep the calibration used by the previously working Evo pipeline.
PIXEL_WIDTH <- 0.40625
PIXEL_HEIGHT <- 0.40625
PIXEL_DEPTH <- 1.0
PIXEL_UNIT <- "microns"
CELLLINE <- "S24 GFP"

assert_all_exist <- function(paths, label) {
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    stop(label, " missing:\n", paste(missing, collapse = "\n"))
  }
}

if (!dir.exists(CELL_ROOT)) stop("Cell root does not exist: ", CELL_ROOT)
assert_all_exist(ORIGINAL_IMAGE, "Original TIFF")

cell_dirs <- list.dirs(CELL_ROOT, recursive = FALSE, full.names = TRUE)
cell_dirs <- cell_dirs[grepl("^cell[0-9]+$", basename(cell_dirs))]
cell_dirs <- cell_dirs[order(as.integer(sub("^cell", "", basename(cell_dirs))))]
if (length(cell_dirs) == 0) stop("No cell#### folders found in: ", CELL_ROOT)

cell_numbers <- sub("^cell", "", basename(cell_dirs))
unique_cell_ids <- paste0("NewImage2_", METHOD, "_", cell_numbers)
swc_paths <- file.path(cell_dirs, "seg-000.swc")
traces_paths <- file.path(cell_dirs, "seg.traces")
soma_paths <- file.path(cell_dirs, "soma.zip")
raw_paths <- file.path(cell_dirs, "raw.tif")
bounds_paths <- file.path(cell_dirs, "bounds.zip")
location_paths <- file.path(cell_dirs, "location.zip")

assert_all_exist(swc_paths, "SWC")
assert_all_exist(traces_paths, "TRACES")
assert_all_exist(soma_paths, "soma ROI")
assert_all_exist(raw_paths, "raw TIFF")
assert_all_exist(bounds_paths, "bounds ROI")
assert_all_exist(location_paths, "location ROI")

source_rows <- data.frame(
  id = paste0(unique_cell_ids, "_0"),
  mousename = "NewImage2",
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
  hyperstack_directory = ORIGINAL_IMAGE,
  human = 0,
  em = 0,
  batch = "NewImage2_paper_assignment",
  condition = METHOD,
  treatment = METHOD,
  source_case = "NewImage2",
  source_image = ORIGINAL_IMAGE,
  rt_experiment = 0,
  is_surviving_rt = 0,
  post_rt = 0,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (anyDuplicated(source_rows$id)) stop("Metadata IDs are not unique.")

load_all(PACKAGE_DIR)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
meta_data <- parseSWCFilesStepOne(source_rows, OUTPUT_DIR)
if ("pixel_width_real" %in% names(meta_data)) meta_data$pixel_width_real <- PIXEL_WIDTH
if ("pixel_height_real" %in% names(meta_data)) meta_data$pixel_height_real <- PIXEL_HEIGHT
if ("pixel_depth_real" %in% names(meta_data)) meta_data$pixel_depth_real <- PIXEL_DEPTH
if ("pixel_unit_real" %in% names(meta_data)) meta_data$pixel_unit_real <- PIXEL_UNIT
meta_data <- parseSWCFilesStepTwo(meta_data, OUTPUT_DIR)
meta_data <- prepareColumns(meta_data)

for (column_name in c(
  "id", "mousename", "cellid", "cellnr", "boundingbox_roi",
  "hyperstack_directory", "condition", "treatment", "source_case", "source_image"
)) {
  meta_data[[column_name]] <- source_rows[[column_name]]
}
meta_data$is_swc_scaled <- FALSE

metadata_path <- file.path(OUTPUT_DIR, "meta_data.csv")
write.csv(meta_data, metadata_path, row.names = FALSE)
cat("\nNEWIMAGE2 METADATA READY\n")
cat("Method: ", METHOD, "\n", sep = "")
cat("Metadata: ", metadata_path, "\n", sep = "")
cat("Cells: ", nrow(meta_data), "\n", sep = "")
