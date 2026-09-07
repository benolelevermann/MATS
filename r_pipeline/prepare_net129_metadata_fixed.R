options(stringsAsFactors = FALSE)

library(devtools)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)

CELL_ROOT <- "C:/Ole/20260721_CellClassification_v2/cells_for_r_pipeline"
OVERVIEW_TIF <- paste0(
  "C:/Ole/20260721_CellClassification_v2/",
  "overview_inference/input_2d/overview_max_0000.tif"
)
OUTPUT_DIR <- paste0(
  "C:/Ole/20260721_CellClassification_v2/r_pipeline/",
  "20260724_r_pipeline_output_fixed"
)

# The exported SWC coordinates are local crop PIXELS, not physical microns.
PIXEL_WIDTH <- 1.25
PIXEL_HEIGHT <- 1.25
PIXEL_DEPTH <- 1.0
PIXEL_UNIT <- "microns"
DATASET_NAME <- "overview129"
CELLLINE <- "S24 GFP"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

assert_all_exist <- function(paths, label) {
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    stop(
      paste0(
        label,
        " missing:\n",
        paste(missing, collapse = "\n")
      )
    )
  }
}

# ---------------------------------------------------------------------------
# Build clean metadata directly from the cell folders
# ---------------------------------------------------------------------------

load_all(PACKAGE_DIR)

cell_dirs <- list.dirs(
  CELL_ROOT,
  recursive = FALSE,
  full.names = TRUE
)
cell_dirs <- cell_dirs[grepl("^cell[0-9]+$", basename(cell_dirs))]
cell_dirs <- cell_dirs[order(as.integer(sub("^cell", "", basename(cell_dirs))))]

if (length(cell_dirs) == 0) {
  stop("No cell#### folders found in: ", CELL_ROOT)
}

cell_numbers <- sub("^cell", "", basename(cell_dirs))
swc_paths <- file.path(cell_dirs, "seg-000.swc")
traces_paths <- file.path(cell_dirs, "seg.traces")
soma_paths <- file.path(cell_dirs, "soma.zip")
raw_paths <- file.path(cell_dirs, "raw.tif")
bounds_paths <- file.path(cell_dirs, "bounds.zip")

assert_all_exist(swc_paths, "SWC")
assert_all_exist(traces_paths, "TRACES")
assert_all_exist(soma_paths, "Soma ROI")
assert_all_exist(raw_paths, "Raw TIFF")
assert_all_exist(bounds_paths, "Per-cell bounds ROI")
assert_all_exist(OVERVIEW_TIF, "Overview TIFF")

meta_data <- data.frame(
  id = paste0(DATASET_NAME, "_", cell_numbers, "_0"),
  mousename = DATASET_NAME,
  swc = swc_paths,
  traces = traces_paths,
  cellline = CELLLINE,
  pixel_width = PIXEL_WIDTH,
  pixel_height = PIXEL_HEIGHT,
  pixel_depth = PIXEL_DEPTH,
  pixel_unit = PIXEL_UNIT,
  # Critical: the source SWCs contain local pixel coordinates.
  is_swc_scaled = FALSE,
  cellfolder = cell_dirs,
  cellnr = cell_numbers,
  cellframe = 0,
  cellid = paste0(DATASET_NAME, "_", cell_numbers),
  somamask_tif = NA_character_,
  somaraw_tif = NA_character_,
  roi_groupz = soma_paths,
  crop_raw = raw_paths,
  acquisition = "singleframe_new",
  # Critical: one ROI ZIP with exactly one bounding ROI per cell.
  boundingbox_roi = bounds_paths,
  # Use the image file, not only its containing directory.
  hyperstack_directory = OVERVIEW_TIF,
  human = 0,
  em = 0,
  batch = "DIV07_net129",
  rt_experiment = 0,
  is_surviving_rt = 0,
  post_rt = 0,
  check.names = FALSE
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Recreate all derived SWC paths in a fresh output directory.
meta_data <- parseSWCFilesStepOne(meta_data, OUTPUT_DIR)

# The generated TIFF/TRACES files do not expose calibration in exactly the
# legacy format expected by the parser. The acquisition values are known, so
# set them explicitly before SWC scaling. This also corrects misleading
# non-NA values (for example a detected pixel_depth_real of 2).
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

# Keep these identity and geometry fields explicit after legacy helpers.
meta_data$id <- paste0(DATASET_NAME, "_", cell_numbers, "_0")
meta_data$mousename <- DATASET_NAME
meta_data$cellid <- paste0(DATASET_NAME, "_", cell_numbers)
meta_data$is_swc_scaled <- FALSE
meta_data$boundingbox_roi <- bounds_paths
meta_data$hyperstack_directory <- OVERVIEW_TIF

metadata_path <- file.path(OUTPUT_DIR, "meta_data.csv")
write.csv(meta_data, metadata_path, row.names = FALSE)

cat("\nMetadata successfully written:\n", metadata_path, "\n", sep = "")
cat("Cells: ", nrow(meta_data), "\n", sep = "")
cat("First ID: ", meta_data$id[[1]], "\n", sep = "")
cat(
  "First bounds: ",
  meta_data$boundingbox_roi[[1]],
  "\n",
  sep = ""
)
cat(
  "is_swc_scaled values: ",
  paste(unique(meta_data$is_swc_scaled), collapse = ", "),
  "\n",
  sep = ""
)

