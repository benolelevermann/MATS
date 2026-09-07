options(stringsAsFactors = FALSE)

library(devtools)

# The legacy metadata helpers call these functions without a namespace
# (for example strex::str_before_last and RImageJROI::read.ijroi).
metadata_packages <- c(
  "strex",
  "stringr",
  "RImageJROI"
)

missing_metadata_packages <- metadata_packages[
  !vapply(
    metadata_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_metadata_packages) > 0) {
  stop(
    "Missing R packages required by the legacy metadata parser: ",
    paste(missing_metadata_packages, collapse = ", ")
  )
}

invisible(
  lapply(
    metadata_packages,
    function(package_name) {
      suppressPackageStartupMessages(
        library(
          package_name,
          character.only = TRUE
        )
      )
    }
  )
)

# ---------------------------------------------------------------------------
# Paths and acquisition metadata
# ---------------------------------------------------------------------------

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)

PROJECT <- "C:/Ole/20260721_CellClassification_v2"
ANALYSIS_ROOT <- file.path(
  PROJECT,
  "20260726_TestBleb",
  "analysis_net129"
)
CELL_ROOT <- file.path(
  ANALYSIS_ROOT,
  "04_cells_for_r_pipeline"
)
OUTPUT_DIR <- file.path(
  ANALYSIS_ROOT,
  "05_r_pipeline",
  "metadata"
)

# The source TIFFs contain X/Y resolution 24615.391 pixels per centimetre:
# 10000 micrometres / 24615.391 = 0.40625 micrometres per pixel.
PIXEL_WIDTH <- 0.40625
PIXEL_HEIGHT <- 0.40625
PIXEL_DEPTH <- 1.0
PIXEL_UNIT <- "microns"

CELLLINE <- "S24 GFP"
DATASET_NAME <- "TestBleb"

case_table <- data.frame(
  case_id = c(
    "TestBleb_A3_Blebbistatin",
    "TestBleb_A8_DMSO"
  ),
  condition = c(
    "Blebbistatin",
    "DMSO"
  ),
  original_image = c(
    file.path(
      PROJECT,
      "20260726_TestBleb",
      "firstrun",
      paste0(
        "260706_mc_S24GFPER25#02_div07_postROCKi_",
        "A3_Blebbistatin_ER.tif"
      )
    ),
    file.path(
      PROJECT,
      "20260726_TestBleb",
      "firstrun",
      paste0(
        "260706_mc_S24GFPER25#02_div07_postROCKi_",
        "A8_DMSO_ER.tif"
      )
    )
  ),
  stringsAsFactors = FALSE
)


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

find_cell_directories <- function(case_id) {
  case_root <- file.path(CELL_ROOT, case_id)
  if (!dir.exists(case_root)) {
    stop("Cell root does not exist: ", case_root)
  }

  cell_dirs <- list.dirs(
    case_root,
    recursive = FALSE,
    full.names = TRUE
  )
  cell_dirs <- cell_dirs[
    grepl("^cell[0-9]+$", basename(cell_dirs))
  ]
  cell_dirs[
    order(
      as.integer(
        sub("^cell", "", basename(cell_dirs))
      )
    )
  ]
}


# ---------------------------------------------------------------------------
# Build one combined metadata table with unique IDs
# ---------------------------------------------------------------------------

load_all(PACKAGE_DIR)

metadata_parts <- list()

for (case_index in seq_len(nrow(case_table))) {
  case_id <- case_table$case_id[[case_index]]
  condition <- case_table$condition[[case_index]]
  original_image <- case_table$original_image[[case_index]]
  assert_all_exist(original_image, paste0(case_id, " original TIFF"))

  cell_dirs <- find_cell_directories(case_id)
  if (length(cell_dirs) == 0) {
    stop("No cell#### folders found for case: ", case_id)
  }

  cell_numbers <- sub(
    "^cell",
    "",
    basename(cell_dirs)
  )
  unique_cell_ids <- paste0(
    case_id,
    "_",
    cell_numbers
  )

  swc_paths <- file.path(cell_dirs, "seg-000.swc")
  traces_paths <- file.path(cell_dirs, "seg.traces")
  soma_paths <- file.path(cell_dirs, "soma.zip")
  raw_paths <- file.path(cell_dirs, "raw.tif")
  bounds_paths <- file.path(cell_dirs, "bounds.zip")

  assert_all_exist(swc_paths, paste0(case_id, " SWC"))
  assert_all_exist(traces_paths, paste0(case_id, " TRACES"))
  assert_all_exist(soma_paths, paste0(case_id, " soma ROI"))
  assert_all_exist(raw_paths, paste0(case_id, " raw TIFF"))
  assert_all_exist(bounds_paths, paste0(case_id, " bounds ROI"))

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
    batch = "DIV07_TestBleb",
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

meta_data <- do.call(
  rbind,
  metadata_parts
)

if (anyDuplicated(meta_data$id)) {
  stop("Metadata IDs are not unique.")
}

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

# Recreate the derived SWC files in a clean, dataset-specific output folder.
meta_data <- parseSWCFilesStepOne(
  meta_data,
  OUTPUT_DIR
)

# The exported TIFF/TRACES files do not reliably retain the source
# calibration, so explicitly restore the measured acquisition calibration.
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

meta_data <- parseSWCFilesStepTwo(
  meta_data,
  OUTPUT_DIR
)
meta_data <- prepareColumns(meta_data)

# Ensure that identity, grouping and geometry fields survive legacy helpers.
source_rows <- do.call(rbind, metadata_parts)
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

metadata_path <- file.path(
  OUTPUT_DIR,
  "meta_data.csv"
)
write.csv(
  meta_data,
  metadata_path,
  row.names = FALSE
)

cat("\nMetadata successfully written:\n", metadata_path, "\n", sep = "")
cat("Cells: ", nrow(meta_data), "\n", sep = "")
cat(
  "Conditions:\n",
  paste(
    capture.output(
      print(table(meta_data$condition))
    ),
    collapse = "\n"
  ),
  "\n",
  sep = ""
)
cat(
  "Pixel size: ",
  PIXEL_WIDTH,
  " x ",
  PIXEL_HEIGHT,
  " ",
  PIXEL_UNIT,
  "\n",
  sep = ""
)
