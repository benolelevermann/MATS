options(stringsAsFactors = FALSE)

library(devtools)

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)

ANALYSIS_ROOT <- paste0(
  "C:/Ole/20260721_CellClassification_v2/",
  "20260726_TestBleb/analysis_net129"
)
FEATURE_ROOT <- file.path(
  ANALYSIS_ROOT,
  "05_r_pipeline",
  "feature_extraction"
)
SAVING_ROOT <- file.path(
  ANALYSIS_ROOT,
  "05_r_pipeline",
  "saved_extraction"
)

load_all(PACKAGE_DIR)

run_directories <- list.dirs(
  FEATURE_ROOT,
  recursive = FALSE,
  full.names = TRUE
)
run_directories <- run_directories[
  startsWith(
    basename(run_directories),
    "AP_"
  )
]

if (length(run_directories) == 0) {
  stop("No AP_* feature-extraction directory found in: ", FEATURE_ROOT)
}

run_info <- file.info(run_directories)
latest_run <- run_directories[
  which.max(run_info$mtime)
]

feature_object_path <- file.path(
  latest_run,
  "data",
  "featureExtraction.rds"
)

if (!file.exists(feature_object_path)) {
  stop("FeatureExtraction object is missing: ", feature_object_path)
}

dir.create(
  SAVING_ROOT,
  recursive = TRUE,
  showWarnings = FALSE
)

cat("Loading feature object:\n", feature_object_path, "\n\n", sep = "")

featEx <- readRDS(
  feature_object_path
)

featEx <- saveObjectWithoutMerging(
  featEx,
  SAVING_ROOT
)

extraction_directories <- list.dirs(
  SAVING_ROOT,
  recursive = FALSE,
  full.names = TRUE
)
extraction_directories <- extraction_directories[
  startsWith(
    basename(extraction_directories),
    "Extraction_"
  )
]

if (length(extraction_directories) == 0) {
  stop(
    "saveObjectWithoutMerging did not create an Extraction_* directory."
  )
}

extraction_info <- file.info(
  extraction_directories
)
latest_extraction <- extraction_directories[
  which.max(extraction_info$mtime)
]

required_files <- file.path(
  latest_extraction,
  c(
    "meta.csv",
    "features.csv"
  )
)
missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  stop(
    "Saved extraction is incomplete. Missing:\n",
    paste(missing_files, collapse = "\n")
  )
}

latest_path_file <- file.path(
  SAVING_ROOT,
  "_LATEST_EXTRACTION_PATH.txt"
)
writeLines(
  latest_extraction,
  latest_path_file
)

meta_rows <- nrow(
  read.csv(
    file.path(
      latest_extraction,
      "meta.csv"
    ),
    check.names = FALSE
  )
)
feature_rows <- nrow(
  read.csv(
    file.path(
      latest_extraction,
      "features.csv"
    ),
    check.names = FALSE
  )
)

cat("\n============================================================\n")
cat("SAVED EXTRACTION READY\n")
cat("============================================================\n")
cat("Input AP directory:\n", latest_run, "\n\n", sep = "")
cat("Use this exact path as input_dir:\n")
cat(latest_extraction, "\n\n")
cat("meta.csv rows: ", meta_rows, "\n", sep = "")
cat("features.csv rows: ", feature_rows, "\n", sep = "")
cat("Path also written to:\n", latest_path_file, "\n", sep = "")
