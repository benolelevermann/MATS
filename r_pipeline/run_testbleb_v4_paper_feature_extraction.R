options(stringsAsFactors = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
METHOD <- if (length(arguments) >= 1) arguments[[1]] else "gcut_inspired"
allowed_methods <- c("topology_baseline", "matrix_forest_inspired", "gcut_inspired")
if (!METHOD %in% allowed_methods) stop("Unknown method: ", METHOD)

script_dir <- "C:/Ole/20260721_CellClassification_v2/r_pipeline"
source_path <- file.path(script_dir, "run_testbleb_feature_extraction.R")
if (!file.exists(source_path)) stop("Source extraction script not found: ", source_path)

source_code <- paste(readLines(source_path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
old_root_block <- paste0(
  "ANALYSIS_ROOT <- paste0(\n",
  "  \"C:/Ole/20260721_CellClassification_v2/\",\n",
  "  \"20260726_TestBleb/analysis_net129\"\n",
  ")"
)
new_root_block <- paste0(
  "ANALYSIS_ROOT <- \"C:/Ole/20260721_CellClassification_v2/",
  "20260730_TestBleb_v4/07_evo_pipeline_", METHOD, "\""
)
if (!grepl(old_root_block, source_code, fixed = TRUE)) {
  stop("Could not find expected ANALYSIS_ROOT block in: ", source_path)
}
source_code <- sub(old_root_block, new_root_block, source_code, fixed = TRUE)
source_code <- sub(
  'name = "FeatureExtraction_TestBleb"',
  paste0('name = "FeatureExtraction_TestBlebV4_', METHOD, '"'),
  source_code,
  fixed = TRUE
)
cat("Running Bleb/DMSO feature extraction for method: ", METHOD, "\n", sep = "")
eval(parse(text = source_code), envir = .GlobalEnv)
