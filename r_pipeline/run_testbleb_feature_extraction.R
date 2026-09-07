options(stringsAsFactors = FALSE)
options(warn = 1)

library(devtools)

MIN_SPATSTAT_UNIVAR_VERSION <- package_version("3.2.0")

if (!requireNamespace("spatstat.univar", quietly = TRUE)) {
  stop("Missing R package: spatstat.univar")
}
if (
  packageVersion("spatstat.univar") <
    MIN_SPATSTAT_UNIVAR_VERSION
) {
  stop(
    "spatstat.univar must be >= ",
    MIN_SPATSTAT_UNIVAR_VERSION,
    ". Restart R before updating the package."
  )
}
if (!requireNamespace("spatstat", quietly = TRUE)) {
  stop("Missing or unloadable R package: spatstat")
}

suppressPackageStartupMessages(
  library(spatstat)
)

pipeline_packages <- c(
  "cli",
  "nat",
  "ggplot2",
  "RImageJROI",
  "dplyr",
  "matlib",
  "StereoMorph",
  "misc3d",
  "oce",
  "sf",
  "sp",
  "ks",
  "reticulate",
  "stringr",
  "strex",
  "smoothr",
  "concaveman",
  "crayon"
)

missing_pipeline_packages <- pipeline_packages[
  !vapply(
    pipeline_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_pipeline_packages) > 0) {
  stop(
    "Missing R packages: ",
    paste(missing_pipeline_packages, collapse = ", ")
  )
}

invisible(
  lapply(
    pipeline_packages,
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


# Compatibility layer for the legacy nat package and current igraph.
patch_nat_igraph_compatibility <- function() {
  nat_namespace <- asNamespace("nat")
  patched_functions <- character()

  for (
    function_name in ls(
      nat_namespace,
      all.names = TRUE
    )
  ) {
    function_object <- get(
      function_name,
      envir = nat_namespace,
      inherits = FALSE
    )

    if (!is.function(function_object)) {
      next
    }

    original_body <- paste(
      deparse(
        body(function_object),
        width.cutoff = 500
      ),
      collapse = "\n"
    )
    patched_body <- gsub(
      "igraph::graph\\.dfs([[:space:]]*\\()",
      "igraph::dfs\\1",
      original_body
    )
    patched_body <- gsub(
      "graph\\.dfs([[:space:]]*\\()",
      "igraph::dfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "igraph::graph\\.bfs([[:space:]]*\\()",
      "igraph::bfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "graph\\.bfs([[:space:]]*\\()",
      "igraph::bfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "neimode([[:space:]]*)=",
      "mode\\1=",
      patched_body
    )
    patched_body <- gsub(
      "father([[:space:]]*)=",
      "parent\\1=",
      patched_body
    )

    if (identical(original_body, patched_body)) {
      next
    }

    body(function_object) <- parse(
      text = patched_body,
      keep.source = FALSE
    )[[1]]

    binding_was_locked <- bindingIsLocked(
      function_name,
      nat_namespace
    )
    if (binding_was_locked) {
      unlockBinding(
        function_name,
        nat_namespace
      )
    }
    assign(
      function_name,
      function_object,
      envir = nat_namespace
    )
    if (binding_was_locked) {
      lockBinding(
        function_name,
        nat_namespace
      )
    }
    patched_functions <- c(
      patched_functions,
      function_name
    )
  }

  cat(
    "nat/igraph compatibility patch: ",
    length(patched_functions),
    " function(s) patched\n",
    sep = ""
  )
  invisible(patched_functions)
}

patch_nat_igraph_compatibility()


# ---------------------------------------------------------------------------
# TestBleb paths
# ---------------------------------------------------------------------------

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
CODE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new"
)
ANALYSIS_ROOT <- paste0(
  "C:/Ole/20260721_CellClassification_v2/",
  "20260726_TestBleb/analysis_net129"
)
METADATA_PATH <- file.path(
  ANALYSIS_ROOT,
  "05_r_pipeline",
  "metadata",
  "meta_data.csv"
)
OUTPUT_DIR <- file.path(
  ANALYSIS_ROOT,
  "05_r_pipeline",
  "feature_extraction"
)

# Set TRUE for a quick one-cell check; use FALSE for the complete run.
TEST_ONE_CELL <- FALSE

if (!file.exists(METADATA_PATH)) {
  stop("Metadata file does not exist: ", METADATA_PATH)
}

load_all(PACKAGE_DIR)

meta_data <- read.csv(
  METADATA_PATH,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (nrow(meta_data) == 0) {
  stop("Metadata contains zero cells.")
}

if (TEST_ONE_CELL) {
  meta_data <- meta_data[1, , drop = FALSE]
  cat(
    "TEST MODE: extracting only ",
    meta_data$id[[1]],
    "\n",
    sep = ""
  )
} else {
  cat(
    "FULL MODE: extracting ",
    nrow(meta_data),
    " cells\n",
    sep = ""
  )
}

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

featEx <- newFeatureExtraction(
  name = "FeatureExtraction_TestBleb",
  CODE_DIR
)
featEx <- addSaveDir(
  featEx,
  OUTPUT_DIR
)
featEx <- addMetaData(
  featEx,
  meta_data
)

checkFilePaths(featEx)
checkMetaData(featEx)

featEx <- runFeatureExtraction_NEW(featEx)

run_directories <- list.dirs(
  OUTPUT_DIR,
  full.names = TRUE,
  recursive = FALSE
)
run_directories <- run_directories[
  startsWith(
    basename(run_directories),
    "AP_"
  )
]

if (length(run_directories) > 0) {
  latest_run_directory <- run_directories[
    which.max(
      file.info(run_directories)$mtime
    )
  ]
} else {
  latest_run_directory <- NA_character_
}

failed_indices <- tryCatch(
  as.integer(featEx@failed_cells),
  error = function(error_condition) {
    integer()
  }
)
failed_indices <- failed_indices[
  !is.na(failed_indices) &
    failed_indices >= 1 &
    failed_indices <= nrow(meta_data)
]

cat("\nFeature extraction finished.\n")
cat(
  "Successful cells: ",
  nrow(meta_data) - length(failed_indices),
  " / ",
  nrow(meta_data),
  "\n",
  sep = ""
)
cat(
  "Failed cells: ",
  length(failed_indices),
  "\n",
  sep = ""
)
cat(
  "Latest run directory: ",
  latest_run_directory,
  "\n",
  sep = ""
)

if (
  length(failed_indices) > 0 &&
  !is.na(latest_run_directory)
) {
  failed_cells_report <- meta_data[
    failed_indices,
    ,
    drop = FALSE
  ]
  failed_cells_report$failed_index <- failed_indices

  failed_report_path <- file.path(
    latest_run_directory,
    "failed_cells.csv"
  )
  write.csv(
    failed_cells_report,
    failed_report_path,
    row.names = FALSE
  )
  cat(
    "Failed-cell report: ",
    failed_report_path,
    "\n",
    sep = ""
  )
}
