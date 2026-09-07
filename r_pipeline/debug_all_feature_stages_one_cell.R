options(stringsAsFactors = FALSE)
options(warn = 1)

library(devtools)

pipeline_packages <- c(
  "cli",
  "nat",
  "ggplot2",
  "spatstat",
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

missing_packages <- pipeline_packages[
  !vapply(
    pipeline_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  stop(
    "Missing packages: ",
    paste(missing_packages, collapse = ", ")
  )
}

invisible(lapply(
  pipeline_packages,
  function(package_name) {
    suppressPackageStartupMessages(
      library(package_name, character.only = TRUE)
    )
  }
))

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
    " function(s) patched",
    if (length(patched_functions) > 0) {
      paste0(
        " (",
        paste(patched_functions, collapse = ", "),
        ")"
      )
    } else {
      ""
    },
    "\n",
    sep = ""
  )

  invisible(patched_functions)
}

patch_nat_igraph_compatibility()

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
CODE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new"
)
OUTPUT_DIR <- paste0(
  "C:/Ole/20260721_CellClassification_v2/r_pipeline/",
  "20260724_r_pipeline_output_fixed"
)
METADATA_PATH <- file.path(OUTPUT_DIR, "meta_data.csv")
ERROR_REPORT <- file.path(
  OUTPUT_DIR,
  "feature_stage_error_details.txt"
)
SUCCESS_REPORT <- file.path(
  OUTPUT_DIR,
  "feature_stage_success_details.txt"
)

load_all(PACKAGE_DIR)

meta_data <- read.csv(
  METADATA_PATH,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (nrow(meta_data) < 1) {
  stop("No cells in metadata: ", METADATA_PATH)
}

cell <- meta_data[1, , drop = FALSE]

feature_object <- newFeatureExtraction(
  name = "FeatureExtraction",
  CODE_DIR
)
feature_object <- addSaveDir(
  feature_object,
  OUTPUT_DIR
)
feature_object <- addMetaData(
  feature_object,
  cell
)

debug_root <- file.path(
  OUTPUT_DIR,
  paste0(
    "stage_debug_",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )
)
roi_dump <- file.path(debug_root, "roi_dump")
lmeasure_dump <- file.path(debug_root, "lmeasure_dump")

dir.create(
  roi_dump,
  recursive = TRUE,
  showWarnings = FALSE
)
dir.create(
  lmeasure_dump,
  recursive = TRUE,
  showWarnings = FALSE
)

package_environment <- environment(
  runFeatureExtraction_NEW
)

get_package_function <- function(function_name) {
  get(
    function_name,
    envir = package_environment,
    inherits = TRUE
  )
}

stage_log <- c(
  "ONE-CELL FEATURE-STAGE DEBUG",
  paste0("Timestamp: ", Sys.time()),
  paste0("ID: ", cell$id[[1]]),
  ""
)

run_stage <- function(
  stage_name,
  stage_function,
  operation
) {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("STAGE: ", stage_name, "\n", sep = "")
  cat(strrep("=", 70), "\n", sep = "")

  captured_error <- NULL

  value <- tryCatch(
    operation(),
    error = function(error_condition) {
      captured_error <<- error_condition
      NULL
    }
  )

  if (!is.null(captured_error)) {
    error_report <- c(
      stage_log,
      paste0("FAILED STAGE: ", stage_name),
      paste0(
        "Error class: ",
        paste(class(captured_error), collapse = ", ")
      ),
      paste0(
        "Error message: ",
        conditionMessage(captured_error)
      ),
      paste0(
        "Error call: ",
        paste(
          deparse(conditionCall(captured_error)),
          collapse = " "
        )
      ),
      "",
      paste0(stage_name, " SOURCE"),
      capture.output(print(stage_function))
    )

    writeLines(
      error_report,
      con = ERROR_REPORT,
      useBytes = TRUE
    )

    cat(
      "FAILED: ",
      conditionMessage(captured_error),
      "\nReport: ",
      ERROR_REPORT,
      "\n",
      sep = ""
    )

    stop(
      "Feature-stage debug stopped at ",
      stage_name,
      call. = FALSE
    )
  }

  result_description <- paste(
    class(value),
    collapse = ", "
  )

  if (is.list(value)) {
    result_description <- paste0(
      result_description,
      "; names=",
      paste(names(value), collapse = ",")
    )
  } else if (
    is.matrix(value) ||
    is.data.frame(value)
  ) {
    result_description <- paste0(
      result_description,
      "; dimensions=",
      paste(dim(value), collapse = "x")
    )
  }

  stage_log <<- c(
    stage_log,
    paste0("SUCCESS: ", stage_name),
    paste0("Result: ", result_description),
    ""
  )

  cat("SUCCESS\n")
  value
}

cur_id <- cell$id[[1]]
cur_swc <- cell$swc_file[[1]]
cur_roi <- cell$soma_roi[[1]]
cur_frame <- cell$cellframe[[1]]
cur_px_width <- cell$pixel_width[[1]]
cur_px_height <- cell$pixel_height[[1]]

extract_roi <- get_package_function("extractRoi")
extract_soma_shape <- get_package_function(
  "extractSomaShape"
)
extract_chull_shape <- get_package_function(
  "extractChullShape"
)
extract_sholl <- get_package_function("extractSholl")
calculate_custom_features <- get_package_function(
  "calculate_features"
)
extract_lmeasure <- get_package_function(
  "extractLMeasure"
)

coordinate_result <- run_stage(
  "extractRoi",
  extract_roi,
  function() {
    extract_roi(
      cur_roi,
      cur_frame,
      cur_swc,
      cur_px_width,
      cur_px_height,
      roi_dump
    )
  }
)

soma_result <- run_stage(
  "extractSomaShape",
  extract_soma_shape,
  function() {
    extract_soma_shape(
      cur_id,
      coordinate_result$roi_data_scaled
    )
  }
)

chull_result <- run_stage(
  "extractChullShape",
  extract_chull_shape,
  function() {
    extract_chull_shape(
      cur_id,
      coordinate_result$chullpoints
    )
  }
)

sholl_result <- run_stage(
  "extractSholl",
  extract_sholl,
  function() {
    extract_sholl(
      cur_id,
      cur_swc
    )
  }
)

custom_result <- run_stage(
  "calculate_features",
  calculate_custom_features,
  function() {
    calculate_custom_features(
      cur_swc,
      coordinate_result$roi_data_scaled
    )
  }
)

lmeasure_result <- run_stage(
  "extractLMeasure",
  extract_lmeasure,
  function() {
    extract_lmeasure(
      cur_id,
      cur_swc,
      lmeasure_dump,
      feature_object
    )
  }
)

writeLines(
  c(
    stage_log,
    "ALL SIX FEATURE STAGES SUCCEEDED"
  ),
  con = SUCCESS_REPORT,
  useBytes = TRUE
)

cat(
  "\nAll six stages succeeded for ",
  cur_id,
  ".\nReport: ",
  SUCCESS_REPORT,
  "\n",
  sep = ""
)
