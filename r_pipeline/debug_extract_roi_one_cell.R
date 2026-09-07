options(stringsAsFactors = FALSE)
options(warn = 1)

library(devtools)

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
OUTPUT_DIR <- paste0(
  "C:/Ole/20260721_CellClassification_v2/r_pipeline/",
  "20260724_r_pipeline_output_fixed"
)
METADATA_PATH <- file.path(OUTPUT_DIR, "meta_data.csv")
DEBUG_DIR <- file.path(OUTPUT_DIR, "coordinate_debug")
ERROR_REPORT <- file.path(
  OUTPUT_DIR,
  "extract_roi_error_details.txt"
)

load_all(PACKAGE_DIR)

meta_data <- read.csv(
  METADATA_PATH,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (nrow(meta_data) < 1) {
  stop("The metadata file contains no cells: ", METADATA_PATH)
}

cell <- meta_data[1, , drop = FALSE]
dir.create(DEBUG_DIR, recursive = TRUE, showWarnings = FALSE)

package_environment <- environment(runFeatureExtraction_NEW)
extract_roi <- get(
  "extractRoi",
  envir = package_environment,
  inherits = TRUE
)

required_columns <- c(
  "id",
  "soma_roi",
  "cellframe",
  "swc_file",
  "pixel_width",
  "pixel_height"
)
missing_columns <- setdiff(required_columns, names(cell))

if (length(missing_columns) > 0) {
  stop(
    "Missing metadata columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

cat("DIRECT extractRoi DEBUG\n")
cat("ID:           ", cell$id[[1]], "\n", sep = "")
cat("Soma ROI:     ", cell$soma_roi[[1]], "\n", sep = "")
cat("SWC:          ", cell$swc_file[[1]], "\n", sep = "")
cat("Frame:        ", cell$cellframe[[1]], "\n", sep = "")
cat("Pixel width:  ", cell$pixel_width[[1]], "\n", sep = "")
cat("Pixel height: ", cell$pixel_height[[1]], "\n", sep = "")
cat("ROI dump:     ", DEBUG_DIR, "\n\n", sep = "")

result <- tryCatch(
  extract_roi(
    cell$soma_roi[[1]],
    cell$cellframe[[1]],
    cell$swc_file[[1]],
    cell$pixel_width[[1]],
    cell$pixel_height[[1]],
    DEBUG_DIR
  ),
  error = function(error_condition) {
    call_stack <- capture.output(print(sys.calls()))
    function_source <- capture.output(print(extract_roi))

    report <- c(
      "DIRECT extractRoi ERROR",
      paste0("Timestamp: ", Sys.time()),
      paste0("ID: ", cell$id[[1]]),
      paste0("Soma ROI: ", cell$soma_roi[[1]]),
      paste0("SWC: ", cell$swc_file[[1]]),
      paste0(
        "ROI exists: ",
        file.exists(cell$soma_roi[[1]])
      ),
      paste0(
        "SWC exists: ",
        file.exists(cell$swc_file[[1]])
      ),
      paste0(
        "Error class: ",
        paste(class(error_condition), collapse = ", ")
      ),
      paste0(
        "Error message: ",
        conditionMessage(error_condition)
      ),
      paste0(
        "Error call: ",
        paste(deparse(conditionCall(error_condition)), collapse = " ")
      ),
      "",
      "CALL STACK",
      call_stack,
      "",
      "extractRoi SOURCE",
      function_source
    )

    writeLines(
      report,
      con = ERROR_REPORT,
      useBytes = TRUE
    )

    cat(
      "\nThe original extractRoi error was captured:\n",
      conditionMessage(error_condition),
      "\n\nFull report:\n",
      ERROR_REPORT,
      "\n",
      sep = ""
    )

    return(structure(
      list(condition = error_condition),
      class = "extract_roi_debug_failure"
    ))
  }
)

if (inherits(result, "extract_roi_debug_failure")) {
  stop(
    "Direct extractRoi test failed. ",
    "See extract_roi_error_details.txt",
    call. = FALSE
  )
}

SUCCESS_REPORT <- file.path(
  OUTPUT_DIR,
  "extract_roi_success_details.txt"
)

writeLines(
  c(
    "DIRECT extractRoi SUCCESS",
    paste0("Timestamp: ", Sys.time()),
    paste0("ID: ", cell$id[[1]]),
    paste0(
      "Returned names: ",
      paste(names(result), collapse = ", ")
    )
  ),
  con = SUCCESS_REPORT,
  useBytes = TRUE
)

cat(
  "\nDirect extractRoi test succeeded.\nReport:\n",
  SUCCESS_REPORT,
  "\n",
  sep = ""
)
