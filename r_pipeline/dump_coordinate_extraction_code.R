options(stringsAsFactors = FALSE)

library(devtools)

PACKAGE_DIR <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
OUTPUT_FILE <- paste0(
  "C:/Ole/20260721_CellClassification_v2/r_pipeline/",
  "coordinate_extraction_source_dump.txt"
)

load_all(PACKAGE_DIR)

package_environment <- environment(runFeatureExtraction_NEW)
object_names <- ls(package_environment, all.names = TRUE)

function_names <- object_names[vapply(
  object_names,
  function(object_name) {
    is.function(get(object_name, envir = package_environment))
  },
  logical(1)
)]

search_terms <- c(
  "Extracting Coordinates",
  "Extraction Failed",
  "custom_features_mtx"
)

matching_functions <- function_names[vapply(
  function_names,
  function(function_name) {
    function_text <- paste(
      deparse(body(get(function_name, envir = package_environment))),
      collapse = "\n"
    )
    any(vapply(
      search_terms,
      grepl,
      logical(1),
      x = function_text,
      fixed = TRUE
    ))
  },
  logical(1)
)]

if (!"runFeatureExtraction_NEW" %in% matching_functions) {
  matching_functions <- c(
    "runFeatureExtraction_NEW",
    matching_functions
  )
}

matching_functions <- unique(matching_functions)

output_lines <- c(
  paste0("Package directory: ", PACKAGE_DIR),
  paste0("Matching functions: ", paste(matching_functions, collapse = ", ")),
  ""
)

for (function_name in matching_functions) {
  function_object <- get(
    function_name,
    envir = package_environment
  )

  output_lines <- c(
    output_lines,
    strrep("=", 80),
    paste0("FUNCTION: ", function_name),
    strrep("=", 80),
    capture.output(print(function_object)),
    ""
  )
}

writeLines(
  output_lines,
  con = OUTPUT_FILE,
  useBytes = TRUE
)

cat(
  "\nCoordinate-extraction source written to:\n",
  OUTPUT_FILE,
  "\n",
  sep = ""
)
