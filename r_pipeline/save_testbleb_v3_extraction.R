options(stringsAsFactors = FALSE)

# Reuse the tested save script and point it at the new TestBleb-v3 run.
script_dir <- "C:/Ole/20260721_CellClassification_v2/r_pipeline"
source_path <- file.path(script_dir, "save_testbleb_extraction.R")

if (!file.exists(source_path)) {
  stop("Source save script not found: ", source_path)
}

source_code <- paste(
  readLines(source_path, warn = FALSE, encoding = "UTF-8"),
  collapse = "\n"
)

old_root_block <- paste0(
  "ANALYSIS_ROOT <- paste0(\n",
  "  \"C:/Ole/20260721_CellClassification_v2/\",\n",
  "  \"20260726_TestBleb/analysis_net129\"\n",
  ")"
)
new_root_block <- paste0(
  "ANALYSIS_ROOT <- ",
  "\"C:/Ole/20260721_CellClassification_v2/20260728_TestBleb_v3\""
)

if (!grepl(old_root_block, source_code, fixed = TRUE)) {
  stop(
    "Could not find the expected old ANALYSIS_ROOT block in: ",
    source_path
  )
}

source_code <- sub(
  old_root_block,
  new_root_block,
  source_code,
  fixed = TRUE
)

eval(parse(text = source_code), envir = .GlobalEnv)

