# Run generated automatic SWCs through the same two scEvoView parser stages
# that produce the manual swc_final files used in tracing comparisons.

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(paste(
    "Aufruf: Rscript normalize_automatic_swc_for_comparison.R",
    "<cell_root> <output_dir> [pixel_width_um]"
  ))
}

cell_root <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
pixel_width_um <- if (length(args) >= 3L) as.numeric(args[[3]]) else 0.2875008
if (!is.finite(pixel_width_um) || pixel_width_um <= 0) {
  stop("pixel_width_um muss eine positive Zahl sein.")
}

script_arg <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_arg[grepl("^--file=", script_arg)][1])
script_dir <- dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE))
source(file.path(script_dir, "restore_mats_swc_types.R"))

suppressPackageStartupMessages(library(devtools))
required_packages <- c("strex", "stringr")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Fehlende R-Pakete: ", paste(missing_packages, collapse = ", "))
}
invisible(lapply(required_packages, function(package_name) {
  suppressPackageStartupMessages(library(package_name, character.only = TRUE))
}))

package_dir <- paste0(
  "X:/Robert_temp/data_combined/code/pipelines/",
  "extract_features_for_framework_new/r_package2"
)
if (!dir.exists(package_dir)) {
  stop("scEvoView-Parserpaket fehlt: ", package_dir)
}

cell_dirs <- list.dirs(cell_root, recursive = FALSE, full.names = TRUE)
cell_dirs <- cell_dirs[grepl("^cell[0-9]+$", basename(cell_dirs))]
cell_dirs <- cell_dirs[order(as.integer(sub("^cell", "", basename(cell_dirs))))]
if (!length(cell_dirs)) {
  stop("Keine cell####-Ordner gefunden: ", cell_root)
}

swc_paths <- file.path(cell_dirs, "seg-000.swc")
missing_swc <- swc_paths[!file.exists(swc_paths)]
if (length(missing_swc)) {
  stop("Fehlende seg-000.swc:\n", paste(missing_swc, collapse = "\n"))
}

cell_ids <- basename(cell_dirs)
source_rows <- data.frame(
  id = cell_ids,
  swc = swc_paths,
  pixel_width = pixel_width_um,
  pixel_height = pixel_width_um,
  pixel_depth = 1.0,
  pixel_unit = "microns",
  is_swc_scaled = FALSE,
  cellfolder = cell_dirs,
  stringsAsFactors = FALSE
)

load_all(package_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
meta_data <- parseSWCFilesStepOne(source_rows, output_dir)
if ("pixel_width_real" %in% names(meta_data)) {
  meta_data$pixel_width_real <- pixel_width_um
}
if ("pixel_height_real" %in% names(meta_data)) {
  meta_data$pixel_height_real <- pixel_width_um
}
if ("pixel_depth_real" %in% names(meta_data)) {
  meta_data$pixel_depth_real <- 1.0
}
if ("pixel_unit_real" %in% names(meta_data)) {
  meta_data$pixel_unit_real <- "microns"
}
meta_data <- parseSWCFilesStepTwo(meta_data, output_dir)
meta_data <- restoreMatsGeneratedSwcTypes(meta_data)

metadata_path <- file.path(output_dir, "meta_data.csv")
utils::write.csv(meta_data, metadata_path, row.names = FALSE)

final_paths <- file.path(output_dir, "swc_final", paste0(cell_ids, ".swc"))
missing_final <- final_paths[!file.exists(final_paths)]
if (length(missing_final)) {
  stop("scEvoView hat nicht alle swc_final-Dateien erzeugt.")
}

cat("\nAUTOMATIC SWC NORMALIZATION COMPLETE\n")
cat("Cells: ", length(cell_ids), "\n", sep = "")
cat("Pixel size: ", pixel_width_um, " um/px\n", sep = "")
cat("swc_final: ", file.path(output_dir, "swc_final"), "\n", sep = "")
