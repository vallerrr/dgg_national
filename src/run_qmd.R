# Runs a .qmd's R chunks without Quarto.
#
# `quarto render` is the normal way to run 02 and 03, but Quarto is not installed on every
# machine here and these two files produce CSVs rather than a report — the rendered HTML is a
# by-product. `knitr::purl()` extracts the chunks in order and this sources the result, which is
# the same execution path minus the document.
#
# Usage:  Rscript src/run_qmd.R src/02_ground_truth_data_calculation.qmd
#
# Chunks marked `eval = FALSE` (or gated on a flag, like 03's map chunks) are dropped by purl,
# so a chunk you want skipped should be gated rather than commented out.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript src/run_qmd.R <file.qmd>", call. = FALSE)

qmd <- normalizePath(args[1], mustWork = TRUE)
script <- tempfile(fileext = ".R")
knitr::purl(qmd, output = script, quiet = TRUE, documentation = 0L)

message("running ", basename(qmd), " (", length(readLines(script)), " lines of R)")
source(script, echo = TRUE, max.deparse.length = 200)
