# R-side companion to params.py — the paths the R and Quarto stages (00-04) read and write.
#
# CONVENTIONS.md §1: nothing here is an absolute path. Everything resolves through the gitignored
# `data/` symlinks into the shared Dropbox tree (see data/README.md), so the scripts stay portable.
#
# Source it from a script or a Quarto chunk:
#     source(here::here("src", "R_params.R"))

library(here)

# ---- the three symlink roots (mirror params.RAW / PROCESSED / EXTERNAL) --------------------------
RAW <- here("data", "raw")             # -> new_national_pipeline_files/files   (source inputs)
PROCESSED <- here("data", "processed") # -> new_national_pipeline_files/data    (derived)
EXTERNAL <- here("data", "external")   # -> ~/Dropbox/dgg_research              (upstream tree)

# `data/raw` is the *files* subfolder; a few legacy outputs land one level up, next to it.
PIPELINE_ROOT <- file.path(EXTERNAL, "national", "data_refresh", "new_national_pipeline_files")

# ---- survey microdata ---------------------------------------------------------------------------
# The DHS/MICS/ITU microdata lives under adolescent_modelling, not under data_refresh — this is the
# single most common reason the 02/03 Quarto files fail to run on a fresh machine.
SURVEY_ROOT <- file.path(EXTERNAL, "national", "adolescent_modelling", "data_refresh")
DHS_DIR <- file.path(SURVEY_ROOT, "dhs")
MICS_DIR <- file.path(SURVEY_ROOT, "mics")
ITU_DIR <- file.path(SURVEY_ROOT, "itu")

# DHS survey rounds. `wave8+` holds the 2023-2025 surveys added in the 2026-08 refresh; earlier
# versions of 02_ground_truth_data_calculation.qmd called this folder `update`.
DHS_WAVE7 <- file.path(DHS_DIR, "wave7")
DHS_WAVE8 <- file.path(DHS_DIR, "wave8")
DHS_WAVE8_PLUS <- file.path(DHS_DIR, "wave8+")
DHS_CONTINUOUS <- file.path(DHS_DIR, "continuous")

# ---- UN World Population Prospects ---------------------------------------------------------------
UN_POP <- file.path(PROCESSED, "un_pop")   # WPP2024 raws in raw/, processed panels alongside
UN_POP_RAW <- file.path(UN_POP, "raw")
# The population panel the Python stages read (params.RAW / 'un_1950_2023_processed.csv').
UN_POP_PROCESSED <- file.path(RAW, "un_1950_2023_processed.csv")

# ---- outputs -------------------------------------------------------------------------------------
# CONVENTIONS.md §3: generated files carry their save date, so a rerun never silently replaces the
# artefact an earlier analysis was built on.
STAMP <- format(Sys.Date(), "%Y%m%d")
stamped <- function(name, ext = "csv") file.path(RAW, sprintf("%s_%s.%s", name, STAMP, ext))

# Where 02 writes the harmonised survey outcomes.
GROUNDTRUTH_OUT <- file.path(EXTERNAL, "national", "adolescent_modelling",
                             sprintf("update_full_groundtruth_%s.csv", STAMP))

# ---- guard ---------------------------------------------------------------------------------------
# Fail loudly and early rather than three joins later with an empty data frame.
check_paths <- function() {
  required <- c(RAW = RAW, PROCESSED = PROCESSED, EXTERNAL = EXTERNAL,
                DHS_DIR = DHS_DIR, MICS_DIR = MICS_DIR, ITU_DIR = ITU_DIR)
  missing <- required[!dir.exists(required)]
  if (length(missing)) {
    stop("missing data paths — recreate the Dropbox symlinks (see data/README.md):\n  ",
         paste(names(missing), unname(missing), sep = " -> ", collapse = "\n  "), call. = FALSE)
  }
  invisible(TRUE)
}
