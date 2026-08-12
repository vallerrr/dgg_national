# Unzips freshly downloaded DHS / MICS / PMA archives and files them into the folder layout that
# 02_ground_truth_data_calculation.qmd expects.
#
# This runs *before* the rest of the pipeline and only when new survey archives land. It was
# originally written against a Windows machine (`D:/Study/RA/micro dataset/...`) with `setwd()`
# between blocks; all of that now resolves through src/R_params.R.
#
# IT MOVES AND DELETES FILES IN SHARED DROPBOX. `DRY_RUN` is TRUE by default: every step prints
# what it would do and touches nothing. Read the plan, then set DRY_RUN <- FALSE and rerun.
#
# Run from the project root:  Rscript src/00_data_cleaning_primary.R

library(haven)
library(tidyverse)
library(here)
source(here("src", "R_params.R"))
check_paths()

DRY_RUN <- TRUE

# Which download folder to process. Point this at the staging folder holding the new archives;
# `dhs/wave8+` is where the 2023-2025 refresh landed.
INBOX <- DHS_WAVE8_PLUS


# ---- primitives -----------------------------------------------------------------------------
# Each wraps one destructive operation so DRY_RUN has a single place to intervene.

say <- function(...) message(if (DRY_RUN) "[dry-run] " else "[apply]   ", ...)

do_unzip <- function(zip, exdir) {
  say("unzip ", basename(zip), " -> ", exdir)
  if (!DRY_RUN) unzip(zip, exdir = exdir)
}

do_remove <- function(paths) {
  if (!length(paths)) return(invisible(NULL))
  say("remove ", length(paths), " file(s): ", paste(head(basename(paths), 5), collapse = ", "),
      if (length(paths) > 5) " ..." else "")
  if (!DRY_RUN) file.remove(paths)
}

do_move <- function(from, to) {
  say("move ", basename(from), " -> ", to)
  if (!DRY_RUN) {
    dir.create(dirname(to), showWarnings = FALSE, recursive = TRUE)
    if (!file.rename(from, to)) warning("failed to move ", from)
  }
}


#' Unzip every archive in `dir` in place and drop everything that is not survey microdata.
#'
#' @param dir  folder holding the .zip downloads
#' @param keep regex of extensions worth keeping (Stata microdata plus the shipped .R readers)
unpack_archives <- function(dir, keep = "\\.(DTA|R)$") {
  zips <- list.files(dir, pattern = "\\.zip$", full.names = TRUE)
  for (z in zips) do_unzip(z, exdir = dir)

  extracted <- list.files(dir, full.names = TRUE, recursive = TRUE)
  do_remove(extracted[!grepl(keep, extracted, ignore.case = TRUE)])
}


# DHS filenames encode the round in the recode letter: IR7/MR7 are wave 7, IR8/MR8 wave 8.
DHS_ROUND_DIRS <- list(
  "IR7" = file.path(DHS_WAVE7, "women"), "MR7" = file.path(DHS_WAVE7, "men"),
  "IR8" = file.path(DHS_WAVE8, "women"), "MR8" = file.path(DHS_WAVE8, "men")
)

#' Sort DHS microdata into round/sex folders based on the recode tag in the filename.
sort_dhs_by_round <- function(dir, round_dirs = DHS_ROUND_DIRS) {
  for (path in list.files(dir, full.names = TRUE, recursive = TRUE)) {
    tag <- names(round_dirs)[vapply(names(round_dirs),
                                    function(k) grepl(k, basename(path)), logical(1))]
    if (!length(tag)) next
    do_move(path, file.path(round_dirs[[tag[1]]], basename(path)))
  }
}


# DHS two-letter country codes -> folder names.
country_codes <- c(AF = "Afghanistan", AL = "Albania", AO = "Angola", AM = "Armenia",
                   AZ = "Azerbaijan", BD = "Bangladesh", BJ = "Benin", BO = "Bolivia",
                   BT = "Botswana", BR = "Brazil", BF = "Burkina_Faso", BU = "Burundi",
                   KH = "Cambodia", CM = "Cameroon", CV = "Cape_Verde",
                   CF = "Central_African_Republic", TD = "Chad", CO = "Colombia",
                   KM = "Comoros", CG = "Congo", CD = "Congo_Democratic_Republic",
                   CI = "Cote_d'Ivoire", DR = "Dominican_Republic", EC = "Ecuador",
                   EG = "Egypt", ES = "El_Salvador", EK = "Equatorial_Guinea",
                   ER = "Eritrea", ET = "Ethiopia", GA = "Gabon", GM = "Gambia",
                   GH = "Ghana", GU = "Guatemala", GN = "Guinea", GY = "Guyana",
                   HT = "Haiti", HN = "Honduras", IA = "India", ID = "Indonesia",
                   JO = "Jordan", KK = "Kazakhstan", KE = "Kenya", KY = "Kyrgyz_Republic",
                   LA = "Lao_People's_Democratic_Republic", LS = "Lesotho",
                   LB = "Liberia", MD = "Madagascar", MW = "Malawi", MV = "Maldives",
                   ML = "Mali", MR = "Mauritania", MX = "Mexico", MB = "Moldova",
                   MA = "Morocco", MZ = "Mozambique", MM = "Myanmar", NM = "Namibia",
                   NP = "Nepal", NC = "Nicaragua", NI = "Niger", NG = "Nigeria",
                   OS = "Nigeria_Ondo_State", PK = "Pakistan", PY = "Paraguay",
                   PE = "Peru", PH = "Philippines", RW = "Rwanda", WS = "Samoa",
                   ST = "Sao_Tome_and_Principe", SN = "Senegal", SL = "Sierra_Leone",
                   ZA = "South_Africa", LK = "Sri_Lanka", SD = "Sudan", SZ = "Swaziland",
                   TJ = "Tajikistan", TZ = "Tanzania", TH = "Thailand", TL = "Timor-Leste",
                   TG = "Togo", TT = "Trinidad_and_Tobago", TN = "Tunisia", TR = "Turkey",
                   TM = "Turkmenistan", UG = "Uganda", UA = "Ukraine", UZ = "Uzbekistan",
                   VN = "Vietnam", YE = "Yemen", ZM = "Zambia", ZW = "Zimbabwe")

#' Move DHS files into per-country folders, keyed on the first two letters of the filename.
sort_dhs_by_country <- function(base_dir, codes = country_codes) {
  files <- list.files(base_dir, pattern = "\\.DTA$", full.names = TRUE,
                      recursive = TRUE, ignore.case = TRUE)
  for (path in files) {
    name <- basename(path)
    country <- codes[substr(name, 1, 2)]
    if (is.na(country)) {
      message("no matching country for code ", substr(name, 1, 2), " - ", name)
      next
    }
    folder <- gsub("'", "", gsub(" ", "_", country))
    do_move(path, file.path(base_dir, folder, name))
  }
}


#' Normalise the MICS folder names (lowercase, underscores) and flatten the one nested level
#' the MICS archives ship with.
tidy_mics_folders <- function(base_dir = MICS_DIR) {
  for (dir_path in list.dirs(base_dir, full.names = TRUE, recursive = FALSE)) {
    new_path <- gsub(" ", "_", tolower(dir_path))
    if (new_path != dir_path) do_move(dir_path, new_path)
  }

  for (parent in list.dirs(base_dir, full.names = TRUE, recursive = FALSE)) {
    for (sub in list.dirs(parent, full.names = TRUE, recursive = FALSE)) {
      for (f in list.files(sub, full.names = TRUE)) {
        do_move(f, file.path(parent, basename(f)))
      }
      say("drop empty folder ", sub)
      if (!DRY_RUN) unlink(sub, recursive = TRUE)
    }
  }
}


#' Which files lack a usable variable — the check 02 repeats before every `process_*` call.
#'
#' @return the subset of `file_list` where `variable_name` is absent or entirely NA
check_files_for_variable <- function(file_list, variable_name) {
  files_without_variable <- c()
  for (file in file_list) {
    data <- tryCatch(read_dta(file, n_max = 1000), error = function(e) NULL)
    if (!is.null(data)) {
      if (!variable_name %in% names(data) || all(is.na(data[[variable_name]]))) {
        files_without_variable <- c(files_without_variable, file)
      }
    }
  }
  files_without_variable
}


# ---- what a refresh actually runs -------------------------------------------------------------
# Uncomment the steps that apply to the batch you just downloaded.

if (sys.nframe() == 0) {
  message("INBOX: ", INBOX)
  message("DRY_RUN = ", DRY_RUN, if (DRY_RUN) "  (nothing will be modified)" else "")

  # unpack_archives(INBOX)
  # sort_dhs_by_country(INBOX)
  # tidy_mics_folders()

  # Coverage check on what is already filed, mirroring 02's per-wave guards.
  ir <- list.files(INBOX, pattern = "IR.*\\.DTA$", recursive = TRUE,
                   full.names = TRUE, ignore.case = TRUE)
  mr <- list.files(INBOX, pattern = "MR.*\\.DTA$", recursive = TRUE,
                   full.names = TRUE, ignore.case = TRUE)
  message("women's recodes: ", length(ir), " | men's recodes: ", length(mr))
  if (length(ir)) message("  ", paste(basename(ir), collapse = "\n  "))
}
