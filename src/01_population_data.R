# UN WPP population counts by age and sex, joined onto the survey ground truth.
#
# Produces the three legacy files the Python stages still read:
#   files/population_count.csv        raw counts by band and sex        (06_facebook_data.py)
#   files/un_pop_<span>.csv           band totals + female/male ratios
#   groundtruth_offline_predictors_pop.csv   ground truth + population  (03_internet_indicator_cleaning)
# plus files/new_groundtruth_national_data.csv, the long-format outcome file.
#
# For the *full* 1950-2023 panel that 13/15/16_*.py read, use 01_01_un_population_wpp.py instead —
# this script keeps the original narrow scope (recent years, no both-sexes column).
#
# Run from the project root:  Rscript src/01_population_data.R

library(tidyverse)
library(janitor)
library(readxl)
library(here)
source(here("src", "R_params.R"))
source(here("src", "R_utils.R"))
check_paths()

# WPP2022 was the edition this script was written against; those workbooks are no longer on
# Dropbox, so it now reads the WPP2024 raws in data/processed/un_pop/raw. See doc/decisions.md.
WPP_EDITION <- "WPP2024"
WPP_SHEET <- "Estimates"
YEAR_FROM <- 2001  # kept as ">" below, so the panel starts at 2002

# CONVENTIONS.md §1: `data/raw` is immutable. Writing the regenerated files straight back over the
# shipped ones would silently re-base every downstream model, so outputs are date-stamped (§3).
# Set to TRUE only when deliberately promoting this run to be the new pipeline input.
OVERWRITE_SHIPPED <- FALSE

out_path <- function(name) {
  if (OVERWRITE_SHIPPED) file.path(RAW, paste0(name, ".csv")) else stamped(name)
}

ground_truth <- read_csv(file.path(RAW, "groundtruth_offline_predictors.csv"),
                         show_col_types = FALSE)

#' Read one WPP single-age workbook and collapse it to the 27 age bands.
#'
#' The workbooks carry a 12-row preamble, so row 13 holds the real header; `row_to_names()`
#' promotes it after the preamble is dropped. Columns 12:112 are single ages 0..100+, which is
#' why every `c_across()` range below is an age + 12.
#'
#' @param path  workbook path
#' @param suffix  "f" or "m" — appended to each band name
read_population <- function(path, suffix) {
  data <- read_excel(path, sheet = WPP_SHEET, .name_repair = "minimal")
  data <- data[-c(1:11), ]
  data <- data %>% row_to_names(row_number = 1)

  data <- data %>%
    filter(Type == "Country/Area", Year > YEAR_FROM) %>%
    mutate(survey_start = as.numeric(Year))

  char_cols <- c(0:99, "100+")
  data <- data %>% mutate(across(all_of(char_cols), as.numeric))

  # WPP publishes thousands; the pipeline works in persons.
  population <- data %>%
    rowwise() %>%
    mutate("18_inf" = sum(c_across(30:112)) * 1000,
           "14_15"  = sum(c_across(26:27)) * 1000,
           "15_16"  = sum(c_across(27:28)) * 1000,
           "16_17"  = sum(c_across(28:29)) * 1000,
           "17_18"  = sum(c_across(29:30)) * 1000,
           "18_19"  = sum(c_across(30:31)) * 1000,
           "13_14"  = sum(c_across(25:26)) * 1000,
           "15_19"  = sum(c_across(27:31)) * 1000,
           "20_24"  = sum(c_across(32:36)) * 1000,
           "25_29"  = sum(c_across(37:41)) * 1000,
           "30_34"  = sum(c_across(42:46)) * 1000,
           "35_39"  = sum(c_across(47:51)) * 1000,
           "40_44"  = sum(c_across(52:56)) * 1000,
           "45_49"  = sum(c_across(57:61)) * 1000,
           "50_54"  = sum(c_across(62:66)) * 1000,
           "55_59"  = sum(c_across(67:71)) * 1000,
           "60_64"  = sum(c_across(72:76)) * 1000,
           "18_23"  = sum(c_across(30:35)) * 1000,
           "20_inf" = sum(c_across(32:112)) * 1000,
           "20_64"  = sum(c_across(32:76)) * 1000,
           "21_inf" = sum(c_across(33:112)) * 1000,
           "25_inf" = sum(c_across(37:112)) * 1000,
           "25_49"  = sum(c_across(37:61)) * 1000,
           "25_64"  = sum(c_across(37:76)) * 1000,
           "50_inf" = sum(c_across(62:112)) * 1000,
           "60_inf" = sum(c_across(72:112)) * 1000,
           "65_inf" = sum(c_across(77:112)) * 1000) %>%
    ungroup()

  population %>%
    select(`ISO3 Alpha-code`, `Region, subregion, country or area *`, c(113:140)) %>%
    rename_with(~ paste0(.x, "_", suffix), -c(1:3))
}

wpp_file <- function(n, sex) {
  file.path(UN_POP_RAW,
            sprintf("%s_POP_F01_%d_POPULATION_SINGLE_AGE_%s.xlsx", WPP_EDITION, n, sex))
}

population_female <- read_population(wpp_file(3, "FEMALE"), "f")
population_male <- read_population(wpp_file(2, "MALE"), "m")

population_count <- population_male %>%
  full_join(population_female, by = c("ISO3 Alpha-code",
                                      "Region, subregion, country or area *",
                                      "survey_start"))

write_csv(population_count, out_path("population_count"))

# Band total (f + m) and female/male ratio, one pair per band. The original script referenced an
# undefined `population` here, so it could never run past this line.
population <- population_count %>%
  mutate(across(ends_with("_f"),
                ~ . + get(sub("_f$", "_m", cur_column())),
                .names = "{sub('_f$', '', .col)}")) %>%
  mutate(across(ends_with("_f"),
                ~ . / get(sub("_f$", "_m", cur_column())),
                .names = "{sub('_f$', '_r', .col)}"))

population <- population %>%
  select(-ends_with("_f"), -ends_with("_m")) %>%
  rename(iso3 = `ISO3 Alpha-code`,
         country = `Region, subregion, country or area *`)

span <- sprintf("un_pop_%d_%d", min(population$survey_start), max(population$survey_start))
write_csv(population, out_path(span))

# One year rule, shared with 02 and with the Python stages (D32): the survey's own year when the
# population panel has it, otherwise the latest earlier year. Replaces the original 2022 -> 2021
# special case, which handled one year and left every later survey unmatched.
ground_truth_pop <- join_latest_available_year(
  ground_truth, population, key = "iso3", year = "survey_start", used_year_col = "pop_year")

carried <- carried_year_report(ground_truth_pop, key = "iso3", year = "survey_start",
                               used_year_col = "pop_year")
message("population carried forward for ", nrow(carried), " of ", nrow(ground_truth_pop), " rows")

write_csv(ground_truth_pop,
          if (OVERWRITE_SHIPPED) {
            file.path(PIPELINE_ROOT, "groundtruth_offline_predictors_pop.csv")
          } else {
            file.path(RAW, sprintf("groundtruth_offline_predictors_pop_%s.csv", STAMP))
          })


## make new national data file
data <- read_csv(file.path(RAW, "internet_mobile_indicator_clean.csv"), show_col_types = FALSE)

data_long <- data %>%
  select(iso3, survey_start, survey_type, internet_use_in_12_months_men,
         internet_use_in_12_months_wom, used_internet_past12months_fm_perc_ratio,
         owns_mobile_phone_men, owns_mobile_phone_wom, owns_mobile_phone_fm_perc_ratio) %>%
  pivot_longer(cols = c(internet_use_in_12_months_men, internet_use_in_12_months_wom,
                        used_internet_past12months_fm_perc_ratio, owns_mobile_phone_wom,
                        owns_mobile_phone_men, owns_mobile_phone_fm_perc_ratio),
               names_to = "outcome",
               values_to = "observed") %>%
  mutate(outcome = case_when(
    outcome == "internet_use_in_12_months_men" ~ "internet_men",
    outcome == "internet_use_in_12_months_wom" ~ "internet_women",
    outcome == "used_internet_past12months_fm_perc_ratio" ~ "internet_fm_ratio",
    outcome == "owns_mobile_phone_wom" ~ "mobile_women",
    outcome == "owns_mobile_phone_men" ~ "mobile_men",
    outcome == "owns_mobile_phone_fm_perc_ratio" ~ "mobile_fm_ratio"
  )) %>%
  rename(gid_0 = iso3,
         survey_year = survey_start,
         source = survey_type)

data_long <- na.omit(data_long)

write_csv(data_long, out_path("new_groundtruth_national_data"))

message("done — outputs written to ", RAW,
        if (OVERWRITE_SHIPPED) " (SHIPPED FILES OVERWRITTEN)" else sprintf(" (date-stamped %s)", STAMP))
