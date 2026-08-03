library(tidyverse)
library(janitor)
library(readxl)
library(here)

ground_truth <- read_csv(here("data/national/data_refresh/new_national_pipeline_files/groundtruth_offline_predictors.csv"))

data_female <- read_excel(here("data/national/data_refresh/population_data/WPP2022_POP_F01_3_POPULATION_SINGLE_AGE_fEMALE.xlsx"))

data_female <- data_female[-c(1:11),]

data_female <- data_female %>%
  row_to_names(row_number = 1)

data_female <- data_female %>%
  filter(Type == "Country/Area", Year > 2001) %>%
  mutate(survey_start = as.numeric(Year))

char_cols <- c(0:99,"100+")

data_female <- data_female %>%
  mutate(across(all_of(char_cols), as.numeric))

population_female <- data_female %>%
  rowwise() %>%
  mutate("18_inf_f" = sum(c_across(30:112)),
         "14_15_f" = sum(c_across(26:27)),
         "15_16_f" = sum(c_across(27:28)),
         "16_17_f" = sum(c_across(28:29)),
         "17_18_f" = sum(c_across(29:30)),
         "18_19_f" = sum(c_across(30:31)),
         "13_14_f" = sum(c_across(25:26)),
         "15_19_f" = sum(c_across(27:31)),
         "20_24_f" = sum(c_across(32:36)),
         "25_29_f" = sum(c_across(37:41)),
         "30_34_f" = sum(c_across(42:46)),
         "35_39_f" = sum(c_across(47:51)),
         "40_44_f" = sum(c_across(52:56)),
         "45_49_f" = sum(c_across(57:61)),
         "50_54_f" = sum(c_across(62:66)),
         "55_59_f" = sum(c_across(67:71)),
         "60_64_f" = sum(c_across(72:76)),
         "18_23_f" = sum(c_across(30:35)),
         "20_inf_f" = sum(c_across(32:112)),
         "20_64_f" = sum(c_across(32:76)),
         "21_inf_f" = sum(c_across(33:112)),
         "25_inf_f" = sum(c_across(37:112)),
         "25_49_f" = sum(c_across(37:61)),
         "25_64_f" = sum(c_across(37:76)),
         "50_inf_f" = sum(c_across(62:112)),
         "60_inf_f" = sum(c_across(72:112)),
         "65_inf_f" = sum(c_across(77:112)))

population_female <- population_female %>%
  select(`ISO3 Alpha-code`,`Region, subregion, country or area *`,c(113:140))


data_male <- read_excel(here("data/national/data_refresh/population_data/WPP2022_POP_F01_2_POPULATION_SINGLE_AGE_MALE.xlsx"))

data_male <- data_male[-c(1:11),]

data_male <- data_male %>%
  row_to_names(row_number = 1)

data_male <- data_male %>%
  filter(Type == "Country/Area", Year > 2001) %>%
  mutate(survey_start = as.numeric(Year))

char_cols <- c(0:99,"100+")

data_male <- data_male %>%
  mutate(across(all_of(char_cols), as.numeric))

population_male <- data_male %>%
  rowwise() %>%
  mutate("18_inf_m" = sum(c_across(30:112)),
         "14_15_m" = sum(c_across(26:27)),
         "15_16_m" = sum(c_across(27:28)),
         "16_17_m" = sum(c_across(28:29)),
         "17_18_m" = sum(c_across(29:30)),
         "18_19_m" = sum(c_across(30:31)),
         "13_14_m" = sum(c_across(25:26)),
         "15_19_m" = sum(c_across(27:31)),
         "20_24_m" = sum(c_across(32:36)),
         "25_29_m" = sum(c_across(37:41)),
         "30_34_m" = sum(c_across(42:46)),
         "35_39_m" = sum(c_across(47:51)),
         "40_44_m" = sum(c_across(52:56)),
         "45_49_m" = sum(c_across(57:61)),
         "50_54_m" = sum(c_across(62:66)),
         "55_59_m" = sum(c_across(67:71)),
         "60_64_m" = sum(c_across(72:76)),
         "18_23_m" = sum(c_across(30:35)),
         "20_inf_m" = sum(c_across(32:112)),
         "20_64_m" = sum(c_across(32:76)),
         "21_inf_m" = sum(c_across(33:112)),
         "25_inf_m" = sum(c_across(37:112)),
         "25_49_m" = sum(c_across(37:61)),
         "25_64_m" = sum(c_across(37:76)),
         "50_inf_m" = sum(c_across(62:112)),
         "60_inf_m" = sum(c_across(72:112)),
         "65_inf_m" = sum(c_across(77:112)))

population_male <- population_male %>%
  select(`ISO3 Alpha-code`,`Region, subregion, country or area *`,c(113:140))

population_count <- population_male %>%
  full_join(population_female)

write_csv(population_count, file = here("data/national/data_refresh/new_national_pipeline_files/files/population_count.csv"))

population <- population %>%
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

write_csv(population, here("data/national/data_refresh/new_national_pipeline_files/files/un_pop_2001_2021.csv"))

ground_truth_pop <- ground_truth %>%
  left_join(population, by = c("iso3", "survey_start")) 

ground_truth_2022 <- ground_truth %>%
  filter(survey_start == 2022)

population_2021 <- population %>%
  filter(survey_start == 2021) %>%
  select(-survey_start)

ground_truth_2022_pop <- ground_truth_2022 %>%
  left_join(population_2021, by = "iso3")

ground_truth_pop <- ground_truth_pop %>%
  filter(survey_start != 2022) %>%
  bind_rows(ground_truth_2022_pop)

write_csv(ground_truth_pop, here("data/national/data_refresh/new_national_pipeline_files/groundtruth_offline_predictors_pop.csv"))


##make new national data file
data <- read_csv(here("data/national/data_refresh/new_national_pipeline_files/files/internet_mobile_indicator_clean.csv"))

data_long <- data %>%
  select(iso3, survey_start, survey_type, internet_use_in_12_months_men, internet_use_in_12_months_wom, used_internet_past12months_fm_perc_ratio, owns_mobile_phone_men, owns_mobile_phone_wom, owns_mobile_phone_fm_perc_ratio) %>%
  pivot_longer(cols = c(internet_use_in_12_months_men, internet_use_in_12_months_wom, used_internet_past12months_fm_perc_ratio, owns_mobile_phone_wom, owns_mobile_phone_men, owns_mobile_phone_fm_perc_ratio),
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

write_csv(data_long, here("data/national/data_refresh/new_national_pipeline_files/files/new_groundtruth_national_data.csv"))
