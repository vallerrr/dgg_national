library(tidyverse)
library(here)
library(haven)


fefilelist <- list.files(here("data/national/data_refresh/dhs"), pattern = "IR", recursive = T, full.names = T)


check_files_for_variable <- function(file_list, variable_name) {
  files_without_variable <- c()  # Initialize an empty vector to store file names
  
  # Check each file for the specified variable
  for (file in file_list) {
    data <- tryCatch({
      read_dta(file, n_max = 1000)  # Try to read the first 1000 rows of the file
    }, error = function(e) {
      NULL  # Return NULL if there's an error reading the file
    })
    
    # If data is successfully read, check for the variable
    if (!is.null(data)) {
      # Check if the variable doesn't exist or only contains NA values
      if (!variable_name %in% names(data) || all(is.na(data[[variable_name]]), na.rm = FALSE)) {
        files_without_variable <- c(files_without_variable, file)
      }
    }
  }
  
  return(files_without_variable)  # Return the vector of file names
}


family_planning_fe <- check_files_for_variable(fefilelist, "v384d")

family_planning_fe_filtered <- fefilelist[!fefilelist %in% family_planning_fe]

family_planning_fe_filtered

mfilelist <- list.files(here("data/national/data_refresh/dhs"), pattern = "MR", recursive = T, full.names = T)

family_planning_m <- check_files_for_variable(mfilelist, "mv384d")

family_planning_m_filtered <- mfilelist[!mfilelist %in% family_planning_m]

family_planning_m_filtered


mobile_transanction_fe <- check_files_for_variable(fefilelist, "v169b")

mobile_transanction_fe_filtered <- fefilelist[!fefilelist %in% mobile_transanction_fe]

mobile_transanction_fe_filtered

mobile_transanction_m <- check_files_for_variable(mfilelist, "mv169b")

mobile_transanction_m_filtered <- mfilelist[!mfilelist %in% mobile_transanction_m]

mobile_transanction_m_filtered

internet_penetration_fe <- check_files_for_variable(fefilelist, "v171a")

internet_penetration_fe_filtered <- fefilelist[!fefilelist %in% internet_penetration_fe]

internet_penetration_fe_filtered

internet_penetration_m <- check_files_for_variable(mfilelist, "mv171a")

internet_penetration_m_filtered <- mfilelist[!mfilelist %in% internet_penetration_m]

internet_penetration_m_filtered

internet_frequency_fe <- check_files_for_variable(fefilelist, "v171b")

internet_frequency_fe_filtered <- fefilelist[!fefilelist %in% internet_frequency_fe]

internet_frequency_fe_filtered

internet_frequency_m <- check_files_for_variable(mfilelist, "mv171b")

internet_frequency_m_filtered <- mfilelist[!mfilelist %in% internet_frequency_m]

internet_frequency_m_filtered


mobile_ownership_fe <- check_files_for_variable(fefilelist, "v169a")

mobile_ownership_fe_filtered <- fefilelist[!fefilelist %in% mobile_ownership_fe]

mobile_ownership_fe_filtered

mobile_ownership_m <- check_files_for_variable(mfilelist, "mv169a")

a <- mobile_ownership_m[!mobile_ownership_m %in% internet_frequency_m_filtered]

mobile_ownership_m_filtered <- mfilelist[!mfilelist %in% mobile_ownership_m]

mobile_ownership_m_filtered

ground_truth <- read_csv(here("data/national/data_refresh/new_national_pipeline_files/groundtruth_offline_predictors.csv"))

ground_truth_check <- ground_truth %>%
  group_by(country) %>%
  slice(1) %>%
  ungroup() %>%
  select(c(50:5275))

non_na_counts <- ground_truth_check %>% summarise(across(everything(), ~ sum(!is.na(.))))

result <- non_na_counts %>%
  pivot_longer(cols = everything(), names_to = "column_name", values_to = "non_na_count") %>%
  mutate(year = sub(".*_(\\d+)$", "\\1", column_name), 
         variable = sub("_(\\d+)$", "", column_name)) %>% 
  select(-column_name) %>%
  pivot_wider(names_from = variable, values_from = non_na_count, values_fill = list(non_na_count = 0))

result_long <- result %>%
  pivot_longer(cols = -year, names_to = "variable", values_to = "non_na_count") %>%
  filter(year > 1990)

heatmap_plot <- ggplot(result_long, aes(x = variable, y = year, fill = non_na_count)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "darkred") + # Color gradient
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  labs(title = "Heatmap of Non-NA Counts by Year and Variable", x = "Variable", y = "Year", fill = "Count")

ggsave(plot = heatmap_plot, here("data/national/data_refresh/new_national_pipeline_files/graphs/heatmap_for_non_na_counts.png"), width = 10, height = 8)
