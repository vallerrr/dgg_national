##These code are used for unzipping and classifying the original data downloaded. 

##It will not run smoothly within the current directory

##It is just for reference

library(haven)
library(tidyverse)
library(here)

zip_files <- list.files(pattern = "\\.zip$")
for (zip_file in zip_files) {
  # Unzip files into the current directory
  unzip(zip_file, exdir = ".") 
}
  
  # List all files extracted
  extracted_files <- list.files()
  
  # Identify files that are not .dta
  files_to_remove <- extracted_files[!grepl("\\.DTA$", extracted_files) & !grepl("\\.R$", extracted_files)]
  
  # Remove those files
  file.remove(files_to_remove)


  
  # Destination directories
dest_dir_dhs7_women <- "D:/Study/RA/micro dataset/DHS/DHS7 Women"
dest_dir_dhs7_men <- "D:/Study/RA/micro dataset/DHS/DHS7 Men"
dest_dir_dhs8_women <- "D:/Study/RA/micro dataset/DHS/DHS8 Women"
dest_dir_dhs8_men <- "D:/Study/RA/micro dataset/DHS/DHS8 Men"
  
files <- list.files(full.names = TRUE)
for (file_path in files) {
  file_name <- basename(file_path)
  
  # Determine the destination directory based on the file name
  if (grepl("IR7", file_name)) {
    dest_dir <- dest_dir_dhs7_women
  } else if (grepl("MR7", file_name)) {
    dest_dir <- dest_dir_dhs7_men
  } else if (grepl("IR8", file_name)) {
    dest_dir <- dest_dir_dhs8_women
  } else if (grepl("MR8", file_name)) {
    dest_dir <- dest_dir_dhs8_men
  } else {
    next # Skip files that do not match the criteria
  }
  
  # Construct the destination path
  dest_path <- file.path(dest_dir, file_name)
  
  # Move the file
  file.rename(from = file_path, to = dest_path)
}

setwd("D:/Study/RA/micro dataset/PMA")
zip_files <- list.files(pattern = "\\.zip$")
for (zip_file in zip_files) {
  # Unzip files into the current directory
  unzip(zip_file, exdir = ".") 
}

setwd("D:/Study/RA/micro dataset/MICS")
zip_files <- list.files(pattern = "\\.zip$")
for (zip_file in zip_files) {
  # Unzip files into the current directory
  unzip(zip_file, exdir = ".") 
}
file.remove(zip_files)

files <- list.files(pattern = "SPSS")
for (file in files) {
  # Generate the new folder name by removing "SPSS " from the file name
  new_folder_name <- gsub("SPSS ", "", file)
  
  # Remove the file extension to ensure the folder name is clean
  new_folder_name <- tools::file_path_sans_ext(new_folder_name)
  
  # Create the new folder if it doesn't already exist
  if (!dir.exists(new_folder_name)) {
    dir.create(new_folder_name)
  }
  
  # Construct the new file path
  new_file_path <- file.path(new_folder_name, file)
  
  # Move the file to the new folder
  file.rename(from = file, to = new_file_path)
}
files <- list.files(pattern = "\\.sav$")
file.remove(files)



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

setwd(here("data/national/data_refresh/DHS/Continuous DHS"))

list_of_fefiles <- list.files(pattern = "IR", full.names = T)
list_of_mfiles <- list.files(pattern = "MR", full.names = T)

files_without_v171a <- check_files_for_variable(list_of_fefiles, "v171a")
files_without_mv171a <- check_files_for_variable(list_of_mfiles, "mv171a")


# Base directory where the folders will be created
base_directory <- here("data/national/data_refresh/DHS/wave8")

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
countries_years <- c(
  "Afghanistan_2015", "Albania_2017-18", "Angola_2015-16",
  "Armenia_2015-16", "Bangladesh_2014", "Bangladesh_2017-18",
  "Benin_2017-18", "Burundi_2016-17", "Cameroon_2018",
  "Ethiopia_2016", "Gabon_2019-21", "Guatemala_2014-15",
  "Haiti_2016-17", "Indonesia_2017", "Jordan_2017-18",
  "Liberia_2019-20", "Malawi_2015-16", "Maldives_2016-17",
  "Mali_2018", "Mauritania_2019-21", "Myanmar_2015-16",
  "Nepal_2016", "Nigeria_2018", "Papua_New_Guinea_2016-18",
  "Philippines_2017", "Sierra_Leone_2019", "South_Africa_2016",
  "Tajikistan_2017", "Tanzania_2015-16", "Timor-Leste_2016",
  "Turkey_2018", "Uganda_2016", "Zambia_2018", "Zimbabwe_2015"
)


# Assuming you have a list of DTA files in or below the base directory
list_of_files <- list.files(path = base_directory, pattern = "\\.DTA$", full.names = TRUE, recursive = TRUE)

# Function to move files based on the first two letters (country code) of their filenames
move_files_based_on_country_code <- function(file_paths, country_codes, base_dir) {
  for (file_path in file_paths) {
    # Extract the filename and the country code (first two characters)
    file_name <- basename(file_path)
    country_code <- substr(file_name, 1, 2)
    
    # Find the corresponding country name using the country code
    country_name <- country_codes[country_code]
    
    if (!is.na(country_name)) {
      # Replace spaces and special characters in country names with underscores for folder names
      folder_name <- gsub(" ", "_", country_name)
      folder_name <- gsub("'", "", folder_name) # Remove apostrophes
      
      # Construct the full path for the target directory
      target_dir_path <- file.path(base_dir, folder_name)
      
      # Create the directory if it doesn't exist
      if (!dir.exists(target_dir_path)) {
        dir.create(target_dir_path, recursive = TRUE)
      }
      
      # Construct the full target file path
      target_file_path <- file.path(target_dir_path, file_name)
      
      # Move the file
      if (!file.rename(file_path, target_file_path)) {
        cat("Failed to move file:", file_name, "to", target_dir_path, "\n")
      }
    } else {
      cat("No matching country for code:", country_code, "- file name:", file_name, "\n")
    }
  }
}

# Execute the function
move_files_based_on_country_code(list_of_files, country_codes, base_directory)

# Specify the path to the base directory containing the subfolders
base_directory <- here("data/national/data_refresh/mics")

# List all subdirectories within the base directory
subdirectories <- list.dirs(path = base_directory, full.names = TRUE, recursive = FALSE)

# Loop through each subdirectory and rename it to lowercase
for (dir_path in subdirectories) {
  # Construct the new directory path with lowercase name
  new_dir_path <- tolower(dir_path)
  
  # Check if the new directory path is different from the original
  if (new_dir_path != dir_path) {
    # Rename the directory
    if (file.rename(dir_path, new_dir_path)) {
      cat("Renamed:", dir_path, "to", new_dir_path, "\n")
    } else {
      cat("Failed to rename:", dir_path, "\n")
    }
  }
}

# Set the working directory to the parent folder
setwd(here("data/national/data_refresh/mics"))

# List all directories in the current working directory
directories <- list.dirs(path = ".", full.names = TRUE, recursive = FALSE)

# Loop through each directory and rename it by replacing spaces with underscores
for (dir in directories) {
  file.rename(from = dir, to = gsub(" ", "_", dir))
}


# Example list of parent folder paths
parent_folders <- list.files(path = here("data/national/data_refresh/mics"))

list.dirs(here("data/national/data_refresh/mics/Afghanistan MICS6 Datasets"),recursive = FALSE)

# Function to move files from each subfolder to its parent folder and delete the subfolder
move_files_to_parent_and_delete_subfolder <- function(parent_folders) {
  for (parent_folder in parent_folders) {
  # Identify the subfolder within the parent folder (assuming there's only one)
  subfolders <- list.dirs(parent_folder, full.names = TRUE, recursive = FALSE)
    
    # List all files in the subfolder
    files_in_subfolder <- list.files(subfolders, full.names = TRUE)
    
      file_name <- basename(files_in_subfolder)
      
      new_file_path <- file.path(parent_folder, file_name)
      file.rename(files_in_subfolder, new_file_path)
      unlink(subfolders, recursive = TRUE)
  }
}
# Apply the function to each parent folder

move_files_to_parent_and_delete_subfolder(parent_folders)



parent_folder <- here("data/national/data_refresh/mics/Argentina MICS6 Datasets")
subfolders <- list.dirs(parent_folder, full.names = TRUE, recursive = FALSE)
subfolders <- subfolders[1]
subfolders
# List all files in the subfolder
files_in_subfolder <- list.files(subfolders, full.names = T)


  file_name <- basename(files_in_subfolder)
  new_file_path <- file.path(parent_folder, file_name)
  file.rename(files_in_subfolder, new_file_path)
  unlink(subfolders, recursive = TRUE)

