library(mice)      # Required for missing data handling
library(parallel)  # Required for mclapply parallel processing

# source('/Users/siysun/Desktop/CMIE/CMIE_Project/Generate_Missing/Rcode/amputation.R')
source('/home/siyi.sun/CMIE_Project/Generate_Missing/Rcode/amputation.R')

# Main path configuration
# mainPath <- "/Users/siysun/Desktop/CMIE/CMIE_Project/data_stored/"  # Modify this to your main folder path
mainPath <- "/home/siyi.sun/CMIE_Project/data_stored"
completeDataPath <- file.path(mainPath, "Completed_data")
missingDataPath <- file.path(mainPath, "data_miss")
sampleDataPath <- file.path(mainPath, "data_sample")

# Create main output directories
dir.create(missingDataPath, showWarnings = FALSE)
dir.create(sampleDataPath, showWarnings = FALSE)

# List of cohorts
cohorts <- c("C19")

# Missing data configuration
missingMechanism <- c("MCAR","MAR","MNAR")
missList <- c(10, 20, 30, 40, 50)
trainingSampleTime <- 5  # Number of samples for training
testSampleTime <- 1      # Number of samples for testing

# Function to process categorical variables
process_categorical_vars <- function(df) {
  # Get column names
  col_names <- names(df)
  
  # Find indices of categorical variables (starting with "cat_")
  cat_indices <- which(grepl("^cat_", col_names))
  
  if (length(cat_indices) > 0) {
    # Convert categorical columns to factors
    df[cat_indices] <- lapply(df[cat_indices], as.factor)
  } else {
    cat("No categorical variables (starting with 'cat_') found in the dataset\n")
  }
  
  return(df)
}

process_csv <- function(input_path, output_base_path_miss, output_base_path_sample) {
  # Read the CSV file
  cat("Reading file:", input_path, "\n")
  full_df <- read.csv(input_path)
  
  # Process categorical variables
  full_df <- process_categorical_vars(full_df)
  
  # Define columns to exclude from missing data generation
  exclude_cols <- c(1, 2, 3, 4, 7)
  
  # Process for both missing data and samples
  for (output_type in c("miss", "sample")) {
    current_path <- if(output_type == "miss") output_base_path_miss else output_base_path_sample
    current_samples <- if(output_type == "miss") trainingSampleTime else testSampleTime
    
    # Create base output directory if it doesn't exist
    if (!dir.exists(current_path)) {
      dir.create(current_path, recursive = TRUE)
    }
    
    # Generate missing data for each mechanism
    for (method in missingMechanism) {
      method_path <- file.path(current_path, method)
      if (!dir.exists(method_path)) {
        dir.create(method_path)
      }
      
      # Generate for each missing percentage
      for (miss_rate in missList) {
        miss_path <- file.path(method_path, paste0("miss", miss_rate))
        if (!dir.exists(miss_path)) {
          dir.create(miss_path)
        }
        
        # Generate samples sequentially with error handling
        for (i in 1:current_samples) {
          tryCatch({
            file_name <- paste0(i-1, ".csv")
            file_path <- file.path(miss_path, file_name)
            
            # Check if file already exists
            if (!file.exists(file_path)) {
              # Create a copy of the dataframe for modification
              working_df <- full_df
              
              # Get indices of columns to include in missing data generation
              included_cols <- which(!seq_len(ncol(working_df)) %in% exclude_cols)
              
              # Store original column names for verification
              original_names <- names(working_df)
              
              # Generate missing values
              miss_result <- produce_NA(working_df[, included_cols, drop=FALSE], 
                                        mechanism = method, 
                                        perc.missing = miss_rate/100)
              
              # Verify the column order matches before assignment
              if (!identical(names(miss_result$data.incomp), names(working_df[, included_cols, drop=FALSE]))) {
                stop("Column order mismatch detected!")
              }
              
              # Assign missing values back to the working dataframe
              working_df[, included_cols] <- miss_result$data.incomp
              
              # Verify final column order matches original
              if (!identical(names(working_df), original_names)) {
                stop("Final column order does not match original!")
              }
              
              # Save the result
              write.csv(working_df, file_path, row.names = FALSE)
              cat(sprintf("Generated sample %d for %s, %s, miss%d\n", 
                          i-1, output_type, method, miss_rate))
            } else {
              cat(sprintf("File already exists: %s\n", file_path))
            }
          }, error = function(e) {
            cat(sprintf("Error generating sample %d: %s\n", i-1, e$message))
          })
        }
      }
    }
  }
}

# Main processing loop
for (cohort in cohorts) {
  # Create cohort directories in both missing and sample data paths
  cohort_miss_path <- file.path(missingDataPath, cohort)
  cohort_sample_path <- file.path(sampleDataPath, cohort)
  dir.create(cohort_miss_path, showWarnings = FALSE)
  dir.create(cohort_sample_path, showWarnings = FALSE)
  
  # Source data path
  source_cohort_path <- file.path(completeDataPath, cohort)
  
  # Process main cohort file
  main_csv <- file.path(source_cohort_path, paste0(cohort, "_all.csv"))
  if (file.exists(main_csv)) {
    # Create all_data directories
    all_data_miss_path <- file.path(cohort_miss_path, paste0(cohort, "_all"))
    all_data_sample_path <- file.path(cohort_sample_path, paste0(cohort, "_all"))
    dir.create(all_data_miss_path, showWarnings = FALSE)
    dir.create(all_data_sample_path, showWarnings = FALSE)
    
    # Process the main file
    process_csv(main_csv, all_data_miss_path, all_data_sample_path)
  }
  
  # # Create STATE directories
  # state_miss_path <- file.path(cohort_miss_path, "STATE")
  # state_sample_path <- file.path(cohort_sample_path, "STATE")
  # dir.create(state_miss_path, showWarnings = FALSE)
  # dir.create(state_sample_path, showWarnings = FALSE)
  # 
  # # Process state-wise files
  # source_state_path <- file.path(source_cohort_path, "STATE")
  # if (dir.exists(source_state_path)) {
  #   # Get list of state CSV files
  #   state_files <- list.files(source_state_path, pattern = "\\.csv$", full.names = TRUE)
  #   
  #   # Process each state file
  #   for (state_file in state_files) {
  #     # Extract the base name without extension
  #     base_name <- tools::file_path_sans_ext(basename(state_file))
  #     
  #     # Create state-specific directories
  #     state_miss_output_path <- file.path(state_miss_path, base_name)
  #     state_sample_output_path <- file.path(state_sample_path, base_name)
  #     
  #     # Process the state file
  #     process_csv(state_file, state_miss_output_path, state_sample_output_path)
  #   }
  # }
}

