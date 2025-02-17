library(mice)
library(dplyr)
library(progress)

# Main Path Configuration 
mainPath <- "/Users/siysun/Desktop/CMIE/CMIE_Project"
dataMainPath <- file.path(mainPath, "data_stored")
missMainPath <- file.path(dataMainPath, "data_miss_mask")
dataMicePath <- file.path(dataMainPath, "data_mice_store")
dir.create(dataMicePath, showWarnings = FALSE)

# Missing Data Configuration
cohorts <- c("C19")
missingMechanism <- c("MAR")
missList <- c(30)
sampleTime <- 1
multi_ImputationTime <- 5
maxiteration <- 5

# Imputation method configuration
imputation_methods <- list(
  "C19" = c('pmm', 'polyreg', 'polyreg', 'logreg', 'logreg', 'pmm', 'logreg', 'polyreg', 'polyreg',
            'pmm', 'pmm', 'pmm', 'logreg', 'logreg', 'logreg', 'logreg', rep("pmm", 12))
)

# Modified quickpred function with minimum correlation threshold
custom_quickpred <- function(data, mincor = 0.1) {
  p <- ncol(data)
  pred <- matrix(FALSE, nrow = p, ncol = p, dimnames = list(names(data), names(data)))
  
  for (j in 1:p) {
    if (is.numeric(data[[j]])) {
      for (i in 1:p) {
        if (i != j && !all(is.na(data[[i]]))) {
          if (is.numeric(data[[i]])) {
            correlation <- cor(data[[j]], data[[i]], use = "pairwise.complete.obs")
            pred[j, i] <- abs(correlation) > mincor
          }
        }
      }
    }
  }
  return(pred)
}

# Function to process categorical variables
process_categorical_vars <- function(df) {
  tryCatch({
    cat_indices <- which(grepl("^cat_", names(df)))
    if (length(cat_indices) > 0) {
      for (col in names(df)[cat_indices]) {
        df[[col]][df[[col]] == ""] <- NA
        df[[col]] <- as.factor(df[[col]])
      }
    }
    return(df)
  }, error = function(e) {
    cat("Error in process_categorical_vars:", conditionMessage(e), "\n")
    return(df)
  })
}

# Initialize timing dataframe
imputation_times <- data.frame(
  Dataset = character(),
  Mechanism = character(),
  MissingRatio = integer(),
  AvgTime = numeric(),
  stringsAsFactors = FALSE
)

# Progress bar setup
total_iterations <- length(cohorts) * length(missingMechanism) * length(missList) * sampleTime
pb <- progress_bar$new(
  format = "  Processing [:bar] :percent eta: :eta",
  total = total_iterations,
  clear = FALSE,
  width = 60
)

# Error log setup
error_log <- data.frame(
  Dataset = character(),
  Mechanism = character(),
  MissingRatio = integer(),
  Sample = integer(),
  Error = character(),
  stringsAsFactors = FALSE
)

# Function to read data and apply missing value mask
apply_missing_mask <- function(data_df, mask_path) {
  mask_df <- read.csv(mask_path)
  if (!identical(dim(data_df), dim(mask_df))) {
    stop("Dimension mismatch between data and mask")
  }
  mask_matrix <- as.matrix(mask_df) == 1
  data_df[mask_matrix] <- NA
  return(data_df)
}

for (cohort in cohorts) {
  source_cohort_path <- file.path(missMainPath, cohort)
  mice_source_cohort_path <- file.path(dataMicePath, cohort)
  dir.create(mice_source_cohort_path, showWarnings = FALSE)
  
  raw_all_data_path <- file.path(dataMainPath, paste0("Completed_data/", cohort, "/", cohort, "_all.csv"))
  raw_all_data <- read.csv(raw_all_data_path)
  all_csv_folder <- file.path(source_cohort_path, paste0(cohort, "_all"))
  mice_all_csv_folder <- file.path(mice_source_cohort_path, paste0(cohort, "_all"))
  dir.create(mice_all_csv_folder, showWarnings = FALSE)
  
  for (method in missingMechanism) {
    methodPath <- file.path(all_csv_folder, method)
    methodMicePath <- file.path(mice_all_csv_folder, method)
    dir.create(methodMicePath, showWarnings = FALSE)
    
    for (miss_rate in missList) {
      miss_path <- file.path(methodPath, paste0("miss", miss_rate))
      missMicePath <- file.path(methodMicePath, paste0("miss", miss_rate))
      dir.create(missMicePath, showWarnings = FALSE)
      
      total_time <- 0
      
      for (i in 1:sampleTime) {
        tryCatch({
          pb$tick()
          file_name <- paste0(i-1, ".csv")
          file_path <- file.path(miss_path, file_name)
          
          if (!file.exists(file_path)) {
            cat(sprintf("File not found: %s\n", file_path))
            next
          }
          
          mis_df <- apply_missing_mask(raw_all_data, file_path)
          mis_df <- process_categorical_vars(mis_df)
          
          # Use custom predictor matrix with higher correlation threshold
          pred <- custom_quickpred(mis_df, mincor = 0.3)
          
          start_time <- Sys.time()
          imputed_data <- mice(mis_df,
                               pred = pred,
                               m = multi_ImputationTime,
                               maxit = maxiteration,
                               method = imputation_methods[[cohort]],
                               print = FALSE)
          end_time <- Sys.time()
          
          total_time <- total_time + as.numeric(difftime(end_time, start_time, units = "secs"))
          
          holderMissMicePath <- file.path(missMicePath, toString(i-1))
          dir.create(holderMissMicePath, showWarnings = FALSE)
          
          for (index in 1:multi_ImputationTime) {
            fileImputedName <- paste0(index-1, ".csv")
            fileImputedPath <- file.path(holderMissMicePath, fileImputedName)
            write.csv(complete(imputed_data, index), fileImputedPath, row.names = FALSE)
          }
          
        }, error = function(e) {
          error_message <- conditionMessage(e)
          cat(sprintf("Error processing %s, %s, miss rate %d, sample %d: %s\n",
                      cohort, method, miss_rate, i-1, error_message))
          
          error_log <<- rbind(error_log, data.frame(
            Dataset = cohort,
            Mechanism = method,
            MissingRatio = miss_rate,
            Sample = i-1,
            Error = error_message,
            stringsAsFactors = FALSE
          ))
        })
      }
      
      avg_time <- total_time / sampleTime
      imputation_times <- rbind(imputation_times,
                                data.frame(Dataset = cohort,
                                           Mechanism = method,
                                           MissingRatio = miss_rate,
                                           AvgTime = avg_time))
      
      write.csv(imputation_times,
                file.path(mainPath, "imputation_times.csv"),
                row.names = FALSE)
      
      write.csv(error_log,
                file.path(mainPath, "imputation_errors.csv"),
                row.names = FALSE)
      
      cat(sprintf("Saved results for %s, %s, miss rate %d\n",
                  cohort, method, miss_rate))
    }
  }
}