library(mice)
library(dplyr)
library(parallel)
library(progress)

# Main Path Configuration
# mainPath <- "/home/siyi.sun/CMIE_Project"
mainPath <- "/Users/siysun/Desktop/CMIE/CMIE_Project"
dataMainPath <- file.path(mainPath, "data_stored")
missMainPath <- file.path(dataMainPath, "data_miss")
dataMicePath <- file.path(dataMainPath, "data_mice_store")

dir.create(dataMicePath, showWarnings = FALSE)

# Missing Data Configuration
cohorts <- c("C19")
missingMechanism <- c("MAR")
missList <- c(10)  # Matching sample_miss.R
sampleTime <- 3
multi_ImputationTime <- 5
maxiteration <- 5

# Imputation method configuration
imputation_methods <- list(
  "C18" = c('pmm', 'pmm', 'pmm', 'polyreg', 'polyreg', 'logreg', 'logreg', 'pmm', 'logreg', 'polyreg', 'polyreg', 'polyreg', 'pmm', 'pmm', 'logreg', 'logreg', 'logreg', 'logreg', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm'),
  "C19" = c('pmm', 'pmm', 'pmm', 'polyreg', 'polyreg', 'logreg', 'logreg', 'pmm', 'logreg', 'polyreg', 'polyreg', 'pmm', 'pmm', 'pmm', 'logreg', 'logreg', 'logreg', 'logreg', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm', 'pmm')
)

# Function to process categorical variables
process_categorical_vars <- function(df) {
  cat_indices <- which(grepl("^cat_", names(df)))
  if (length(cat_indices) > 0) {
    df[cat_indices] <- lapply(df[cat_indices], as.factor)
  }
  return(df)
}

# Initialize timing dataframe
imputation_times <- data.frame(Dataset = character(), Mechanism = character(), MissingRatio = integer(), AvgTime = numeric(), stringsAsFactors = FALSE)
total_time <- 0
# Progress bar setup
pb <- progress_bar$new(
  format = "  Processing [:bar] :percent eta: :eta",
  total = length(cohorts) * length(missingMechanism) * length(missList) * sampleTime, clear = FALSE, width = 60
)

for (cohort in cohorts) {
  source_cohort_path <- file.path(missMainPath, cohort)
  mice_source_cohort_path <- file.path(dataMicePath, cohort)
  dir.create(mice_source_cohort_path, showWarnings = FALSE)
  
  # all csv
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
      
      # record time
      total_time <- 0

      mclapply(1:sampleTime, function(i) {
        pb$tick()
        file_name <- paste0(i-1, ".csv")
        file_path <- file.path(miss_path, file_name)
        if (!file.exists(file_path)) return()
        
        mis_df <- read.csv(file_path)
        mis_df <- process_categorical_vars(mis_df)
        pred <- quickpred(mis_df)
      
        start_time <- Sys.time()
        imputed_data <- mice(mis_df, pred = pred, m = multi_ImputationTime, maxit = maxiteration, method = imputation_methods[[cohort]], print = FALSE)
        end_time <- Sys.time()
        
        total_time <<- total_time + as.numeric(difftime(end_time, start_time, units = "secs"))
        
        holderMissMicePath <- file.path(missMicePath, toString(i-1))
        dir.create(holderMissMicePath, showWarnings = FALSE)
        
        for (index in 1:multi_ImputationTime) {
          fileImputedName <- paste0(index-1, ".csv")
          fileImputedPath <- file.path(holderMissMicePath, fileImputedName)
          write.csv(complete(imputed_data, index), fileImputedPath, row.names = FALSE)
        }
      }, mc.cores = min(sampleTime, detectCores()))
      
      avg_time <- total_time / sampleTime
      imputation_times <- rbind(imputation_times, data.frame(Dataset = cohort, Mechanism = method, MissingRatio = miss_rate, AvgTime = avg_time))
    }
  }
}

# Save imputation times to CSV
write.csv(imputation_times, file.path(mainPath, "imputation_times.csv"), row.names = FALSE)