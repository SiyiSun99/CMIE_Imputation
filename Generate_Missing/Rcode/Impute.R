library(mice)
library(missForest)
library(dplyr)

library(MASS)
library(Amelia)
library(missMDA)
library(softImpute)
library(tidyr)
library(ggplot2)
library(devtools)
library(gdata)
library(mltools)
library(parallel)
library(progress)

#Please modify the path o current code_path in your system
mainPath <- "/data/coml-data-imputation/shug7754"

#find the file that contain miss and full datasets
dataMainPath <- file.path(mainPath, "data_stored")
missMainPath <- file.path(dataMainPath, "data_miss")

missingMechanism <- list("MNAR")
casenum <- 14# the case index list, new index could be added in by expanding the range.
missList <- as.numeric(list(30,50)) # missing ratio from ./Rcode/sample_miss.R
sampleTime <- 10 # num of training test defined in ./Rcode/sample_miss.R
multi_ImputationTime <- 5 # num of multiple imputation
maxiteration <- 5 # num of iterations

dataMicePath <- file.path(dataMainPath, "data_mice_store")
# dir.create(dataMicePath)

pb <- progress_bar$new(
  format = "  downloading [:bar] :percent eta: :eta",
  total = 100, clear = FALSE, width= 60)

for(method_index in 1:length(missingMechanism)){
  method <- missingMechanism[method_index]
  methodPath <- file.path(missMainPath, method)
  methodMicePath <- file.path(dataMicePath, method)
  # dir.create(methodMicePath)
  # eg: data_stored/data_mice_store/MCAR

  case_name <- paste("Case", toString(casenum), sep = "") 
  case_add <- file.path(methodPath, case_name)
  #initial the case path for store the imputed datasets from mice
  caseMicePath <- file.path(methodMicePath, case_name)
  # dir.create(caseMicePath)
  # eg: data_stored/data_mice_store/MCAR/Case12
  
  for(j in 1:length(missList)){
    
    #find the path of missing case in data file
    miss_name <- paste("miss", toString(missList[j]), sep = "")
    mis_add <- file.path(case_add, miss_name)
    missMicePath <- file.path(caseMicePath, miss_name)
    
    if(!dir.exists(missMicePath)){
      dir.create(missMicePath)
      # eg: data_stored/data_mice_store/MCAR/Case12/miss10
    }
    
    lapply(1:sampleTime, function(i) {
      pb$tick()
      file_name <- paste(toString(i-1), ".csv", sep = "")
      file_path <- file.path(mis_add, file_name)
      mis_df <- read.csv(file_path)
      
      if(casenum == 1){
        matrix <- c(rep("pmm", 93))
      }
      else if(casenum == 3){
        matrix <- c(rep("pmm", 57))
      }
      else if(casenum == 2){
        matrix <- c(rep("pmm", 8))
      }
      else if(casenum == 4){
        cat <- c(1:7)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c(rep("logreg", 7))
      }
      else if(casenum == 5){
        cat <- c(1:180)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c(rep("logreg", 180))
      }
      else if(casenum == 6){
        cat <- c(1, 2, 5, 7, 9, 10)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c("polyreg", "logreg", "pmm", "pmm", "logreg", "pmm", "logreg", "pmm", "logreg", "polyreg")
      }
      else if(casenum == 7){
        cat <- c(4, 5, 6, 8, 11)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c(rep("pmm", 3), "logreg", "logreg", "polyreg", "pmm", "logreg", "pmm", "pmm", "polyreg", "pmm")
      }
      else if(casenum == 11){
        cat <- c(1)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c('polyreg',rep("pmm", 7)) #here is list of regression methods for each categorical features#)
      }
      else if(casenum == 12){
        cat <- c(2,4,6,7,8,9,10,14,15) # index of categorical feature
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c('pmm','pmm','pmm','pmm','pmm','polyreg','pmm','polyreg','polyreg','logreg',rep('pmm',3),'pmm','polyreg') #here is list of regression methods for each categorical features#)
      }
      else if(casenum == 13){
        cat <- c(2,3,8,9,11) # index of categorical feature
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c('pmm','polyreg','logreg','pmm','pmm','pmm','pmm','logreg','logreg','pmm','logreg') 
      }
      else if(casenum == 14){
        cat <- c(1,6,7,8) # index of categorical feature
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c('polyreg','pmm','pmm','pmm','pmm','polr','logreg','logreg',rep("pmm", 4)) 
      }
      #If new dataset was added in, after assigned the new data with a new integer name as case index.
      #you could add code below in "else if" block for querying;
      #else if(k == Index_new){
        #cat <- c({The index for the categorical features counted from initial 1})
        #mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        #matrix <- c(rep("pmm", 10), #here is list of regression methods for each categorical features#)
      #}
      #
      else if(casenum > 7){
        cat <- c(11, 12, 13, 14, 15)
        mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
        matrix <- c(rep("pmm", 10), "polyreg", "polyreg", "logreg", "logreg", "logreg")
      }
      
      
      #project_imputed <- mice(mis_dat, m=1, maxit = 30, method = 'pmm', predictorMatrix=predM, seed = 500)
      pred <- quickpred(mis_df)
      project_imputed <- mice(mis_df, pred = pred, m = multi_ImputationTime, maxit = maxiteration, method = matrix, print=FALSE)
      
      holderMissMicePath <- file.path(missMicePath, toString(i-1))
      dir.create(holderMissMicePath)
      for(index in 1:multi_ImputationTime){
        fileImputedName <-  paste(toString(index-1), ".csv", sep = "")
        fileImputedPath <- file.path(holderMissMicePath, fileImputedName)
        data_imputed <- complete(project_imputed, index)
        write.csv(data_imputed, fileImputedPath, row.names = FALSE)
      }
      
    })# \# of cores for multi-processing of imputing process.
  }
}