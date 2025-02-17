#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   drop unnecessary columns (constant or colinear) and mask those unchange columns
@Author      :   siyi.sun
@Time        :   2025/02/13 01:44:05
"""

import os
import pandas as pd

def process_csv_files(folder_path, fix_df):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            
            # Read CSV file
            df = pd.read_csv(file_path, header=0)
            
            # drop columns
            drop_col = [i for i in df.columns if i[:-1] == "con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_" or i in ['con_HH_ID', 'con_MEM_ID', "cat_EMPLOYMENT_STATUS"]]
            df.drop(columns=drop_col, inplace=True)
            
            # fill in missingness for those impossible missing
            fix_col = ['con_WAVE_NO', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE', 'cat_GENDER']
            df[fix_col] = fix_df
            
            # Save back to the same file (overwrite)
            df.to_csv(file_path, index=False)
            print(f"Processed: {file_path}")


# Replace 'your_folder_path' with the actual folder containing the CSV files
cohorts = ["C19"]
mm = ["MNAR"]
mr = [10, 20, 30, 40, 50]
for cohort in cohorts:
    fix_df = pd.read_csv(f"/home/siyi.sun/CMIE_Project/data_stored/Completed_data/{cohort}/{cohort}_all.csv",
                         header=0,
                         usecols=['con_WAVE_NO', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE', 'cat_GENDER'])
    for m in mm:
        for r in mr:
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_miss/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_sample/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)