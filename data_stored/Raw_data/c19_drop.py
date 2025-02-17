#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   
@Author      :   siyi.sun
@Time        :   2025/02/13 03:26:12
"""

import os
import pandas as pd

cohorts = ["C19"]
mm = ["MNAR"]
mr = [10, 20, 30, 40, 50]

def process_csv_files(folder_path, fix_df):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read CSV file
            df = pd.read_csv(file_path, header=0)
            # drop columns
            drop_col = [i for i in df.columns if i in ['con_HH_ID', 'con_MEM_ID']]
            df.drop(columns=drop_col, inplace=True)
            # fill in missingness for those impossible missing
            fix_col = ['cat_GENDER']
            df[fix_col] = fix_df
            # Save back to the same file (overwrite)
            df.to_csv(file_path, index=False)
            print(f"Processed: {file_path}")


for cohort in cohorts:
    fix_df = pd.read_csv(f"/home/siyi.sun/CMIE_Project/data_stored/Completed_data/{cohort}/{cohort}_all.csv",
                         header=0,
                         usecols=['cat_GENDER'])
    for m in mm:
        for r in mr:
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_miss/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_sample/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)