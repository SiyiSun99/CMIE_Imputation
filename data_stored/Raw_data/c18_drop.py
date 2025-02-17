#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   
@Author      :   siyi.sun
@Time        :   2025/02/13 03:26:12
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
            drop_col = ['con_HH_ID', 'con_MEM_ID']
            df.drop(columns=drop_col, inplace=True)
            # fill in missingness for those impossible missing
            fix_col = ['cat_GENDER',"cat_EMPLOYMENT_STATUS" ]
            df[fix_col] = fix_df
            df = df[['con_WAVE_NO', 'cat_STATE', 'cat_HR',
                    'cat_REGION_TYPE', 'cat_MEM_STATUS', 'con_AGE_YRS', 'cat_GENDER',
                    'cat_RELIGION', 'cat_EMPLOYMENT_STATUS', 'cat_PLACE_OF_WORK',
                    'con_TS_ON_TRAVEL', 'con_TS_ON_OUTDOOR_SPORTS', 'cat_IS_HOSPITALISED',
                    'cat_HAS_BANK_AC', 'cat_HAS_MOBILE', 'cat_IS_HEALTHY', 'con_TOT_INC_1',
                    'con_TOT_INC_2', 'con_TOT_INC_3', 'con_TOT_INC_4',
                    'con_INC_OF_HH_FRM_ALL_SRCS_1', 'con_INC_OF_HH_FRM_ALL_SRCS_2',
                    'con_INC_OF_HH_FRM_ALL_SRCS_3', 'con_INC_OF_HH_FRM_ALL_SRCS_4', 'con_INC_OF_MEM_FRM_ALL_SRCS_1',
                    'con_INC_OF_MEM_FRM_ALL_SRCS_2', 'con_INC_OF_MEM_FRM_ALL_SRCS_3',
                    'con_INC_OF_MEM_FRM_ALL_SRCS_4']]
            # Save back to the same file (overwrite)
            df.to_csv(file_path, index=False)
            print(f"Processed: {file_path}")


cohorts = ["C18"]
mm = ["MCAR", "MAR", "MNAR"]
mr = [10, 20, 30, 40, 50]
for cohort in cohorts:
    fix_df = pd.read_csv(f"/home/siyi.sun/CMIE_Project/data_stored/Completed_data/{cohort}/{cohort}_all.csv",
                         header=0,
                         usecols=['cat_GENDER',"cat_EMPLOYMENT_STATUS"])
    for m in mm:
        for r in mr:
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_miss/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)
            folder_path = f"/home/siyi.sun/CMIE_Project/data_stored/data_sample/{cohort}/{cohort}_all/{m}/miss{r}"
            process_csv_files(folder_path, fix_df)