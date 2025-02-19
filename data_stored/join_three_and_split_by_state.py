#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Final processing of three datasets.
@Author      :   siyi.sun
@Time        :   2025/02/02 12:14:12
"""

# export PATH="/home/siyi.sun/miniconda3/bin:$PATH"
import pandas as pd

def combine_dataframes(c19_df, income_df):
    """
    Combine three dataframes using inner joins.
    
    Args:
        c19_df: DataFrame with WAVE_NO, HH_ID, MEM_ID
        income_hh_df: DataFrame with WAVE_NO, HH_ID
        income_mem_df: DataFrame with WAVE_NO, HH_ID, MEM_ID
        
    Returns:
        pandas.DataFrame: Combined dataframe with all matching records
    """
    print(c19_df.shape)

    # merge C19 with income
    # Join on WAVE_NO, HH_ID, and MEM_ID to ensure correct matching
    final_df = c19_df.merge(
        income_df,
        on=['WAVE_NO', 'HH_ID', 'MEM_ID'],
        how='inner'
    )
    print(final_df.shape)

    return final_df

def split_and_save_by_state(final_df,C_num,save_url):
    """
    Split dataframe by STATE and save each subset with formatted filename.
    
    Args:
        final_df: DataFrame containing a 'STATE' column
    """
    # Get unique states
    states = final_df['cat_STATE'].unique()
    state_count = {}
    
    # Process each state
    for state in states:
        # Create state-specific dataframe
        state_df = final_df[final_df['cat_STATE'] == state]
        
        # Format state name for filename:
        # 1. Replace spaces with underscore
        # 2. Replace & with underscore
        # 3. Strip any leading/trailing spaces
        formatted_state = state.strip().replace(' ', '_')
        
        # Create filename
        filename = f'C{C_num}_{formatted_state}'
        
        # Save the dataframe
        state_df.to_csv(f'{save_url}{filename}.csv', index=False)
        state_count[formatted_state] = len(state_df)
        
        # Print confirmation
        print(f"Saved {filename}.csv with {len(state_df)} rows")
        
    return state_count

def convert_to_int(df, columns_to_convert):
    """
    Convert specified columns to integer type, handling potential errors.
    
    Args:
        df: pandas DataFrame
        columns_to_convert: list of column names to convert to int
    
    Returns:
        pandas.DataFrame: DataFrame with converted columns
    """
    df_copy = df.copy()
    
    for col in columns_to_convert:
        if col in df_copy.columns:
            try:
                # First convert to float to handle any potential NaN values
                df_copy[col] = df_copy[col].fillna(-999)  # or any other value you want to use for NaN
                df_copy[col] = df_copy[col].astype(float).astype(int)
            except Exception as e:
                print(f"Error converting {col}: {str(e)}")
        else:
            print(f"Column {col} not found in DataFrame")
    
    return df_copy

c19 = pd.read_csv("Raw_data/complete_data_wo_was_hospitalized.csv", header = 0, index_col = 0)
c18 = pd.read_csv("Raw_data/complete_data_wo_ts_was.csv", header = 0, index_col = 0)
c22 = pd.read_csv("Raw_data/complete_data_onehot_was.csv", header = 0, index_col = 0)
income = pd.read_csv("Raw_data/income.csv", header = 0, index_col = 0)
columns_int = ['WAVE_NO', 'HH_ID', 'MEM_ID', 'AGE_YRS', 
                       'IS_HOSPITALISED', 'HAS_BANK_AC', 'HAS_MOBILE',
                       'IS_HEALTHY', 'TOT_INC_1', 'TOT_INC_2', 'TOT_INC_3', 'TOT_INC_4',
                       'INC_OF_HH_FRM_ALL_SRCS_1', 'INC_OF_HH_FRM_ALL_SRCS_2',
                       'INC_OF_HH_FRM_ALL_SRCS_3', 'INC_OF_HH_FRM_ALL_SRCS_4',
                       'INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 'INC_OF_ALL_MEMS_FRM_ALL_SRCS_2',
                       'INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 'INC_OF_ALL_MEMS_FRM_ALL_SRCS_4',
                       'INC_OF_MEM_FRM_ALL_SRCS_1', 'INC_OF_MEM_FRM_ALL_SRCS_2',
                       'INC_OF_MEM_FRM_ALL_SRCS_3', 'INC_OF_MEM_FRM_ALL_SRCS_4']
new_columns = ['con_WAVE_NO', 'con_HH_ID', 'con_MEM_ID', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE',
                 'cat_MEM_STATUS', 'con_AGE_YRS', 'cat_GENDER', 'cat_RELIGION', 'cat_EMPLOYMENT_STATUS',
                 'cat_PLACE_OF_WORK', 'con_TS_ON_WORK_FOR_EMPLOYER', 'con_TS_ON_TRAVEL',
                 'con_TS_ON_OUTDOOR_SPORTS', 'cat_IS_HOSPITALISED', 'cat_HAS_BANK_AC', 'cat_HAS_MOBILE',
                 'cat_IS_HEALTHY', 'con_TOT_INC_1', 'con_TOT_INC_2', 'con_TOT_INC_3', 'con_TOT_INC_4',
                 'con_INC_OF_HH_FRM_ALL_SRCS_1','con_INC_OF_HH_FRM_ALL_SRCS_2','con_INC_OF_HH_FRM_ALL_SRCS_3', 
                 'con_INC_OF_HH_FRM_ALL_SRCS_4','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_2','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_4','con_INC_OF_MEM_FRM_ALL_SRCS_1', 
                 'con_INC_OF_MEM_FRM_ALL_SRCS_2','con_INC_OF_MEM_FRM_ALL_SRCS_3', 'con_INC_OF_MEM_FRM_ALL_SRCS_4']

################################# C19 #######################################
print("Start processing C19.")
final_c19 = combine_dataframes(c19, income)
final_c19 = convert_to_int(final_c19, columns_int)
final_c19.columns = ['con_WAVE_NO', 'con_HH_ID', 'con_MEM_ID', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE',
                 'cat_MEM_STATUS', 'con_AGE_YRS', 'cat_GENDER', 'cat_RELIGION', 'cat_EMPLOYMENT_STATUS',
                 'cat_PLACE_OF_WORK', 'con_TS_ON_WORK_FOR_EMPLOYER', 'con_TS_ON_TRAVEL',
                 'con_TS_ON_OUTDOOR_SPORTS', 'cat_IS_HOSPITALISED', 'cat_HAS_BANK_AC', 'cat_HAS_MOBILE',
                 'cat_IS_HEALTHY', 'con_TOT_INC_1', 'con_TOT_INC_2', 'con_TOT_INC_3', 'con_TOT_INC_4',
                 'con_INC_OF_HH_FRM_ALL_SRCS_1','con_INC_OF_HH_FRM_ALL_SRCS_2','con_INC_OF_HH_FRM_ALL_SRCS_3', 
                 'con_INC_OF_HH_FRM_ALL_SRCS_4','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_2','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_4','con_INC_OF_MEM_FRM_ALL_SRCS_1', 
                 'con_INC_OF_MEM_FRM_ALL_SRCS_2','con_INC_OF_MEM_FRM_ALL_SRCS_3', 'con_INC_OF_MEM_FRM_ALL_SRCS_4']

final_c19.to_csv("Completed_data/C19/C19_all.csv", index=False)
dataset_state_c19 = split_and_save_by_state(final_c19, 19, "Completed_data/C19/STATE/")

################################# C18 #######################################
print("Start processing C18.")
final_c18 = combine_dataframes(c18, income)
final_c18 = convert_to_int(final_c18, columns_int)
final_c18.columns = ['con_WAVE_NO', 'con_HH_ID', 'con_MEM_ID', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE',
                 'cat_MEM_STATUS', 'con_AGE_YRS', 'cat_GENDER', 'cat_RELIGION', 'cat_EMPLOYMENT_STATUS',
                 'cat_PLACE_OF_WORK', 'con_TS_ON_TRAVEL',
                 'con_TS_ON_OUTDOOR_SPORTS', 'cat_IS_HOSPITALISED', 'cat_HAS_BANK_AC', 'cat_HAS_MOBILE',
                 'cat_IS_HEALTHY', 'con_TOT_INC_1', 'con_TOT_INC_2', 'con_TOT_INC_3', 'con_TOT_INC_4',
                 'con_INC_OF_HH_FRM_ALL_SRCS_1','con_INC_OF_HH_FRM_ALL_SRCS_2','con_INC_OF_HH_FRM_ALL_SRCS_3', 
                 'con_INC_OF_HH_FRM_ALL_SRCS_4','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_2','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_4','con_INC_OF_MEM_FRM_ALL_SRCS_1', 
                 'con_INC_OF_MEM_FRM_ALL_SRCS_2','con_INC_OF_MEM_FRM_ALL_SRCS_3', 'con_INC_OF_MEM_FRM_ALL_SRCS_4']

final_c18.to_csv("Completed_data/C18/C18_all.csv", index=False)
dataset_state_c18 = split_and_save_by_state(final_c18, 18, "Completed_data/C18/STATE/")

################################# C22 #######################################
print("Start processing C22.")
final_c22 = combine_dataframes(c22, income)
final_c22 = convert_to_int(final_c22, ['WAVE_NO', 'HH_ID', 'MEM_ID', 'AGE_YRS', 
                       'IS_HOSPITALISED', 'WAS_HOSPITALISED_0','WAS_HOSPITALISED_1', 
                       'WAS_HOSPITALISED_NAN','HAS_BANK_AC', 'HAS_MOBILE',
                       'IS_HEALTHY', 'TOT_INC_1', 'TOT_INC_2', 'TOT_INC_3', 'TOT_INC_4',
                       'INC_OF_HH_FRM_ALL_SRCS_1', 'INC_OF_HH_FRM_ALL_SRCS_2',
                       'INC_OF_HH_FRM_ALL_SRCS_3', 'INC_OF_HH_FRM_ALL_SRCS_4',
                       'INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 'INC_OF_ALL_MEMS_FRM_ALL_SRCS_2',
                       'INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 'INC_OF_ALL_MEMS_FRM_ALL_SRCS_4',
                       'INC_OF_MEM_FRM_ALL_SRCS_1', 'INC_OF_MEM_FRM_ALL_SRCS_2',
                       'INC_OF_MEM_FRM_ALL_SRCS_3', 'INC_OF_MEM_FRM_ALL_SRCS_4'])
final_c22.columns = ['con_WAVE_NO', 'con_HH_ID', 'con_MEM_ID', 'cat_STATE', 'cat_HR', 'cat_REGION_TYPE',
                 'cat_MEM_STATUS', 'con_AGE_YRS', 'cat_GENDER', 'cat_RELIGION', 'cat_EMPLOYMENT_STATUS',
                 'cat_PLACE_OF_WORK', 'con_TS_ON_WORK_FOR_EMPLOYER','con_TS_ON_TRAVEL', 
                 'con_TS_ON_OUTDOOR_SPORTS', 'cat_IS_HOSPITALISED', 'cat_WAS_HOSPITALISED_0', 
                 'cat_WAS_HOSPITALISED_1', 'cat_WAS_HOSPITALISED_NAN', 'cat_HAS_BANK_AC', 'cat_HAS_MOBILE',
                 'cat_IS_HEALTHY', 'con_TOT_INC_1', 'con_TOT_INC_2', 'con_TOT_INC_3', 'con_TOT_INC_4',
                 'con_INC_OF_HH_FRM_ALL_SRCS_1','con_INC_OF_HH_FRM_ALL_SRCS_2','con_INC_OF_HH_FRM_ALL_SRCS_3', 
                 'con_INC_OF_HH_FRM_ALL_SRCS_4','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_1', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_2','con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_3', 
                 'con_INC_OF_ALL_MEMS_FRM_ALL_SRCS_4','con_INC_OF_MEM_FRM_ALL_SRCS_1', 
                 'con_INC_OF_MEM_FRM_ALL_SRCS_2','con_INC_OF_MEM_FRM_ALL_SRCS_3', 'con_INC_OF_MEM_FRM_ALL_SRCS_4']

final_c22.to_csv("Completed_data/C22/C22_all.csv", index=False)
dataset_state_c22 = split_and_save_by_state(final_c22, 22, "Completed_data/C22/STATE/")

################################ summary count for each state for each dataset #################
df_c19 = pd.DataFrame.from_dict(dataset_state_c19, orient='index', columns=["C19"])
df_c18 = pd.DataFrame.from_dict(dataset_state_c18, orient='index', columns=["C18"])
df_c22 = pd.DataFrame.from_dict(dataset_state_c22, orient='index', columns=["C22"])
merge_df = pd.merge(df_c19, df_c18, left_index = True, right_index = True)
merge_df = pd.merge(merge_df, df_c22, left_index = True, right_index = True)
merge_df.to_csv("state_count_summary.csv")
