#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   combine the data from mice imputation
@Author      :   siyi.sun
@Time        :   2025/02/18 22:49:25
"""
# export PATH="/home/siyi.sun/miniconda3/bin:$PATH"

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import gc

def combine_mice_imputations(cohort, miss_method, miss_ratio, index_file, n_imputations=5):
    """
    Combine multiple MICE-imputed datasets with specific handling for continuous and categorical variables.
    
    Args:
        cohort (str): Cohort identifier
        miss_method (str): Missing data method identifier
        miss_ratio (str): Missing ratio identifier
        index_file (str): Index file identifier
        n_imputations (int): Number of imputations to combine (default=5)
    
    Returns:
        pd.DataFrame: Combined dataset with imputed values
    """
    # Define paths
    base_path = Path('/home/siyi.sun/CMIE_Project/data_stored')
    input_pattern = f"data_mice_store/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{index_file}"
    output_pattern = f"data_mice/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}"
    mask_path = f"data_miss_mask/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{index_file}.csv"
    
    # Create all necessary directories in the output path
    output_dir = base_path / output_pattern
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all imputed datasets
    imputed_dfs = []
    for i in range(n_imputations):
        file_path = base_path / input_pattern / f"{i}.csv"
        df = pd.read_csv(file_path)
        imputed_dfs.append(df)
    
    # Read the mask
    mask_df = pd.read_csv(base_path / mask_path)
    mask_df = mask_df.astype(bool)
    
    # Initialize the combined DataFrame with the structure of the first imputed dataset
    combined_df = pd.DataFrame(index=imputed_dfs[0].index, columns=imputed_dfs[0].columns)
    
    # Process each column
    for column in combined_df.columns:
        if column.startswith('con_'):  # Continuous variables
            # Calculate mean across all imputations
            column_values = np.mean([df[column].values for df in imputed_dfs], axis=0)
            
        elif column.startswith('cat_'):  # Categorical variables
            # Get most common value for each position
            column_values = []
            for pos in range(len(combined_df)):
                values = [df[column].iloc[pos] for df in imputed_dfs]
                most_common = Counter(values).most_common(1)[0][0]
                column_values.append(most_common)
            
        else:  # Other columns (copy from first imputation)
            column_values = imputed_dfs[0][column].values
        
        combined_df[column] = column_values
    
    # Save the combined dataset
    output_file = output_dir / f"{index_file}.csv"
    combined_df = combined_df.mask(~mask_df,0)
    combined_df.to_csv(output_file, index=False)
    
    return combined_df

cohorts = ["C19"]
miss_ratios = [10,20,30,40,50]
miss_methods = ["MNAR"]
Sampletime = 5
for cohort in cohorts:
    for miss_method in miss_methods:
        for miss_ratio in tqdm(miss_ratios):  
            for index_file in tqdm(range(Sampletime)):
                combined_df = combine_mice_imputations(
                    cohort=cohort,
                    miss_method=miss_method,
                    miss_ratio=miss_ratio,
                    index_file=index_file
                )
                    
gc.collect()