#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Preprocessing Income household data (income_hh.csv).
@Author      :   siyi.sun
@Time        :   2025/02/02 09:52:41
"""

import pandas as pd
import numpy as np

def get_wave_number(month_year):
    """
    Convert 'Month Year' format to wave number.
    Wave 1 starts from Jan 2014, with 3 waves per year.
    Jan-Apr: Wave 1
    May-Aug: Wave 2
    Sep-Dec: Wave 3
    
    Args:
        month_year (str): String in format 'Month Year' (e.g., 'Jan 2014')
    
    Returns:
        int: Wave number
    """
    # Convert month names to numbers (1-12)
    month_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Split the month and year
    month, year = month_year.split()
    month_num = month_to_num[month]
    year = int(year)
    
    # Calculate wave within year (1, 2, or 3)
    wave_in_year = (month_num - 1) // 4 + 1
    
    # Calculate total waves since start (2014)
    waves_before = (year - 2014) * 3
    
    # Final wave number
    wave_number = waves_before + wave_in_year
    
    return wave_number

# Apply to your dataframe
def apply_wave_mapping(data):
    """
    Apply wave mapping to a dataframe with a 'Month' column.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Month' column in 'Month Year' format
    
    Returns:
        pd.DataFrame: Original dataframe with new 'Wave' column
    """
    data['WAVE_NO'] = data['MONTH'].apply(get_wave_number)
    return data.drop("MONTH",axis = 1)

income_hh = pd.read_csv("income_hh.csv", header = 0, index_col=0)
income_hh_na = income_hh.replace([-99],np.nan)
income_hh_na.dropna(inplace = True)
apply_wave_mapping(income_hh_na).to_csv("completed_income_hh.csv")
