import pandas as pd

def transform_data(data_1):
    """
    Transform dataframe with wave mapping, relative month calculation, and multiindex restructuring.
    
    Args:
        data_1: DataFrame with columns HH_ID, MONTH, income columns, and MEM_ID
    
    Returns:
        pandas.DataFrame: Transformed dataframe with multiindex and unstacked relative months
    """
    # Create a copy to avoid modifying original
    df = data_1.copy()
    
    # 1. Month to Wave mapping
    def get_wave_number(month_year):
        month_to_num = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month, year = month_year.split()
        month_num = month_to_num[month]
        year = int(year)
        wave_in_year = (month_num - 1) // 4 + 1
        waves_before = (year - 2014) * 3
        return waves_before + wave_in_year
    
    # Add WAVE_NO column
    df['WAVE_NO'] = df['MONTH'].apply(get_wave_number)
    
    # 2. Add Relative_Month column
    def get_relative_month(month_year):
        month, _ = month_year.split()
        # First month of each wave period (Jan, May, Sep)
        if month in ['Jan', 'May', 'Sep']:
            return 1
        # Second month of each wave period (Feb, Jun, Oct)
        elif month in ['Feb', 'Jun', 'Oct']:
            return 2
        # Third month of each wave period (Mar, Jul, Nov)
        elif month in ['Mar', 'Jul', 'Nov']:
            return 3
        # Fourth month of each wave period (Apr, Aug, Dec)
        else:
            return 4
    
    df['Relative_Month'] = df['MONTH'].apply(get_relative_month)
    
    # 3. Set multiindex and unstack
    # First set the multiindex
    df = df.set_index(['WAVE_NO', 'HH_ID', 'MEM_ID', 'Relative_Month'])
    
    # List of columns to unstack (excluding the index columns)
    cols_to_unstack = ['TOT_INC', 'INC_OF_HH_FRM_ALL_SRCS', 
                      'INC_OF_ALL_MEMS_FRM_ALL_SRCS', 'INC_OF_MEM_FRM_ALL_SRCS']
    
    # Unstack the Relative_Month
    df_unstacked = df[cols_to_unstack].unstack('Relative_Month')
    
    # Rename columns to be more descriptive
    # This will create columns like ('TOT_INC', 1), ('TOT_INC', 2), etc.
    
    return df_unstacked

# Example usage:
# transformed_data = transform_data(data_1)