#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   
@Author      :   siyi.sun
@Time        :   2024/12/09 15:40:24
"""
import pandas as pd
data = pd.read_csv('data_need.csv', usecols=[0,1,2], index_col=0)
data.columns = ['WAVE_NO','HH_ID']
data = data.drop_duplicates()

# Ensure the data is sorted by WAVE_NO and HH_ID
data = data.sort_values(by=['WAVE_NO', 'HH_ID'])

# Create a set of all households observed in previous waves
all_previous_households = set()

# Initialize a dictionary to store the results
wave_stats = []

# Iterate through each wave
for wave, group in data.groupby('WAVE_NO'):
    # Get households in the current wave
    current_households = set(group['HH_ID'])
    
    # Calculate the number of households taking the survey in the current wave
    num_current = len(current_households)
    
    # Calculate disappeared households
    if wave > data['WAVE_NO'].min():
        prev_wave_households = data[data['WAVE_NO'] == wave - 1]['HH_ID']
        disappeared = set(prev_wave_households) - current_households
        num_disappeared = len(disappeared)
    else:
        num_disappeared = 0

    # Calculate new households
    new_households = current_households - all_previous_households
    num_new = len(new_households)
    
    # Calculate households that are back
    if wave > data['WAVE_NO'].min():
        back_households = all_previous_households - set(prev_wave_households)
        num_back = len(back_households & current_households)
    else:
        num_back = 0
    
    # Update the set of all previous households
    all_previous_households.update(current_households)
    
    # Append the stats for this wave
    wave_stats.append({
        'WAVE_NO': wave,
        'Total': num_current,
        'Disappeared': num_disappeared,
        'New_IN': num_new,
        'Back': num_back
    })

# Convert the results to a DataFrame
wave_stats_df = pd.DataFrame(wave_stats)

# Save the results to a CSV file
wave_stats_df.to_csv('wave_stats.csv', index=False)
