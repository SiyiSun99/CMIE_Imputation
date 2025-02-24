#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   store the performance of the imputation
@Author      :   siyi.sun
@Time        :   2025/02/21 01:01:37
"""

import numpy as np
import pandas as pd
from pathlib import Path


class Performance_store:
    def __init__(
        self,
        base_path: Path,  # Path to the base directory where the data will be stored
        cohort: str,
        miss_method: str,
        miss_ratio: float,
        index_file: str,
        label_reverse: dict,
        label_ori: dict,
        column_location: list,
        column_name: list,
        name: str,
        mode: str = "one_hot",
        index_pick: str = "continuous_first",
    ):
        """Initialize the Performance_store object"""
        self.base_path = base_path
        self.cohort = cohort
        self.mode = mode
        self.miss_method = miss_method
        self.miss_ratio = miss_ratio
        self.index_file = index_file
        self.label_reverse = label_reverse
        self.label_ori = label_ori
        self.column_location = column_location
        self.column_name = column_name
        self.name = name
        self.index_pick = index_pick

    def decode_column(self, index, values):
        """Convert encoded values back to their original form"""
        if self.label_reverse[index][0] == "con":
            # Handle continuous values
            max_val, min_val = self.label_reverse[index][1]
            # Rescale to original range
            original_values = np.round(values * (max_val - min_val) + min_val, 1)
            return original_values.flatten().tolist()
        else:
            # Handle categorical values
            dictionary = self.label_reverse[index][1][1]
            # Convert one-hot encoded values back to categories
            indices = np.argmax(values, axis=1)
            return [dictionary[idx] for idx in indices]

    def create_imputed_dataframe(self, data):
        """Convert imputed data matrix to a dataframe with original column types"""
        df_imputed = pd.DataFrame()

        for i, col_name in enumerate(self.column_name):
            col_data = (
                data[:, : self.column_location[0]]
                if i == 0
                else data[:, self.column_location[i - 1] : self.column_location[i]]
            )
            df_imputed[col_name] = self.decode_column(i, col_data)

        return df_imputed

    def save_results(self, imputed_data, mask_df):
        """Save imputed data to the appropriate directories"""
        # Construct base folder path
        save_folder = (
            self.base_path
            / f"data_gain/{self.cohort}/{self.cohort}_all/{self.miss_method}/miss{self.miss_ratio}"
        )

        # Create directories if they do not exist
        save_folder.mkdir(parents=True, exist_ok=True)

        # Create full save path with index file
        save_path = save_folder / f"{self.index_file}.csv"

        # Generate and save imputed dataframe
        df_imputed = self.create_imputed_dataframe(imputed_data)
        df_imputed = df_imputed.mask(~mask_df)
        df_imputed.to_csv(save_path, index=False)

    def select_best_index(self, continuous_metrics, categorical_accuracies):
        """Select the best index based on the strategy"""
        if self.index_pick == "continuous_first":
            # Select index with minimum continuous error
            if continuous_metrics:
                return continuous_metrics.index(min(continuous_metrics)) + 1
        else:
            if not categorical_accuracies:
                raise ValueError("categorical_accuracies is empty")
            return categorical_accuracies.index(max(categorical_accuracies)) + 1
