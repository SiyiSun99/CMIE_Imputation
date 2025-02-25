#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Calculate RMSE and accuracy metrics for imputed datasets across different scenarios.
@Author      :   siyi.sun
@Time        :   2025/02/24 01:52:56
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob


def z_score_normalize(data, mean, std):
    """
    Apply z-score normalization using pre-computed mean and standard deviation
    """
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std


def calculate_imputation_metrics(base_path):
    """
    Calculate RMSE and accuracy metrics for imputed datasets across different scenarios.
    The imputed datasets only contain the previously missing values.

    Args:
        base_path (Path): Base path containing all the data directories

    Returns:
        pd.DataFrame: Results table with metrics for each scenario
    """
    cohorts = ["C19"]
    missing_methods = ["MCAR", "MAR", "MNAR"]
    missing_ratios = [10, 20, 30, 40, 50]
    sample_time = 5
    impute_methods = ["mice"]

    results = []

    for method in impute_methods:
        for cohort in cohorts:
            # Load full dataset
            full_data_path = base_path / f"Completed_data/{cohort}/{cohort}_all.csv"
            full_data = pd.read_csv(full_data_path)

            # Get continuous and categorical columns
            cont_cols = [col for col in full_data.columns if col.startswith("con_")]
            cat_cols = [col for col in full_data.columns if col.startswith("cat_")]

            # Calculate mean and standard deviation for each continuous column in full dataset
            cont_means = {col: full_data[col].mean() for col in cont_cols}
            cont_stds = {col: full_data[col].std() for col in cont_cols}

            for miss_method in missing_methods:
                for miss_ratio in missing_ratios:
                    cont_rmse_list = []
                    cat_acc_list = []

                    # Process each sample for each scenario
                    for sample_idx in range(sample_time):
                        # Load missing mask
                        mask_path = (
                            Path(base_path)
                            / f"data_miss_mask/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{sample_idx}.csv"
                        )
                        mask_data = pd.read_csv(mask_path)

                        # Load imputed data
                        imputed_path = (
                            Path(base_path)
                            / f"data_mice/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{sample_idx}.csv"
                        )
                        imputed_data = pd.read_csv(imputed_path)

                        # Calculate RMSE for continuous variables
                        if cont_cols:
                            cont_rmse = 0
                            total_missing = 0

                            for col in cont_cols:
                                # Get indices where values were missing
                                missing_idx = mask_data[col] == 1
                                n_missing = missing_idx.sum()

                                if n_missing > 0:
                                    # Normalize both full and imputed data using full data's min-max
                                    full_normalized = z_score_normalize(
                                        full_data.loc[missing_idx, col],
                                        cont_means[col],
                                        cont_stds[col],
                                    )
                                    imputed_normalized = z_score_normalize(
                                        imputed_data.loc[missing_idx, col],
                                        cont_means[col],
                                        cont_stds[col],
                                    )
                                    # Calculate squared errors using normalized values
                                    squared_errors = (
                                        full_normalized - imputed_normalized
                                    ) ** 2
                                    cont_rmse += squared_errors.sum()
                                    total_missing += n_missing

                            if total_missing > 0:
                                cont_rmse = np.sqrt(cont_rmse / total_missing)
                                cont_rmse_list.append(cont_rmse)

                        # Calculate accuracy for categorical variables
                        if cat_cols:
                            cat_acc = 0
                            total_missing = 0

                            for col in cat_cols:
                                missing_idx = mask_data[col] == 1
                                n_missing = missing_idx.sum()

                                if n_missing > 0:
                                    # Calculate accuracy for missing values
                                    correct = (
                                        full_data.loc[missing_idx, col]
                                        == imputed_data.loc[missing_idx, col]
                                    ).sum()
                                    cat_acc += correct
                                    total_missing += n_missing

                            if total_missing > 0:
                                cat_acc = cat_acc / total_missing
                                cat_acc_list.append(cat_acc)

                    # Calculate mean and variance of metrics
                    cont_rmse_mean = (
                        np.mean(cont_rmse_list) if cont_rmse_list else np.nan
                    )
                    cont_rmse_var = np.var(cont_rmse_list) if cont_rmse_list else np.nan
                    cat_acc_mean = np.mean(cat_acc_list) if cat_acc_list else np.nan
                    cat_acc_var = np.var(cat_acc_list) if cat_acc_list else np.nan

                    # Add results to list
                    results.append(
                        {
                            "impute_method": method,
                            "cohort": cohort,
                            "missing_mechanism": miss_method,
                            "missing_ratio": miss_ratio,
                            "mean_continuous_rmse": cont_rmse_mean,
                            "var_continuous_rmse": cont_rmse_var,
                            "mean_categorical_accuracy": cat_acc_mean,
                            "var_categorical_accuracy": cat_acc_var,
                        }
                    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# Example usage:
base_path = Path("/home/siyi.sun/CMIE_Project/data_stored")
results = calculate_imputation_metrics(base_path)
output_path = "/home/siyi.sun/CMIE_Project/imputation_metrics_results.csv"
results.to_csv(output_path, index=False)
