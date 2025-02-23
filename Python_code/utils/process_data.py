#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Process data files and prepare them for imputation
@Author      :   siyi.sun
@Time        :   2025/02/21 01:02:14
"""
import os
import numpy as np
import pandas as pd


def create_category_mappings(column_values):
    """Create dictionaries to map categorical values to indices and back"""
    unique_values = list(set([v for v in column_values if str(v) != "nan"]))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    index_to_value = {idx: val for idx, val in enumerate(unique_values)}

    return value_to_index, index_to_value


def encode_categorical(values, mapping):
    """Encode categorical values using one-hot"""
    encoded = []
    dict_size = len(mapping)

    for val in values:
        if pd.isna(val):
            encoded.append([np.nan] * dict_size)
        else:
            encoded.append(list(np.eye(dict_size)[mapping[val]]))

    return np.array(encoded, dtype=np.float32)


def normalize_continuous(values, mode="one_hot"):
    """Normalize continuous values"""
    values_array = np.array([v for v in values if not pd.isna(v)])

    if len(values_array) == 0:
        return np.zeros((len(values), 1), dtype=np.float32), 0, 0

    max_val = np.max(values_array)
    min_val = np.min(values_array)

    if min_val == max_val:
        return np.zeros((len(values), 1), dtype=np.float32), min_val, max_val

    normalized = []
    for val in values:
        if pd.isna(val):
            normalized.append(np.nan)
        else:
            norm_val = (val - min_val) / (max_val - min_val)
            normalized.append(norm_val)

    return np.array(normalized, dtype=np.float32).reshape(-1, 1), min_val, max_val


def processed_data(
    base_path,
    cohort,
    miss_method,
    miss_ratio,
    index_file,
    mode="one_hot",
    sampletest=False,
):
    """Process data files and prepare them for imputation"""
    # Load data
    # Full data path
    full_path = base_path / f"Completed_data/{cohort}/{cohort}_all.csv"
    if sampletest:
        # Missing data mask path
        mask_path = (
            base_path
            / f"data_miss_mask_sample/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{index_file}.csv"
        )
    else:
        # Missing data mask path
        mask_path = (
            base_path
            / f"data_miss_mask/{cohort}/{cohort}_all/{miss_method}/miss{miss_ratio}/{index_file}.csv"
        )

    # Load data
    df_full = pd.read_csv(full_path)
    df_mask = pd.read_csv(mask_path)
    df_mask = df_mask.astype(bool)
    df_miss = df_full.mask(df_mask)

    # Identify column types
    con_cols = [col for col in df_mask.columns if col.startswith("con")]
    cat_cols = [col for col in df_mask.columns if col.startswith("cat")]

    # Process all columns
    column_info_miss = []
    column_info_full = []
    feature_boundaries = []
    current_boundary = 0
    processed_features = None

    for col in df_full.columns:
        miss_values = df_miss[col].tolist()
        full_values = df_full[col].tolist()

        if col in cat_cols:
            # Process categorical column
            miss_mapping, miss_inverse = create_category_mappings(miss_values)
            full_mapping, full_inverse = create_category_mappings(full_values)
            encoded = encode_categorical(miss_values, miss_mapping)

            column_info_miss.append(["cat", [miss_mapping, miss_inverse]])
            column_info_full.append(["cat", [full_mapping, full_inverse]])

        elif col in con_cols:
            # Process continuous column
            encoded, min_val, max_val = normalize_continuous(miss_values, mode)
            full_min = np.min([v for v in full_values if not pd.isna(v)])
            full_max = np.max([v for v in full_values if not pd.isna(v)])

            column_info_miss.append(["con", [max_val, min_val]])
            column_info_full.append(["con", [full_max, full_min]])

        else:
            # Skip other columns
            continue

        # Update feature array
        if processed_features is None:
            processed_features = encoded
        else:
            processed_features = np.concatenate((processed_features, encoded), axis=1)

        # Update feature boundaries
        new_boundary = current_boundary + encoded.shape[1]
        feature_boundaries.append(new_boundary)
        current_boundary = new_boundary

    return (
        processed_features,
        df_full.columns,
        feature_boundaries,
        column_info_miss,
        df_full,
        df_miss,
        column_info_full,
        df_mask,
    )
