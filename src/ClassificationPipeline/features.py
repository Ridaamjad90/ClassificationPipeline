import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.encoding import RareLabelEncoder

class MissingFeaturesError(Exception):
    """Custom exception for missing feature sets."""
    pass

# impute missing values 
def impute_numeric_feature(df, config):
    """
    Imputes missing values in numeric features of the DataFrame based on the specified imputation method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the numeric features to impute.
    config (dict): A dictionary containing configuration details. It must have a key 'numeric_imputation' specifying the imputation method ('mean' or a specific numeric value).

    Returns:
    pd.DataFrame: A DataFrame with the imputed numeric features.

    Raises:
    ValueError: If numeric features specified are not present in the DataFrame or if an invalid imputation method is specified.    
    """
    if numeric_feats is None:
        print("No numeric features to impute") 
        return df

    # Check if numeric features exist in the DataFrame
    missing_cols = [col for col in numeric_feats if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These features are not present in the DataFrame: {missing_cols}")

    #Convert the numeric columns to as numeric first
    df[numeric_feats] = df[numeric_feats].apply(pd.to_numeric, errors = 'coerce')
    
    # Now define the imputation value based on what's provided
    if config['numeric_imputation'] == 'mean':
        impute_values = df[numeric_feats].mean()
    elif isinstance(imputation_method, (int, float)):
        impute_values = imputation_method
    else:
        raise ValueError(f"Invalid imputation method specified: {imputation_method}")    
    
    # Impute the values
    df[numeric_feats] = df[numeric_feats].fillna(impute_values)
        
    return df[numeric_feats]


# impute missing values
def impute_categorical_feature(df, config):
    """
    Imputes missing values in categorical features of the DataFrame with a default value 'unknown'.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the categorical features to impute.
    config (dict): A dictionary containing configuration details (currently not used within the function but included for potential future use or compatibility).

    Returns:
    pd.DataFrame: A DataFrame with the imputed categorical features.

    Raises:
    ValueError: If categorical features specified are not present in the DataFrame. 
    """
    if not categorical_feats or all(feat is None for feat in categorical_feats):
        print("No categorical features to impute.")
        return pd.DataFrame()

    # Check if categorical features exist in the DataFrame
    missing_cols = [col for col in categorical_feats if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These features are not present in the DataFrame: {missing_cols}")

    impute_values = 'unknown'
    df[categorical_feats] = df[categorical_feats].fillna(impute_values)   
    return df[categorical_feats]

# Encoding Function
def encode_features(df, config):
    """
    it transforms features so it can be ingested by a tree model

    params:
    df:dataframe returned from last function (numeric imputation one)
    config: config.yaml file
    """
    categorical_feats = config['Categorical_features']
    numeric_feats = config['numeric_features']
    util_feats = config['util_features']
    target_metric = config['Target_metric']
    
    features_ = categorical_feats + numeric_feats + util_feats + target_metric
    
    # remove None from my features if any
    all_features = [x for x in features_ if str(x) != 'None']

    if not categorical_feats or all(feat is None for feat in categorical_feats):
        print("No categorical features to encode.")
        return df
    else:
    # Check names of all categorical columns that have unique value count greater than 50
        cat_cols_1 = []
        cat_cols_2 = []
        for i in categorical_feats:
            if len(df[i].unique()) >= 50:
                cat_cols_1.append(i)
            else:
                cat_cols_2.append(i)
    
        # Initialize empty DataFrames for transformed columns
        df_new_1 = pd.DataFrame(index=df.index)
        df_new_2 = pd.DataFrame(index=df.index)
    
        # Apply ordinal encoding on cat_cols_2
        if cat_cols_2:
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1111)
            ordinal_encoder.fit(df[cat_cols_2])
            df_new_1 = pd.DataFrame(ordinal_encoder.transform(df[cat_cols_2]), columns=cat_cols_2, index=df.index)
        
        # Apply rare label encoding on cat_cols_1
        if cat_cols_1:
            rare_encoder = RareLabelEncoder(tol=0.05, max_n_categories=50, variables=cat_cols_1, ignore_format=True)
            rare_encoder.fit(df[cat_cols_1])
            df_new_2 = pd.DataFrame(rare_encoder.transform(df[cat_cols_1]), columns=cat_cols_1, index=df.index)
    
        # Concatenate the transformed data
        if not df_new_1.empty and not df_new_2.empty:
            df_new = pd.concat([df_new_1, df_new_2], axis='columns')
        elif not df_new_1.empty:
            df_new = df_new_1
        elif not df_new_2.empty:
            df_new = df_new_2
        else:
            df_new = pd.DataFrame(index=df.index)  # No categorical features to encode
    
        # Add any other columns from the original DataFrame that are not in categorical_feats
        other_columns = df.drop(columns=categorical_feats)
        df_final = pd.concat([df_new, other_columns], axis='columns')
    
        print(df_final.head())
        return df_final


