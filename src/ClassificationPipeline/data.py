import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

def fetch_csv(config):
    """
    Reads a CSV file from a given path specified in the configuration dictionary and returns a DataFrame.

    Parameters:
    config (dict): A dictionary containing configuration details. It must have a key 'Csv_path' that specifies the path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data read from the CSV file specified in the 'Csv_path'.
    
    If the 'Csv_path' is an empty string, the function prints an error message and does not return any DataFrame.
"""
    categorical_feats = config['Categorical_features']
    numeric_feats = config['numeric_features']
    util_feats = config['util_features']
    target_metric = config['Target_metric']
    
    features = categorical_feats + numeric_feats + util_feats + target_metric
    # remove None from my features
    all_features = [x for x in features if str(x) != 'None']
    if config['Csv_path'] == "":
        print("No csv path given")
    else:
        x = pd.read_csv(config['Csv_path'], usecols = all_features)
        return x


def data_summary(df,config):
    """
     Prints a summary of the given DataFrame, including the total number of rows and the number of missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.
    config (dict): A configuration dictionary (currently not used within the function but included for potential future use or compatibility).

    Returns:
    None: This function prints the summary directly and does not return any value.
    """
    print(f"The total number of rows of this data is:{df.shape[0]}")
    if df.isna().sum().reset_index()[0].sum() == 0:
        print("There are no missing values in the provided dataframe")
    else:
        print(f"There is a total of {df.isna().sum()} missing values in the dataset")


def target_metric_check_convert(df,config):

    """
    Checks if the target metric column in the DataFrame is binary and converts it to binary values (0 and 1) if necessary.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the target metric column.
    config (dict): A dictionary containing configuration details. It must have a key 'Target_metric' that specifies the target column.

    Returns:
    pd.Series: A Series containing the converted binary target metric values.
    """
    target_metric = config['Target_metric']
    print("Checking whether target metric is binary")
   
    if df[target_metric].nunique().iloc[0] != 2:
        raise ValueError("Target column must have exactly two unique values.")
    else:    
        print("Target metric is binary.")

    # Check if unique values are already 0 and 1
    unique_values = df[target_metric].iloc[:,0].unique()
    if set(unique_values) != {0, 1}:
        print("Converting target metric to binary values 0 and 1.")
        
        # Map original values to 0 and 1
        value_map = {unique_values[0]: 0, unique_values[1]: 1}
        y = df[target_metric].iloc[:,0]
        y = y.map(value_map)
    
    return y
    

