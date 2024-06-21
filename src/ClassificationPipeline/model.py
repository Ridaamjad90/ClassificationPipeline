import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import lightgbm as lgb
import sklearn
#from lightgbm import LGBMCLassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.model_selection import ParameterGrid, KFold

from sklearn.model_selection import train_test_split as sk_train_test_split

def train_test_split(X, y, config):
    """
    Splits the dataset into training and testing sets based on the specified configuration.

    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target variable.
    config (dict): A dictionary containing configuration details. It must have a key 'train_test_split' specifying the train-test split ratio as a float between 0 and 1.

    Returns:
    tuple: A tuple containing four elements:
        - X_train (pd.DataFrame): Training feature matrix.
        - X_test (pd.DataFrame): Testing feature matrix.
        - y_train (pd.Series): Training target variable.
        - y_test (pd.Series): Testing target variable.

    Raises:
    ValueError: If the 'train_test_split' configuration is missing, not a float, or not within the range (0, 1).
    """
    if config['train_test_split'] is None:
        raise ValueError("The 'train_test_split' configuration is missing.")

    if not isinstance(config['train_test_split'], float):
        raise ValueError("The 'train_test_split' configuration is not a float.")

    if not (0 < config['train_test_split'] < 1):
        raise ValueError("The 'train_test_split' configuration must be greater than 0 and less than 1.")

    test_size = 1 - config['train_test_split']
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size = test_size, random_state = 1)

    return X_train, X_test, y_train, y_test
    

def run_lightgbm(X,y, config):
    """
    Trains a LightGBM model using the provided feature matrix and target variable. It performs a randomized search
    for hyperparameter tuning and fits the final model with the best parameters.

    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target variable.
    config (dict): A dictionary containing configuration details. The dictionary should include the list of categorical features
                   as `categorical_feats` and model parameters as `params`.

    Returns:
    dict: A dictionary containing the randomized search object (`search`) and the trained model (`model`).

    """
    
    if not categorical_feats or all(feat is None for feat in categorical_feats):
        categorical_cols = []
    else:
        categorical_cols = categorical_feats
    params['metric'] = ['auc']
    neg_pos_ratio = np.sum(y==0)/np.sum(y==1)
    params['scale_pos_weight'] = [1, neg_pos_ratio, neg_pos_ratio*2, neg_pos_ratio*4]
    print(categorical_cols)

#    y = pd.Series(y)
    n_rows = X.shape[0]
    
    model = lgb.LGBMClassifier()
# Add this if you want to be able to run GridSearchCV 
#    if n_rows < 50000:
#        print("Dataset has less than 50000 rows, Using GridSearchCV.")
#        search = GridSearchCV(model, params, cv = 5,scoring = 'roc_auc',
#                              n_jobs = -1)
#    else:
#        n_iter = 50
#        search = RandomizedSearchCV(model, params, cv = 5, n_jobs = -1, random_state =1 )

    n_iter = 50
    search = RandomizedSearchCV(model, params, cv = 5, n_jobs = -1, random_state =1 )
    search.fit(X,y, categorical_feature = categorical_cols )
    print("The highest AUC Score achieved" %  search.best_score_)
    final_params = search.best_estimator_.get_params()
    #Fit model with final parameters

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X,y,categorical_feature = categorical_cols)
    return {'search': search,
           'model': model}
    
