import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from ClassificationPipeline import data, features, model, evaluate

class ClassificationPipeline:
    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.categorical_feats = config['Categorical_features']
        self.numeric_feats = config['numeric_features']
        self.util_feats = config['util_features']
        self.target_metric = config['Target_metric']
        self.features_ = self.categorical_feats + self.numeric_feats + self.util_feats + self.target_metric
        self.all_features = [x for x in self.features_ if str(x) != 'None']

    def run(self):
        # Fetch the CSV (only the util features + categorical features + numeric features + target metric defined in configs)
        x = data.fetch_csv(self.config)
        
        # Check total rows of the CSV and check if there's any missing values in the dataframe
        data.data_summary(x, self.config)

        # Check if the metric is binary. If it's not already in form of 0 and 1, then force map it to 0 and 1
        y = data.target_metric_check_convert(x, self.config)

        # Create dataframes that can be replaced in case of imputation
        r = pd.DataFrame()
        s = pd.DataFrame()

        if all(feat is not None for feat in self.numeric_feats) and x[self.numeric_feats].isna().sum().reset_index()[0].sum() != 0:
            r = features.impute_numeric_feature(x, self.config)
        elif all(feat is not None for feat in self.numeric_feats):
            r = x[self.numeric_feats]

        if all(feat is not None for feat in self.categorical_feats) and x[self.categorical_feats].isna().sum().reset_index()[0].sum() != 0:
            s = features.impute_categorical_feature(x, self.config)
        elif all(feat is not None for feat in self.categorical_feats):
            s = x[self.categorical_feats]

        # Concat both dataframes where missing values are imputed
        df_new = pd.concat([r, s], axis='columns')

        # Encode Categorical Features
        df_final = features.encode_features(df_new, self.config)

        # Train test split
        X_train, X_test, y_train, y_test = model.train_test_split(df_final, y, self.config)
        print("Done with Train test split")

        # Run the model
        lgbm_model = model.run_lightgbm(X_train, y_train, self.config)

        # Look at the model evaluation
        z = evaluate.model_evaluation(X_test, y_test, lgbm_model['model'], self.config)

        return z