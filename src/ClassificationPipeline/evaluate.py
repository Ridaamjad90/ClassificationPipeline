import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve


def model_evaluation(X_test, y_test, model,config):
    """ 
    Evaluates the performance of a trained model on the test dataset and generates various evaluation metrics and plots.

    Parameters:
    X_test (pd.DataFrame): The feature matrix for the test dataset.
    y_test (pd.Series): The target variable for the test dataset.
    model: The trained model to be evaluated.
    config (dict): A dictionary containing configuration details. It must have a key 'models' specifying the model name.

    """

    
    y = pd.DataFrame()
    y['model_name'] = config['models']
    y['accuracy'] = sklearn.metrics.accuracy_score(y_test,model.predict(X_test))

    # fpr, tpr and thresholds
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test,model.predict(X_test))
    y['auc'] = sklearn.metrics.auc(fpr, tpr)
    y['precision'] = sklearn.metrics.average_precision_score(y_test,model.predict(X_test))
    y['briers_score'] = sklearn.metrics.brier_score_loss(y_test,model.predict(X_test))
    y['f1_score'] = sklearn.metrics.f1_score(y_test,model.predict(X_test))
    print(y)

    #Plot roc_curve
    y_pred_proba = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1]) 
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Breast Cancer Classification')
    plt.legend()
    plt.show()

    #Plot precision recall curve
    yhat = model.predict(X_test)
    lr_probs = model.predict_proba(X_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs[:,1])
    # summarize scores
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Model')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()