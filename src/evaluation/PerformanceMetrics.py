# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, recall_score, precision_score
from sklearn_evaluation.plot import ConfusionMatrix



# function for plotting the roc curve
def plot_roc(model, X_test, y_test):
    '''Returns the roc curve for the model.'''

    # predict the probabilities
    y_pred_proba = model.predict(X_test)

    # calculate the fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # calculate the auc
    roc_auc = auc(fpr, tpr)
    
    # plot the roc curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')

    # set labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # set title
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # show the auc score
    plt.text(0.6, 0.2, 'AUC: '+str(round(roc_auc, 2)), fontsize=12)
    
    return fig


def metrics(model, X_test, y_test, new_threshold = 0.5):
    '''Returns the confusion matrix, recall, precision, and f1 score for the model.'''


    # predict the probabilities
    y_pred = model.predict(X_test)

    # Apply the new threshold to determine class labels
    y_pred = (y_pred >= new_threshold).astype(int)

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # calculate the recall
    recall = recall_score(y_test, y_pred)

    # calculate the precision
    precision = precision_score(y_test, y_pred)

    # calculate the f1 score
    f1 = f1_score(y_test, y_pred)

    # return the metrics
    return conf_matrix, recall, precision, f1


def optimal_threshold(model, X_test, y_test):

    y_pred_proba = model.predict(X_test)

    # calculate the fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    '''Returns the optimal threshold for the roc curve.'''
    # maximize the sum of TPR and 1 - FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, optimal_idx