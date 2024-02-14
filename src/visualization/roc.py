# script for visualizing roc curve

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


# function for plotting the roc curve
def plot_roc(model, X_test, y_test):
    '''Returns the roc curve for the model.'''

    # predict the probabilities
    y_pred_proba = model.predict_proba(X_test)[:,1]

    # calculate the fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # calculate the auc
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # plot the roc curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')

    # set labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # set title
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # show the auc score
    plt.text(0.6, 0.2, 'AUC: '+str(round(auc, 2)), fontsize=12)
    
    plt.show()