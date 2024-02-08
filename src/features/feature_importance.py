# function for feature importance

# import libraries
import pandas as pd

# feature_importance: Returns the feature importance of the supported models.
def feature_importance(model, data):
    '''Returns the feature importance of the model.'''
    
    # create a dataframe of the feature importance
    importance = pd.DataFrame(model.feature_importances_, index=data.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    # return the feature importance
    return importance

# feature importance for logistic regression
def feature_importance_logreg(model, data):
    '''Returns the feature importance of the logistic regression model.'''
    
    # create a dataframe of the feature importance
    importance = pd.DataFrame(model.coef_.T, index=data.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    # return the feature importance
    return importance