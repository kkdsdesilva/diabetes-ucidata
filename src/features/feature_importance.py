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


# pick the top k features
def pick_top_k_features(data, importance, k=10):
    '''Returns the top k features.'''

    # get the absolute value of the importance
    importance['importance'] = abs(importance['importance'])
    
    # pick the top k features
    top_k_features = importance.head(k)

    # subset data with the top k features
    data_top = data[top_k_features.index]

    # return the top k features
    return data_top