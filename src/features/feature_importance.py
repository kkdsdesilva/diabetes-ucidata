# function for feature importance

# import libraries
import pandas as pd

# feature_importance: Returns the feature importance of the model.
def feature_importance(model, data):
    '''Returns the feature importance of the model.'''
    
    # create a dataframe of the feature importance
    importance = pd.DataFrame(model.feature_importances_, index=data.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    # return the feature importance
    return importance