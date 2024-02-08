# function to select features using the chi2 test

# import libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# select_features: Returns the selected features using the chi2 test.
def select_features(data, target, k=10):
    '''Returns the selected features using the chi2 test.'''
    
    # select the features
    X_new = SelectKBest(chi2, k=k).fit_transform(data.drop(target, axis=1), data[target])
    
    # return the selected features
    return pd.DataFrame(X_new, columns=data.drop(target, axis=1).columns)