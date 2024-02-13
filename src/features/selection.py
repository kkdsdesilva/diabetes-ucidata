# function for feature importance

# import libraries
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def select_features(model, X_train, X_test, y_train, step=1, cv=5):
    '''Returns the most important features using the Recursive Feature Elimination Cross Validation (RFECV) method.'''
    
    original_num_features = X_train.shape[1]

    if model == 'Logistic':
        model = LogisticRegression()
    
    elif model == 'DecisionTree':
        model = DecisionTreeClassifier()

    elif model == 'RandomForest':
        model = RandomForestClassifier(n_jobs=-1)

    # create the RFECV model
    selector = RFECV(model, step=step, cv=cv)

    # fit the model
    selector = selector.fit(X_train, y_train)

    # new X_train and X_test
    X_train = X_train[X_train.columns[selector.support_]]
    X_test = X_test[X_test.columns[selector.support_]]
    
    print('-'*20)
    print('Feature selection complete.')
    print('Original number of features:', original_num_features)
    print('Optimal number of features:', X_train.shape[1])
    print('-'*20)

    return X_train, X_test

