# Logistic regression model

from sklearn.linear_model import LogisticRegression


def train_Logistic(X_train, y_train, pen='l2', max_iter=1000, C=1.0, solver='lbfgs'):
    '''Returns the logistic regression model and print the predictions.'''
    
    # create the logistic
    logreg = LogisticRegression(penalty=pen, n_jobs=-1, max_iter=max_iter, C=C, solver=solver)

    # fit the model
    logreg.fit(X_train, y_train)
    
    # return the model
    return logreg