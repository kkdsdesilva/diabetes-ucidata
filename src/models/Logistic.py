# Logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_LogReg(X_train, y_train, pen='l2', max_iter=1000, C=1.0, solver='lbfgs'):
    '''Returns the logistic regression model and print the predictions.'''
    
    # create the logistic
    logreg = LogisticRegression(penalty=pen, n_jobs=-1, max_iter=max_iter, C=C, solver=solver)

    # fit the model
    logreg.fit(X_train, y_train)
    
    # return the model
    return logreg


def evaluate_model(model, X_train, X_test, y_train, y_test):
    '''Returns the accuracy of the model.'''
    
    # predict values
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # prediction accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # print the accuracy
    print('Train accuracy: ', train_acc)
    print('Test accuracy: ', test_acc)

    # return the accuracy
    return train_acc, test_acc