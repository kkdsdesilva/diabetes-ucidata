# Logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def LogReg(X_train, X_test, y_train, y_test, pen='l2', max_iter=1000):
    '''Returns the logistic regression model and print the predictions.'''
    
    # create the logistic
    logreg = LogisticRegression(penalty=pen, n_jobs=-1, max_iter=max_iter)

    # fit the model
    logreg.fit(X_train, y_train)

    # make predictions
    y_train_pred = logreg.predict(X_train)
    y_test_pred = logreg.predict(X_test)

    # prediction accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # print the accuracy
    print('Train accuracy: ', train_acc)
    print('Test accuracy: ', test_acc)

    # return the model
    return logreg