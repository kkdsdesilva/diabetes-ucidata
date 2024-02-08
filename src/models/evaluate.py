# evaluate a given model

# import libraries
from sklearn.metrics import accuracy_score

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