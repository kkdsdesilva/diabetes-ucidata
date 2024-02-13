# evaluate a given model

# import libraries
from sklearn.metrics import accuracy_score, f1_score, recall_score

def evaluate_model(model, X_train, X_test, y_train, y_test, metric='recall'):
    '''Returns the accuracy of the model.'''
    
    if metric == 'accuracy':    
        # predict values
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # prediction accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # print the accuracy
        print('-'*5+'Accuracy'+'-'*5)
        print('Train accuracy: ', train_acc)
        print('Test accuracy: ', test_acc)

        # return the accuracy
        return train_acc, test_acc
    
    elif metric == 'f1':
        # predict values
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # prediction accuracy
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # print the accuracy
        print('-'*5+'F1 Score'+'-'*5)
        print('Train f1: ', train_f1)
        print('Test f1: ', test_f1)

        # return the accuracy
        return train_f1, test_f1
    
    else:
        # predict values
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # prediction accuracy
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        # print the accuracy
        print('-'*5+'Recall'+'-'*5)
        print('Train recall: ', train_recall)
        print('Test recall: ', test_recall)

        # return the accuracy
        return train_recall, test_recall
