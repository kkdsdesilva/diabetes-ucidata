# decision tree model

# import libraries
from sklearn.tree import DecisionTreeClassifier

def train_DecisionTree(X_train, y_train, criterion='gini', max_depth=None, min_samples_split=2):
    '''Returns the decision tree model and print the predictions.'''
    
    # create the decision tree
    dtree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
    
    # fit the model
    dtree.fit(X_train, y_train)
    
    # return the model
    return dtree
