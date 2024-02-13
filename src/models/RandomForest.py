# random forest model

from sklearn.ensemble import RandomForestClassifier

def train_RandomForest(X_train, y_train, n_estimators=100, max_depth=None, \
                       min_samples_split=2, min_samples_leaf=1, random_state=0, criterion='gini'):
    '''Train the random forest model.'''
    
    # create the random forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, \
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1,\
                                    random_state=random_state, criterion=criterion)
    
    # train the model
    rf.fit(X_train, y_train)
    
    # return the model
    return rf