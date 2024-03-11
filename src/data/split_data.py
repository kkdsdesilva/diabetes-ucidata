# split the data into train and test
# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, target, test_size=0.3, random_state=0):
    '''Returns the train and test data.'''
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], \
                                                        test_size=test_size, random_state=random_state, stratify=data[target])
    
    # return the data
    return X_train, X_test, y_train, y_test