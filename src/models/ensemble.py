# ensemble model using the neural network model and the random forest model

# import libraries

from .NN import train_NN
from .RandomForest import train_RandomForest

def train_ensemble(X_train, y_train, input_dim, epochs=10, batch_size=32, validation_split=0.1, learning_rate=0.01, \
                     n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, criterion='gini'):
    '''Train the ensemble model.'''

    # train the neural network model   
    nn, history = train_NN(X_train, y_train, input_dim, epochs, batch_size, validation_split, learning_rate)
    
    # train the random forest model
    rf = train_RandomForest(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, criterion)

    # return the models
    nn, rf