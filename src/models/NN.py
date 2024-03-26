# neural network

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Recall


# define the nn model
def create_nn(input_dim, learning_rate=0.01, hidden_layers=3, config=[32, 32, 8], \
                    activations=['relu', 'relu', 'relu', 'sigmoid'], \
                        l2_reg=[False, False, False], l2_lambda=[0.01, 0.01, 0.01]):
    '''Create a neural network model.
    input_dim: int, the number of features
    learning_rate: float, the learning rate
    hidden_layers: int, the number of hidden layers
    config: list, the number of neurons in each hidden layer
    activations: list, the activation functions for each layer
    '''
    
    # create the neural network model
    model = Sequential()

    if l2_reg[0]:
        model.add(Dense(config[0], input_dim=input_dim, activation=activations[0], kernel_regularizer=regularizers.l2(l2_lambda[0])))
    else:
        model.add(Dense(config[0], input_dim=input_dim, activation=activations[0]))

    for i in range(1, hidden_layers):
        if l2_reg[i]:
            model.add(Dense(config[i], activation=activations[i], kernel_regularizer=regularizers.l2(l2_lambda[i])))
        else:
            model.add(Dense(config[i], activation=activations[i]))

        
    model.add(Dense(1, activation=activations[-1]))
    
    # optimizer
    opt = Adam(learning_rate = learning_rate)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt, \
                  metrics= ['accuracy', Recall(thresholds=0.45), AUC()])

    # return the model
    return model



# train the nn model
def train_nn(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
    '''Train the neural network model.'''
    
    # train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # return the model
    return model, history