# neural network

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Recall



def train_NN(X_train, y_train, input_dim, epochs=10, batch_size=32, validation_split=0.1, learning_rate=0.01):
    '''Train the neural network model.'''
    
    # create the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # optimizer
    opt = Adam(learning_rate = learning_rate)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Recall(), AUC()])

    # train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # return the model
    return model, history