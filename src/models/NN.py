# neural network

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Recall

import tensorflow as tf

    

def train_NN(layers_config, X_train, y_train, input_dim, n_ini=64, epochs=10, batch_size=32, validation_split=0.1, learning_rate=0.01):
    """
    Create a sequential model based on the provided layers configuration.

    Parameters:
    - layers_config: A list of tuples, each tuple representing (units, activation)
      for a layer.

    n_ini: Number of units in the first layer.

    """
    model = tf.keras.Sequential()
    model.add(Dense(n_ini, input_dim=input_dim, activation='relu'))

    for units, activation in layers_config:
        model.add(Dense(units, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))  # Example output layer
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # return the model
    return model, history
