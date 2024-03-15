# neural network

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

    

def train_NN(layers_config, X_train, y_train, input_dim, n_ini=64, epochs=10, batch_size=32, validation_split=0.1, lr=['constant',0.01]):
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


    # Set the learning rate scheduler
    if lr[0] == 'varying':

      lr_schedule = lr[1]  
      #lr_schedule = tf.keras.callbacks.LearningRateScheduler(
      #  lambda epoch: 1e-8 * 10**(epoch / 20))
      
      # Instantiate the Adam optimizer
      adam_optimizer = Adam()

      model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy', Recall(name='recall')])

      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0,
                          callbacks=[lr_schedule, TqdmCallback(verbose=1)])

    else:
       
    # train the model
      model.compile(optimizer=Adam(learning_rate=lr[1]), loss='binary_crossentropy', metrics=['accuracy', Recall(name='recall')])

      history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0,
                          callbacks=[TqdmCallback(verbose=1)])
    
    #early_stopping = EarlyStopping(monitor='val_recall', mode='max', patience=5, restore_best_weights=True)
   # history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0,
   #                     callbacks=[TqdmCallback(verbose=1), early_stopping])

    # return the model
    return model, history
