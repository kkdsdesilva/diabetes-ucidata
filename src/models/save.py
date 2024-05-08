# save the models

import joblib
import os

# function to save the model
def save_model(model, path):
    ''' Save the model
    Args:
        model (sklearn estimator): Trained model to be saved.
        path (str): Path where the model will be saved.
        '''
    
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)