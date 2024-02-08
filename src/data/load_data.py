# load processed data

# import libraries
import pandas as pd

# append the path
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../../')
sys.path.append(root_dir)

from src.features.feature_labeling import label_encode, one_hot_encode

# function to load the data
def load_data(processed=True, weight=False):
    '''Returns the data.
    kind: bool: type of data to load (raw or processed)
    weight: bool: whether to include weight in the data
    '''

    if processed==False:
        # load the raw data
        data = pd.read_csv(root_dir+'/data/raw/diabetes.csv')
        return data
    
    elif processed:
        # load the processed data
        if weight:
            data = pd.read_csv(root_dir+'/data/processed/diabetes_with_weight_cleaned.csv')
            return data
        else:
            data = pd.read_csv(root_dir+'/data/processed/diabetes_without_weight_cleaned.csv')
            return data
        
def label_data(data):
    """Load and preprocess the dataset."""
    data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 1})
    data = one_hot_encode(data)
    return data