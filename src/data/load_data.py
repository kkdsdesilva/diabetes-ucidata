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


# add_dtypes: Assigns dtypes to the columns of the dataframe.
def add_dtypes(data):
    '''Returns data with dtypes correctly assigned.

    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    categorical = data.columns.difference(numeric_cols)

    # assign dtypes to float to numeric columns
    data[numeric_cols] = data[numeric_cols].astype('float')

    # assign dtypes to object to categorical columns
    data[categorical] = data[categorical].astype('object')'''

    # return data
    return data


# function to load the data
def load_data(processed=True, weight=False, columns=None):

    '''Returns the data.
    kind: bool: type of data to load (raw or processed)
    weight: bool: whether to include weight in the data
    '''

    if processed==False:
        # load the raw data
        data = pd.read_csv(root_dir+'/data/raw/diabetes.csv', usecols=columns)
        return add_dtypes(data)
    
    elif processed:
        # load the processed data
        if weight:
            data = pd.read_csv(root_dir+'/data/processed/diabetes_with_weight_cleaned.csv', usecols=columns)
            return add_dtypes(data)
        else:
            data = pd.read_csv(root_dir+'/data/processed/diabetes_without_weight_cleaned.csv', usecols=columns)
            return add_dtypes(data)


# label_data: Load and preprocess the dataset.
def label_data(data):
    """Load and preprocess the dataset."""
    
    # label encode the target variable
    data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 1})
    data = one_hot_encode(data)

    return data