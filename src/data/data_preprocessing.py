# Description: This script contains functions to preprocess the data.

# import libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# add_dtypes: Assigns dtypes to the columns of the dataframe.
def add_dtypes(data):
    '''Returns data with dtypes correctly assigned.'''

    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    categorical = data.columns.difference(numeric_cols)

    # assign dtypes to float to numeric columns
    data[numeric_cols] = data[numeric_cols].astype('float')

    # assign dtypes to object to categorical columns
    data[categorical] = data[categorical].astype('object')

    # return data
    return data


# data_wo_weight: Returns data with weight column removed.
def data_wo_weight(data):
    '''Returns data with weight column removed.'''

    # remove weight column
    data = data.drop(columns='weight')

    # return data
    return data


# data_w_weight: Returns values where weight values are present.
def data_w_weight(data):
    '''Returns values where weight values are present.'''
    
    # get rows where weight is present
    data = data[data['weight'].notna()]

    # return data
    return data


# impute_data: Returns data with missing values imputed.
def impute_data(data):
    '''Returns data with missing values imputed.'''

    # get all categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns

    # get all numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # get a copy of the data
    data_copy = data.copy()

    # impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_cat_data = imputer.fit_transform(data[cat_cols])
    data_copy.loc[:, cat_cols] = imputed_cat_data

    # impute missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_num_data = imputer.fit_transform(data[num_cols])
    data_copy.loc[:, num_cols] = imputed_num_data

    # return data
    return data_copy