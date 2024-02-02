# Description: This script contains functions to preprocess the data.
# add_dtypes: Assigns dtypes to the columns of the dataframe.
def add_dtypes(data):
    '''Returns data with dtypes correctly assigned.'''
    # import libraries
    import pandas as pd

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

# impute_data: Returns data with missing values imputed.
def impute_data(data):
    '''Returns data with missing values imputed.'''

    # import libraries
    from sklearn.impute import SimpleImputer

    # get all categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns

    # get all numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = imputer.fit_transform(data[cat_cols])

    # impute missing values
    imputer = SimpleImputer(strategy='mean')
    data[num_cols] = imputer.fit_transform(data[num_cols])

    # return data
    return data



# main script
if __name__ == '__main__':
    # import data
    import sys
    sys.path.append('../../')
    from src.data.data_collection import import_data
    data = import_data()

    # add dtypes
    data = add_dtypes(data)

    # remove weight column
    data_wo_weight = data_wo_weight(data)

    # impute missing values
    data_imputed = impute_data(data)
    data_imputed_wo_weight = impute_data(data_wo_weight)