# This script contains functions to scale the data.

# standardize_data: Returns data with numerical columns standardized.
def standardize_data(data):
    '''Returns data with numerical columns standardized.'''
    # import libraries
    from sklearn.preprocessing import StandardScaler

    # get all numerical columns
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    # standardize numerical columns
    scaler = StandardScaler()

    # fit and transform data
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # return data
    return data

# normalize_data: Returns data with numerical columns normalized.
def normalize_data(data):
    '''Returns data with numerical columns normalized.'''
    # import libraries
    from sklearn.preprocessing import MinMaxScaler

    # get all numerical columns
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    # normalize numerical columns
    scaler = MinMaxScaler()

    # fit and transform data
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # return data
    return data
