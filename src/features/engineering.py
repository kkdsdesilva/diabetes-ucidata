# feature engineering functions

# label categorical features and encode them

# import libraries
import pandas as pd

# label_encode: Returns the column with categorical features labeled and encoded.
def label_encode(df, col, dic):
    '''Returns the column with categorical features labeled and encoded.'''
    
    # copy of the dataframe
    data = df.copy()

    data[col] = data[col].map(dic)

    # return the column
    return data


# one hot encode the categorical features
def one_hot_encode(data, cat_cols=None):
    '''Returns the data with categorical features one hot encoded.'''
    
    # one hot encode the categorical features
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    # return the data
    return data


# label and one hot encode the data
def label_and_one_hot_encode(data, readmit_days=False):
    '''Returns the data with categorical features labeled and one hot encoded.
    data: dataframe: the data
    readmit_days: bool: whether to label encode the readmitted column with 3 classes'''

    # readmit_days: whether to label encode the readmitted column with 3 classes
    if readmit_days:
        # label the categorical features
        data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 2})

    else:
        # label the categorical features
        data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 1})
    
    # one hot encode the categorical features
    data = one_hot_encode(data)

    # return the data
    return data


def change_diag_columns(data):
    '''Returns the data with the diag_1, diag_2 or diag_3 columns changed if either one exist.'''

    for x in ['diag_1', 'diag_2', 'diag_3']:
        if x in data.columns:
            data[x] = data[x].map(lambda x: '00'+x[0:3]).map(lambda x: x[-3:][0:2])

    # return the data
    return data

def change_medication_columns(data):
    col = ['metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
    
    for x in col:
        if x in data.columns:
            data[x] = data[x].map(lambda x: 0 if x=='No' else 1)

    # return the data
    return data



def engineer_features(data, readmit_days=False):
    '''Returns the data with engineered features.'''

    # change the diag columns
    data = change_diag_columns(data)

    # change the medication columns
    data = change_medication_columns(data)
    
    # change the diag columns
    data = label_and_one_hot_encode(data, readmit_days=readmit_days)
    
    # return the data
    return data