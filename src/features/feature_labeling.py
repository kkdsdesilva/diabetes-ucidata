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