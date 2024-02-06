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