# feature engineering functions

# label categorical features and encode them

# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# label encode the target variable
def target_encode(df, col, dic):
    '''Returns the column with categorical features labeled and encoded.'''
    
    # copy of the dataframe
    data = df.copy()

    data[col] = data[col].map(dic)

    # return the column
    return data


# label encode the categorical features
def label_encode(data, target, cat_cols=None):
    '''Returns the data with categorical features labeled.'''
    
        # label encode the categorical features
    if cat_cols:
        label_encoder = LabelEncoder()
        for col in cat_cols:
            data[col] = label_encoder.fit_transform(data[col])
    
    else:
        # obtain the categorical columns
        cat_cols = data.select_dtypes(include='object').columns

        # remove the target column if it is in the list
        if target in cat_cols:
            cat_cols = cat_cols.drop(target)

        label_encoder = LabelEncoder()
        for col in cat_cols:
            data[col] = label_encoder.fit_transform(data[col])
    
    # return the data
    return data


# one hot encode the categorical features
def one_hot_encode(data, target, cat_cols=None):
    '''Returns the data with categorical features one hot encoded.'''
    
    # one hot encode the categorical features
    data_encoded = pd.get_dummies(data.drop(columns=[target]), columns=cat_cols, drop_first=True)

    # convert to float64
    data_encoded = data_encoded.astype('float64')

    # add the target column to the dataframe
    data_encoded[target] = data[target]
    
    # return the data
    return data_encoded


# label and encode the data
def data_encode(data, target, readmit_days=False, one_hot=False, cat_cols=None):
    '''Returns the data with the categorical features labeled and encoded.'''

    # readmit_days: whether to label encode the readmitted column with 3 classes
    if readmit_days:
        # label the categorical features
        data = target_encode(data, target, {'NO': 0, '>30': 1, '<30': 2})

    else:
        # label the categorical features
        data = target_encode(data, target, {'NO': 0, '>30': 1, '<30': 1})

    if one_hot:
        # one hot encode the categorical features
        data = one_hot_encode(data, target, cat_cols=cat_cols)
    
    else:
        # label encode the categorical features
        data = label_encode(data, target, cat_cols=cat_cols)

    # return the data
    return data


# change the diag columns
def change_diag_columns(data):
    '''Returns the data with the diag_1, diag_2 or diag_3 columns changed if either one exist.'''

    for x in ['diag_1', 'diag_2', 'diag_3']:
        if x in data.columns:
            data[x] = data[x].map(lambda x: '00'+x[0:3]).map(lambda x: x[-3:][0:2])

    # return the data
    return data


# change the medication columns
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


# engineer the features
def engineer_features(data, target, readmit_days=False, one_hot=False):
    '''Returns the data with engineered features.'''

    # change the diag columns
    data = change_diag_columns(data)

    # change the medication columns
    data = change_medication_columns(data)

    # label and one hot encode the data
    data = data_encode(data, target, readmit_days=readmit_days, one_hot=one_hot)
    
    print('-'*20)
    print('Feature engineering complete.')
    print('-'*20)
    
    # return the data
    return data