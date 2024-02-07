# script for training logistic regression model

import sys
import os
import pandas as pd

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

# import libraries
from src.data.load_data import load_data
from src.data.data_scaling import standardize_data, normalize_data
from src.features.feature_labeling import label_encode, one_hot_encode
from src.data.split_data import split_data
from src.models.Logistic import train_Logistic
from src.models.evaluate import evaluate_model

# import data
data = load_data()

# label encode the categorical features
data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 1})

# one hot encode the categorical features
data = one_hot_encode(data)

# split the data
X_train, X_test, y_train, y_test = split_data(data, 'readmitted')

# standardize the data
X_train = standardize_data(X_train)
X_test = standardize_data(X_test)

# train the model
logreg = train_Logistic(X_train, y_train)

# evaluate the model
evaluate_model(logreg, X_train, X_test, y_train, y_test)