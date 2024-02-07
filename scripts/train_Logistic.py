# script for training logistic regression model

import sys
import os
import pandas as pd

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

# import libraries
from src.data.data_scaling import standardize_data, normalize_data
from src.features.feature_labeling import label_encode, one_hot_encode
from src.data.split_data import split_data
from src.models.Logistic import train_Logistic
from src.models.evaluate_model import evaluate_model

# import data
data = pd.read_csv(root_dir+'/data/processed/diabetes_with_weight_cleaned.csv')

