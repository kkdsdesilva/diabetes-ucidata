import os
import sys

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

from src.models.ensemble import train_ensemble
from src.models.NN import train_NN
from src.models.RandomForest import train_RandomForest
from src.data.load_data import load_data
from src.data.split_data import split_data
from src.data.data_scaling import standardize_data, normalize_data
from src.features.engineering import engineer_features
from sklearn.model_selection import train_test_split
import numpy as np
import random
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# select columns to use
cols = ['race', 'gender', 'age', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
    'payer_code', 'medical_specialty', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
    'number_diagnoses', 'readmitted']

# Load and preprocess data
data_nn = engineer_features(load_data(), 'readmitted', one_hot=True)
data_rf = engineer_features(load_data(), 'readmitted', one_hot=False)

# Split the data
train_nn, test_nn = train_test_split(data_nn, test_size=0.1, random_state=3, stratify=data_nn['readmitted'])
train_rf, test_rf = train_test_split(data_rf, test_size=0.1, random_state=3, stratify=data_rf['readmitted'])

print(train_nn.shape, test_nn.shape, train_rf.shape, test_rf.shape)

X_train_nn, y_train_nn = train_nn.drop(columns='readmitted'), train_nn['readmitted']
X_test_nn, y_test_nn = test_nn.drop(columns='readmitted'), test_nn['readmitted']

X_train_rf, y_train_rf = train_rf.drop(columns='readmitted'), train_rf['readmitted']
X_test_rf, y_test_rf = test_rf.drop(columns='readmitted'), test_rf['readmitted']

# Standardize the data
X_train_nn = standardize_data(X_train_nn)
X_test_nn = standardize_data(X_test_nn)


# train random forest model
rf = train_RandomForest(X_train_rf, y_train_rf, n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=6, criterion='gini')

# train neural network model
nn, history = train_NN(X_train_nn, y_train_nn, input_dim=X_train_nn.shape[1], epochs=10, batch_size=32, validation_split=0, learning_rate=0.01)

# predict using the random forest model
y_pred_rf = rf.predict(X_test_rf).reshape(-1)

# predict using the neural network model
y_pred_nn = (nn.predict(X_test_nn) > 0.5).astype('int32').reshape(-1)

# use both models to predict. If at least one model predict 1, then the ensemble model predicts 1, else 0
y_pred_ensemble = (y_pred_rf + y_pred_nn) > 0


# print the accuracy of the ensemble model
print("Accuracy of the ensemble model: ", sum(y_pred_ensemble == y_test_rf) / len(y_test_rf))
# print recall
print("Recall of the ensemble model: ", sum((y_pred_ensemble == 1) & (y_test_rf == 1)) / sum(y_test_rf == 1))