# script for training random forest model

# import libraries
import sys
import os
import pandas as pd
import mlflow
import mlflow.tensorflow

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

from src.data.load_data import load_data
from src.models.NN import train_NN
from src.data.split_data import split_data
from src.features.selection import select_features
from src.features.engineering import engineer_features
from src.visualization.roc import plot_roc

def main():
    import warnings
    warnings.filterwarnings('ignore')

    # select columns to use
    cols = ['race', 'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'payer_code', 'medical_specialty', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
       'number_diagnoses', 'readmitted']

    # Load and preprocess data
    data = engineer_features(load_data(), 'readmitted', one_hot=True)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted', test_size=0.1)

    # train the model
    nn, history = train_NN(X_train, y_train, input_dim=X_train.shape[1], epochs=10, batch_size=32, validation_split=0.1)

    # plot the roc curve
    #plot_roc(nn, X_test, y_test)

    # print recall and auc
    print("Recall: ", history.history['recall'])
    print("AUC: ", history.history['auc'])

# run the main function
if __name__ == '__main__':
    main()  