# script for training random forest model

# import libraries
import sys
import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

from src.data.load_data import load_data
from src.models.NN import create_nn, train_nn 
from src.data.split_data import split_data
from src.features.selection import select_features
from src.data.data_scaling import standardize_data, normalize_data
from src.features.engineering import engineer_features
from src.visualization.NN_visualization import plot_metrics

def main():
    import warnings
    warnings.filterwarnings('ignore')

    # select columns to use
    cols = ['race', 'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'payer_code', 'medical_specialty', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
       'number_diagnoses', 'A1Cresult', 'readmitted']

    # Load and preprocess data
    data = engineer_features(load_data()[cols], 'readmitted', one_hot=True)

    # Split the data
    X_train, X_val, y_train, y_val = split_data(data, 'readmitted', test_size=0.1, random_state=65)

    # Standardize the data
    #X_train = standardize_data(X_train)
    #X_test = standardize_data(X_test)

    # create the neural network model
    nn = create_nn(input_dim=X_train.shape[1], learning_rate=0.01, hidden_layers=5, config=[64, 32, 32, 32, 8], \
                    activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'], \
                        l2_reg=[False, False, False, False, False], l2_lambda=[0.1, 0.1, 0.1, 0.1])
    
    # train the model
    nn, history = train_nn(nn, X_train, y_train, epochs=30, batch_size=256, validation_data=(X_val, y_val))

    # print recall and auc
    print("Accuracy: ", history.history['val_accuracy'][-1])
    print("Recall: ", history.history['val_recall'][-1])
    print("AUC: ", history.history['val_auc'][-1])

    # plot accuracy, recall and auc and save
    plt.plot(history.history['val_accuracy'], label='accuracy')
    plt.plot(history.history['val_recall'], label='recall')
    plt.plot(history.history['val_auc'], label='auc')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    plt.savefig('reports/figures/NN_metrics_epochs.png')

# run the main function
if __name__ == '__main__':
    main()  