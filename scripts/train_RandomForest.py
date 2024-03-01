# script for training random forest model

# import libraries
import sys
import os
import pandas as pd
import mlflow
import mlflow.sklearn

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

from src.data.load_data import load_data
from src.data.split_data import split_data
from src.models.RandomForest import train_RandomForest
from src.models.evaluate import evaluate_model
from src.models.log import log_model_metrics
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
    data = engineer_features(load_data(), 'readmitted')

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted', test_size=0.1)

    # pick the best features
    #X_train, X_test = select_features('RandomForest', X_train, X_test, y_train)

    # train the model
    rf = train_RandomForest(X_train, y_train, n_estimators=100, max_depth=20, min_samples_split=7, random_state=121263)

    # plot the roc curve
    plot_roc(rf, X_test, y_test)

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('random_forest_experiment')

    with mlflow.start_run():

        # Log model metrics
        log_model_metrics(rf, X_train, X_test, y_train, y_test, predict_proba=True, threshold=0.3)

        # Log parameters
        mlflow.log_params({"n_estimators": rf.get_params()['n_estimators'], \
                        "max_depth": rf.get_params()['max_depth'], \
                        "min_samples_split": rf.get_params()['min_samples_split'], \
                        "min_samples_leaf": rf.get_params()['min_samples_leaf'], \
                        "criterion": rf.get_params()['criterion'], \
                        "number of features": X_train.shape[1]})

        
if __name__ == "__main__":
    main()

