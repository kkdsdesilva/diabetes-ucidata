# script for training decision tree model

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
from src.features.engineering import engineer_features
from src.features.feature_importance import feature_importance_other, pick_top_k_features
from src.data.split_data import split_data
from src.models.DecisionTree import train_DecisionTree
from src.models.evaluate import evaluate_model
from src.models.log import log_model_metrics

def pick_best_k_features(X_train, X_test, y_train, y_test, k=100):
    """Pick the best number of features for the model."""
    # train the model
    model = train_DecisionTree(X_train, y_train, criterion='gini', max_depth=15, min_samples_split=4)

    # get the feature importance
    importance = feature_importance_other(model, X_train)

    # pick the top k features
    X_train_k = pick_top_k_features(X_train, importance, k)

    # pick the top k features for the test set
    X_test_k = X_test[X_train_k.columns]

    # return the top k features
    return X_train_k, X_test_k

def main():
    import warnings
    warnings.filterwarnings('ignore')

    # select columns to use
    cols = ['race',  'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 
       'medical_specialty', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
       'number_diagnoses', 'change', 'diabetesMed', 'readmitted' 
            ]

    # Load and preprocess data
    data = engineer_features(load_data()[['admission_source_id', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'number_diagnoses', 'readmitted']])

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted')

    # pick the best k features
    #X_train, X_test = pick_best_k_features(X_train, X_test, y_train, y_test, k=2)

    # Train the model
    dtree = train_DecisionTree(X_train, y_train, criterion='gini', max_depth=15, min_samples_split=7)

    # Log model metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('decision_tree_experiment')

    with mlflow.start_run():
        mlflow.log_params({"criterion": dtree.get_params()['criterion'], \
                        "max_depth": dtree.get_params()['max_depth'], \
                        "min_samples_split": dtree.get_params()['min_samples_split'], \
                        "number of features": X_train.shape[1]})
        
        log_model_metrics(dtree, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()