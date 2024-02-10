# script for training logistic regression model

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
from src.data.data_scaling import standardize_data, normalize_data
from src.features.feature_labeling import label_and_one_hot_encode
from src.features.feature_importance import feature_importance_logreg, pick_top_k_features
from src.data.split_data import split_data
from src.models.Logistic import train_Logistic
from src.models.evaluate import evaluate_model
from src.models.log import log_model_metrics

def pick_best_k_features(X_train, X_test, y_train, y_test, k):
    """Pick the best number of features for the model."""
    # train the model
    model = train_Logistic(X_train, y_train, max_iter=2000)

    # get the feature importance
    importance = feature_importance_logreg(model, X_train)

    # pick the top k features
    X_train_k = pick_top_k_features(X_train, importance, k)

    # pick the top k features for the test set
    X_test_k = X_test[X_train_k.columns]

    return X_train_k, X_test_k


def main():
    # Load and preprocess data
    data = label_and_one_hot_encode(load_data())

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted')

    # Standardize the data
    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    # select the best k features
    X_train, X_test = pick_best_k_features(X_train, X_test, y_train, y_test, k=1000)

    # Train the model
    logreg = train_Logistic(X_train, y_train, max_iter=1500, C=1, pen='l2', solver= 'lbfgs')

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('logistic_regression_experiment')

    with mlflow.start_run():
        # log model
        mlflow.sklearn.log_model(logreg, "logistic_regression_model")

        #log parameters
        mlflow.log_params({"penalty": logreg.get_params()['penalty'], "C": logreg.get_params()['C']})

        # log model metrics
        log_model_metrics(logreg, X_train, X_test, y_train, y_test)

    
if __name__ == '__main__':
    main()