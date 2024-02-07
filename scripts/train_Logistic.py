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
from src.features.feature_labeling import label_encode, one_hot_encode
from src.data.split_data import split_data
from src.models.Logistic import train_Logistic
from src.models.evaluate import evaluate_model

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    data = load_data()
    data = label_encode(data, 'readmitted', {'NO': 0, '>30': 1, '<30': 1})
    data = one_hot_encode(data)
    return data

def log_model_metrics(logreg, X_train, X_test, y_train, y_test):
    """Log model metrics to MLflow."""
    train_acc, test_acc = evaluate_model(logreg, X_train, X_test, y_train, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted')

    # Standardize the data
    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    # Train the model
    logreg = train_Logistic(X_train, y_train, max_iter=1500, C=1)

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'logistic_regression', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('logistic_regression_experiment')

    with mlflow.start_run():
        mlflow.sklearn.log_model(logreg, "logistic_regression_model")
        mlflow.log_params({"penalty": logreg.get_params()['penalty'], "C": logreg.get_params()['C']})
        log_model_metrics(logreg, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()