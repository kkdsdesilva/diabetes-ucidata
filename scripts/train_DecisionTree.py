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

from src.data.load_data import load_data, preprocess_data
from src.data.split_data import split_data
from src.data.data_scaling import standardize_data
from src.models.DecisionTree import train_DecisionTree
from src.models.evaluate import evaluate_model

def log_model_metrics(dtree, X_train, X_test, y_train, y_test):
    """Log model metrics to MLflow."""
    train_acc, test_acc = evaluate_model(dtree, X_train, X_test, y_train, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

def main():
    # Load and preprocess data
    data = preprocess_data(load_data())

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted')

    # Train the model
    dtree = train_DecisionTree(X_train, y_train, criterion='gini', max_depth=10, min_samples_split=2)

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'decision_tree', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('decision_tree_experiment')

    with mlflow.start_run():
        mlflow.sklearn.log_model(dtree, "decision_tree_model")
        mlflow.log_params({"criterion": dtree.get_params()['criterion'], "max_depth": dtree.get_params()['max_depth'], "min_samples_split": dtree.get_params()['min_samples_split']})
        log_model_metrics(dtree, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
