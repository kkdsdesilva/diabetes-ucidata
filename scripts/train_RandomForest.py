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
from src.features.feature_importance import feature_importance_other, pick_top_k_features
from src.features.feature_labeling import label_and_one_hot_encode
from src.models.log import log_model_metrics


def pick_best_k_features(X_train, X_test, y_train, y_test, k):
    """Pick the best number of features for the model."""
    # train the model
    model = train_RandomForest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=4)

    # get the feature importance
    importance = feature_importance_other(model, X_train)

    # pick the top k features
    X_train_k = pick_top_k_features(X_train, importance, k)

    # pick the top k features for the test set
    X_test_k = X_test[X_train_k.columns]

    return X_train_k, X_test_k

def main():
    # Load and preprocess data
    data = label_and_one_hot_encode(load_data())

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, 'readmitted', test_size=0.2)

    # pick the best k features
    X_train, X_test = pick_best_k_features(X_train, X_test, y_train, y_test, k=50)

    # Train the model
    rf = train_RandomForest(X_train, y_train, n_estimators=100, max_depth=40, min_samples_split=4)

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("file://" + os.path.join(cur_dir, '..', 'experiments', 'mlruns'))

    # Set the experiment
    mlflow.set_experiment('random_forest_experiment')

    with mlflow.start_run():
        # Log model
        #mlflow.sklearn.log_model(rf, "random_forest_model")

        # Log model metrics
        log_model_metrics(rf, X_train, X_test, y_train, y_test)

        # Log parameters
        mlflow.log_params({"n_estimators": rf.get_params()['n_estimators'], \
                        "max_depth": rf.get_params()['max_depth'], \
                        "min_samples_split": rf.get_params()['min_samples_split'], \
                        "min_samples_leaf": rf.get_params()['min_samples_leaf'], \
                        "criterion": rf.get_params()['criterion'], \
                        "number of features": X_train.shape[1]})
        
if __name__ == "__main__":
    main()
