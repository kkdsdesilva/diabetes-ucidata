# log model metrics to MLflow

import mlflow
import sys
import os

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
sys.path.append(root_dir)

from src.models.evaluate import evaluate_model

def log_model_metrics(model, X_train, X_test, y_train, y_test):
    """Log model metrics to MLflow."""
    train_acc, test_acc = evaluate_model(model, X_train, X_test, y_train, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)