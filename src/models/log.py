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
    # evaluate the model
    train_acc, test_acc = evaluate_model(model, X_train, X_test, y_train, y_test)
    train_recall, test_recall = evaluate_model(model, X_train, X_test, y_train, y_test, metric='recall')
    train_f1, test_f1 = evaluate_model(model, X_train, X_test, y_train, y_test, metric='f1')

    # log the metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_recall", train_recall)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)
    