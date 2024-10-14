from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return y_pred, report

def evaluate_neural_network(model, X_test_scaled, y_test):
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
    y_pred = y_pred.flatten()
    report = classification_report(y_test, y_pred)
    return y_pred, report

def display_metrics(metrics):
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)