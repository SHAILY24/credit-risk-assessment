from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"{model.__class__.__name__} Classification Report:")
    print(report)
    return y_pred, report

def evaluate_neural_network(model, X_test_scaled, y_test):
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Neural Network Classification Report:")
    print(report)
    return y_pred, report

def display_metrics(metrics):
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)