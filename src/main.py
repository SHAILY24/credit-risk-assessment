import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_preprocessing import load_data, preprocess_data
from models import (
    train_random_forest, train_gradient_boosting, train_neural_network
)
from evaluation import (
    evaluate_model, evaluate_neural_network, display_metrics
)
from explainable_ai import explain_tree_model, explain_deep_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import numpy as np


def main():
    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    rf_pred, rf_report = evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Classification Report:\n", rf_report)
    
    # Train Gradient Boosting model
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_pred, gb_report = evaluate_model(gb_model, X_test, y_test)
    print("Gradient Boosting Classification Report:\n", gb_report)
    
    # Train Neural Network model
    nn_model, scaler = train_neural_network(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    nn_pred, nn_report = evaluate_neural_network(nn_model, X_test_scaled, y_test)
    print("Neural Network Classification Report:\n", nn_report)
    
    # Compile metrics
    metrics = {
        'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
        'Accuracy': [
            rf_model.score(X_test, y_test),
            gb_model.score(X_test, y_test),
            nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        ],
        'Precision': [
            precision_score(y_test, rf_pred),
            precision_score(y_test, gb_pred),
            precision_score(y_test, nn_pred)
        ],
        'Recall': [
            recall_score(y_test, rf_pred),
            recall_score(y_test, gb_pred),
            recall_score(y_test, nn_pred)
        ],
        'F1 Score': [
            f1_score(y_test, rf_pred),
            f1_score(y_test, gb_pred),
            f1_score(y_test, nn_pred)
        ]
    }
    display_metrics(metrics)
    output_dir = os.path.join('output')
    # Explain models and save plots
    explain_tree_model(rf_model, X_train, output_dir, filename='rf_shap.png')
    explain_tree_model(gb_model, X_train, output_dir, filename='gb_shap.png')
    explain_deep_model(nn_model, scaler.transform(X_train), X_test_scaled, output_dir, filename='nn_shap.png')
    results = {
        'rf_report': rf_report,
        'gb_report': gb_report,
        'nn_report': nn_report,
        'metrics': metrics
    }
    return results

if __name__ == '__main__':
    main()