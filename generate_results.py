import pickle
import os

from src.data_preprocessing import load_data, preprocess_data
from src.models import (
    train_random_forest, train_gradient_boosting, train_neural_network
)
from src.evaluation import (
    evaluate_model, evaluate_neural_network
)
from src.explainable_ai import explain_tree_model, explain_deep_model
import numpy as np

def main():
    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    rf_pred, rf_report = evaluate_model(rf_model, X_test, y_test)
    rf_accuracy = rf_model.score(X_test, y_test)

    # Train Gradient Boosting model
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_pred, gb_report = evaluate_model(gb_model, X_test, y_test)
    gb_accuracy = gb_model.score(X_test, y_test)

    # Train Neural Network model
    nn_model, scaler = train_neural_network(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    nn_pred, nn_report = evaluate_neural_network(nn_model, X_test_scaled, y_test)
    nn_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]

    # Compile metrics
    metrics = {
        'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
        'Accuracy': [rf_accuracy, gb_accuracy, nn_accuracy],
        'Precision': [
            rf_report['1']['precision'],
            gb_report['1']['precision'],
            nn_report['1']['precision']
        ],
        'Recall': [
            rf_report['1']['recall'],
            gb_report['1']['recall'],
            nn_report['1']['recall']
        ],
        'F1 Score': [
            rf_report['1']['f1-score'],
            gb_report['1']['f1-score'],
            nn_report['1']['f1-score']
        ]
    }

    # Generate SHAP plots and save them to the static directory
    output_dir = os.path.join('static', 'output')
    os.makedirs(output_dir, exist_ok=True)
    explain_tree_model(rf_model, X_train, output_dir, filename='rf_shap.png')
    explain_tree_model(gb_model, X_train, output_dir, filename='gb_shap.png')
    explain_deep_model(nn_model, scaler.transform(X_train), X_test_scaled, output_dir, filename='nn_shap.png')

    # Save the results
    results = {
        'rf_report': rf_report,
        'gb_report': gb_report,
        'nn_report': nn_report,
        'metrics': metrics
    }

    with open('saved_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()