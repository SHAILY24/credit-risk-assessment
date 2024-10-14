import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_tree_model(model, X_train, X_test):
    # Initialize the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type='bar')
    plt.show()

def explain_deep_model(model, X_train_scaled, X_test_scaled):
    # Create a background dataset
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 50, replace=False)]
    
    # Initialize the GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    print('Shape of shap_values:', shap_values.shape)
    print('Shape of X_test_scaled:', X_test_scaled[:100].shape)    
    # Compute SHAP values for a subset of test data
    shap_values = explainer.shap_values(X_test_scaled[:100])
    
    # Convert shap_values to the correct format if necessary
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Summary plot
    shap.summary_plot(shap_values, X_test_scaled[:100])
    plt.show()