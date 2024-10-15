import shap
import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_output_directory():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def explain_tree_model(model, X_train, output_dir, filename='shap_plot.png'):
    explainer = shap.TreeExplainer(model)
    
    # Ensure X_train is a DataFrame or array
    if isinstance(X_train, str):
        raise ValueError("X_train should be a DataFrame or array, not a string.")
    
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    
    # Adjust the layout to prevent cutting off labels
    plt.gcf().set_size_inches(12, 8)  # Increase figure size
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

def explain_deep_model(model, X_train_scaled, X_test_scaled, output_dir, filename='deep_model_shap.png'):
    filepath = os.path.join(output_dir, filename)

    # Select a background sample for SHAP
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
    test_samples = X_test_scaled[:100]

    # Create a SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    # Generate the SHAP summary plot
    shap.summary_plot(
        shap_values, 
        test_samples, 
        show=False, 
        plot_size=(12, 8)  # Increase plot size
    )
    plt.title('SHAP Summary Plot for Neural Network')
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()