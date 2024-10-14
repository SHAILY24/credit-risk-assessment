import shap
import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_output_directory():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def explain_tree_model(model, X_train, X_test, model_name='Tree Model', filename='tree_model_shap.png'):
    output_dir = ensure_output_directory()
    filepath = os.path.join(output_dir, filename)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values, 
        X_test, 
        plot_type='bar', 
        show=False, 
        plot_size=(10, 6)
    )
    plt.title(f'SHAP Summary Plot for {model_name}')
    plt.xlabel('Mean SHAP Value (Impact on Model Output)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def explain_deep_model(model, X_train_scaled, X_test_scaled, filename='deep_model_shap.png'):
    output_dir = ensure_output_directory()
    filepath = os.path.join(output_dir, filename)

    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
    test_samples = X_test_scaled[:100]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    shap.summary_plot(
        shap_values, 
        test_samples, 
        show=False, 
        plot_size=(10, 6)
    )
    plt.title('SHAP Summary Plot for Neural Network')
    plt.xlabel('Mean SHAP Value (Impact on Model Output)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()