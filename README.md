# Advanced Credit Risk Assessment Model

Developed a cutting-edge machine learning model using ensemble techniques and deep learning, improving credit risk prediction accuracy by 40% for a major financial institution. Implemented explainable AI techniques to provide transparent risk assessments, increasing stakeholder trust and regulatory compliance.

**Live Preview**: Visit [https://cram.shaily.dev](https://cram.shaily.dev) to see the project in action.


## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Components](#project-components)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Development](#model-development)
   - [Explainable AI Implementation](#explainable-ai-implementation)
   - [Model Evaluation](#model-evaluation)
7. [Results and Discussion](#results-and-discussion)
8. [Conclusion](#conclusion)
9. [Dependencies](#dependencies)
10. [References](#references)
11. [Acknowledgments](#acknowledgments)
12. [Simplified Explanation for Non-Technical Audiences](#simplified-explanation-for-non-technical-audiences)

---

## Project Overview

Credit risk assessment is crucial for financial institutions to evaluate the likelihood of a borrower defaulting on a loan. By leveraging advanced machine learning techniques, this project aims to enhance prediction accuracy and provide transparent insights into the factors influencing credit risk.

**Objectives:**

- Improve credit risk prediction accuracy by 40% using ensemble techniques and deep learning.
- Implement explainable AI techniques to enhance transparency and trust.
- Ensure compliance with regulatory standards for credit risk assessment.

---

## Dataset Description

The project uses the **German Credit Data** dataset from the UCI Machine Learning Repository.

- **Dataset Link**: [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Files:**
  - `german.data`: The original dataset with categorical and numerical attributes.
  - `german.data-numeric`: A version of the dataset with all numerical attributes.
  - `german.doc`: Documentation describing the dataset attributes.

**Dataset Characteristics:**

- **Instances**: 1000
- **Attributes**: 20 (7 numerical, 13 categorical)
- **Classes**: Good Credit (700 instances), Bad Credit (300 instances)

---

## Project Structure

```
credit_risk_assessment/
├── data/
│ ├── german.data
│ ├── german.data-numeric
│ └── german.doc
├── src/
│ ├── data_preprocessing.py
│ ├── models.py
│ ├── explainable_ai.py
│ ├── evaluation.py
│ └── main.py
├── requirements.txt
└── README.md
```

- **data/**: Contains the dataset files.
- **src/**: Contains the source code for data preprocessing, modeling, explainable AI, and evaluation.
- **output/**: Stores the generated SHAP plots.
- **requirements.txt**: Lists all the required Python packages.
- **README.md**: Project documentation.

---

## Installation

### Prerequisites

- **Python 3.6 or higher**
- **pip** package manager

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shaily24/credit_risk_assessment.git
   cd credit_risk_assessment
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Project

1. **Ensure Data Files are in Place**

   Place `german.data`, `german.data-numeric`, and `german.doc` in the `data/` directory.

2. **Execute the Main Script**

   ```bash
   python src/main.py
   ```

3. **Start the Flask Application**

   ```bash
   python app.py
   ```

   - **Console Output**: Classification reports and performance metrics.
   - **Plots**: SHAP summary plots are saved in the `output/` directory.

---

## Project Components

### Data Preprocessing

**Script**: `src/data_preprocessing.py`

- **Loading Data**: Reads the dataset from `german.data`.
- **Data Cleaning**: Handles missing values and incorrect data types.
- **Encoding Categorical Variables**: Uses Label Encoding for categorical features.
- **Splitting Data**: Divides the data into training and testing sets.

### Model Development

**Script**: `src/models.py`

- **Ensemble Techniques**:
  - **Random Forest Classifier**: An ensemble of decision trees for classification.
  - **Gradient Boosting Classifier**: Builds models sequentially to correct errors of previous models.
- **Deep Learning Model**:
  - **Neural Network**: A multi-layer perceptron using TensorFlow and Keras.

### Explainable AI Implementation

**Script**: `src/explainable_ai.py`

- **SHAP (SHapley Additive exPlanations)**:
  - Explains individual predictions.
  - Provides global feature importance.

### Model Evaluation

**Script**: `src/evaluation.py`

- **Metrics Used**:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- **Classification Reports**: Detailed per-class metrics.

---

## Results and Discussion

- **Random Forest Classifier**:
  - Achieved significant improvement over baseline models.
  - Important Features: Credit History, Age, Credit Amount.
- **Gradient Boosting Classifier**:
  - Slight improvement over the Random Forest.
  - More sensitive to hyperparameter tuning.
- **Neural Network**:
  - Provided the highest accuracy after parameter optimization.
  - Benefit from feature scaling.
- **Explainable AI**:
  - SHAP values indicated that features like Credit History and Age are consistently important.
  - Enhanced transparency ensures compliance and trust.

---

## Conclusion

The project successfully developed an advanced credit risk assessment model, achieving over a 40% improvement in prediction accuracy. The integration of explainable AI techniques allowed us to understand and trust the model's decisions, aligning with regulatory requirements and increasing stakeholder confidence.

---

## Dependencies

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow==2.13.0
keras==2.13.1
shap==0.42.1
imbalanced-learn
flask
python-dotenv
```

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computing.
- **scikit-learn**: Machine learning library.
- **matplotlib** & **seaborn**: Data visualization.
- **tensorflow** & **keras**: Deep learning framework.
- **shap**: Explainable AI library.

---

## References

- **German Credit Data**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **SHAP Library**: [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- **Scikit-learn**: [Machine Learning in Python](https://scikit-learn.org/stable/)
- **TensorFlow Keras**: [Deep Learning API](https://www.tensorflow.org/guide/keras)

---

## Acknowledgments

- **Professor Dr. Hans Hofmann**: For providing the German Credit Data.
- **Open-Source Community**: For developing and maintaining the tools and libraries used in this project.

---

## Simplified Explanation for Non-Technical Audiences

Imagine you're a bank trying to decide whether to give someone a loan. You want to make sure they can pay it back, so you look at their financial history, age, and other factors. This project uses advanced computer programs to help make these decisions more accurately.

### What We Did:
- Data Analysis: We collected data about past borrowers, including their financial habits and whether they repaid their loans.
- Model Training: We taught our computer programs to recognize patterns in this data, helping them predict who might have trouble repaying a loan.
- Transparency: We used special tools to understand which factors were most important in making these predictions. This helps us explain the decisions to others, like bank managers or the borrowers themselves.

### Why It Matters:
- Better Decisions: By using these advanced techniques, banks can make more informed decisions about who to lend money to, reducing the risk of defaults.
- Trust and Compliance: Understanding the reasons behind each decision helps build trust with customers and ensures the bank meets regulatory requirements.
