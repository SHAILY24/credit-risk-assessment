import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data():
    # Load the dataset
    df = pd.read_csv('data/german.data', sep=' ', header=None)
    
    # Assign column names based on the dataset's documentation
    df.columns = [
        'Status_of_existing_checking_account', 'Duration_in_month',
        'Credit_history', 'Purpose', 'Credit_amount',
        'Savings_account/bonds', 'Present_employment_since',
        'Installment_rate_in_percentage_of_disposable_income',
        'Personal_status_and_sex', 'Other_debtors/guarantors',
        'Present_residence_since', 'Property', 'Age_in_years',
        'Other_installment_plans', 'Housing',
        'Number_of_existing_credits_at_this_bank', 'Job',
        'Number_of_people_being_liable_to_provide_maintenance_for',
        'Telephone', 'Foreign_worker', 'Target'
    ]
    
    return df

def preprocess_data(df):
    # Map target variable to binary (1: Good, 2: Bad)
    df['Target'] = df['Target'].map({1: 0, 2: 1})
    
    # Identify categorical columns
    categorical_cols = [
        'Status_of_existing_checking_account', 'Credit_history', 'Purpose',
        'Savings_account/bonds', 'Present_employment_since',
        'Personal_status_and_sex', 'Other_debtors/guarantors', 'Property',
        'Other_installment_plans', 'Housing', 'Job', 'Telephone',
        'Foreign_worker'
    ]
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Split features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test