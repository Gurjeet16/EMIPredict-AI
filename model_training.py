import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
DATA_PATH = 'emi_prediction_dataset.csv'
MODEL_SAVE_PATH = 'emi_models.pkl'

# Define Feature Groups
NUMERIC_FEATURES = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure'
]

CATEGORICAL_FEATURES = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario'
]


def load_and_clean_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    df = df.drop_duplicates()

    # Force numeric conversion
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print(f"Data loaded. Shape: {df.shape}")
    return df


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])


def train_models():
    # 1. Load Data
    df = load_and_clean_data(DATA_PATH)

    # Clean Targets
    df['max_monthly_emi'] = pd.to_numeric(
        df['max_monthly_emi'], errors='coerce').fillna(0)

    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y_class = df['emi_eligibility']
    y_reg = df['max_monthly_emi']

    # 3. Split Data (Using smaller subset for training to save size if needed, but max_depth handles it)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    # ==========================================
    # OPTIMIZED MODELS (Reduced Size)
    # ==========================================
    print("\n--- Training Optimized Classification Model ---")
    # Reduced n_estimators to 50 and max_depth to 10 to keep file size small
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    ])
    clf_pipeline.fit(X_train, y_class_train)

    print("\n--- Training Optimized Regression Model ---")
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    ])
    reg_pipeline.fit(X_train, y_reg_train)

    # Evaluation
    print("Evaluating...")
    y_class_pred = clf_pipeline.predict(X_test)
    print("Acc:", accuracy_score(y_class_test, y_class_pred))

    # ==========================================
    # SAVE WITH COMPRESSION
    # ==========================================
    print("\nSaving compressed models...")
    artifacts = {
        'classifier': clf_pipeline,
        'regressor': reg_pipeline,
        'features_num': NUMERIC_FEATURES,
        'features_cat': CATEGORICAL_FEATURES
    }
    # compress=3 drastically reduces file size
    joblib.dump(artifacts, MODEL_SAVE_PATH, compress=3)
    print(f"Successfully saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_models()
