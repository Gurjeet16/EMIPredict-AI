import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# Load the trained models


@st.cache_resource
def load_models():
    try:
        return joblib.load('emi_models.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please run 'model_training.py' first.")
        return None


artifacts = load_models()

# ==========================================
# 2. USER INTERFACE DESIGN
# ==========================================
st.title("💰 EMIPredict AI - Financial Risk Assessment")
st.markdown("""
This intelligent platform predicts your **Loan Eligibility** and calculates the **Maximum EMI** you can afford based on your financial profile.
""")

if artifacts:
    # Organize inputs into tabs for better UX
    tab1, tab2, tab3 = st.tabs(
        ["Personal Details", "Financials", "Loan Request"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            marital = st.selectbox(
                "Marital Status", ['Single', 'Married', 'Divorced'])
        with col2:
            education = st.selectbox(
                "Education", ['High School', 'Graduate', 'Post Graduate', 'Professional'])
            family_size = st.number_input("Family Size", 1, 10, 3)
            dependents = st.number_input("Dependents", 0, 10, 1)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            salary = st.number_input(
                "Monthly Salary (₹)", min_value=0.0, value=50000.0)
            emp_type = st.selectbox(
                "Employment Type", ['Salaried', 'Self-Employed', 'Government', 'Private'])
            comp_type = st.selectbox(
                "Company Type", ['MNC', 'Start-up', 'Government', 'Private', 'Mid-size'])
            years_emp = st.number_input("Years of Employment", 0.0, 50.0, 5.0)
            credit_score = st.slider("Credit Score", 300, 900, 750)
        with col2:
            rent = st.number_input("Monthly Rent", 0.0, value=10000.0)
            expenses = st.number_input(
                "Other Monthly Expenses", 0.0, value=15000.0)
            existing_emi = st.number_input(
                "Current EMI Obligations", 0.0, value=0.0)
            emergency_fund = st.number_input(
                "Emergency Fund", 0.0, value=100000.0)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            loan_amount = st.number_input(
                "Requested Loan Amount", min_value=10000.0, value=500000.0)
            tenure = st.number_input("Requested Tenure (Months)", 6, 360, 24)
        with col2:
            scenario = st.selectbox(
                "Loan Purpose", ['Personal Loan', 'Home Loan', 'Car Loan', 'Education Loan'])
            existing_loans = st.radio(
                "Do you have existing loans?", ['Yes', 'No'])

    # ==========================================
    # 3. PREDICTION LOGIC
    # ==========================================
    if st.button("Analyze Risk & Check Eligibility", type="primary"):
        # Prepare input dataframe matching training columns
        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital,
            'education': education,
            'monthly_salary': salary,
            'employment_type': emp_type,
            'years_of_employment': years_emp,
            'company_type': comp_type,
            'house_type': 'Rented' if rent > 0 else 'Owned',  # Inferring house type
            'monthly_rent': rent,
            'family_size': family_size,
            'dependents': dependents,
            'school_fees': 0,  # Simplified for demo
            'college_fees': 0,
            'travel_expenses': 2000,
            'groceries_utilities': 5000,
            'other_monthly_expenses': expenses,
            'existing_loans': existing_loans,
            'current_emi_amount': existing_emi,
            'credit_score': credit_score,
            'bank_balance': salary * 2,  # Estimation
            'emergency_fund': emergency_fund,
            'emi_scenario': scenario,
            'requested_amount': loan_amount,
            'requested_tenure': tenure
        }

        input_df = pd.DataFrame([input_data])

        # 1. Predict Eligibility
        classifier = artifacts['classifier']
        eligibility_prob = classifier.predict_proba(
            input_df)[0][1]  # Probability of 'Eligible'
        eligibility_pred = classifier.predict(input_df)[0]

        # 2. Predict Max EMI
        regressor = artifacts['regressor']
        max_emi_pred = regressor.predict(input_df)[0]

        # ==========================================
        # 4. RESULTS DISPLAY
        # ==========================================
        st.divider()
        st.subheader("Risk Assessment Results")

        col1, col2 = st.columns(2)

        with col1:
            if eligibility_pred == 'Eligible' or eligibility_pred == 1:
                st.success(f"✅ **Status: ELIGIBLE**")
                st.write(f"Confidence Score: {eligibility_prob:.2%}")
            else:
                st.error(f"❌ **Status: NOT ELIGIBLE**")
                st.write(f"Risk Probability: {(1-eligibility_prob):.2%}")

        with col2:
            st.metric(label="Max Affordability (Monthly EMI)",
                      value=f"₹ {max_emi_pred:,.2f}")

        # Insights Expander
        with st.expander("See Detailed Financial Health Check"):
            st.write(
                f"**Debt-to-Income Ratio:** {((existing_emi + max_emi_pred)/salary):.2%}")
            st.write(
                f"**Credit Health:** {'Excellent' if credit_score > 750 else 'Good' if credit_score > 650 else 'Needs Improvement'}")
            st.info(
                "Note: The Max EMI is an AI-generated estimate based on your disposable income and spending patterns.")

else:
    st.warning("Please train the model first using the training script.")

# Footer
st.markdown("---")
st.caption("EMIPredict AI - Powered by Streamlit & Scikit-Learn")
