# 💰 EMIPredict AI - Intelligent Financial Risk Assessment

## 📌 Project Overview
EMIPredict AI is a comprehensive FinTech platform designed to automate loan decisioning. [cite_start]It utilizes a dual-model Machine Learning approach to assess a borrower's eligibility and calculate their maximum affordable EMI in real-time[cite: 13].

* **Domain:** FinTech & Banking
* [cite_start]**Tech Stack:** Python, Streamlit, Scikit-Learn, MLflow [cite: 13]

## 🎯 Problem Statement
Manual underwriting is time-consuming and prone to human error. This project automates the risk assessment process to:
* [cite_start]Reduce manual underwriting time by up to **80%**[cite: 13].
* [cite_start]Provide instant eligibility checks for digital lending platforms[cite: 13].
* [cite_start]Implement risk-based pricing strategies[cite: 13].

## 🧠 Machine Learning Architecture
The system solves two distinct problems simultaneously:
1.  **Classification Model (Eligibility):**
    * **Goal:** Predict if a user is `Eligible` or `Not Eligible` for a loan.
    * **Algorithm:** Random Forest Classifier.
    * **Features:** Credit Score, Employment History, Existing Loans, Income.
2.  **Regression Model (Affordability):**
    * **Goal:** Estimate the `Max_Monthly_EMI` the user can afford.
    * **Algorithm:** Random Forest Regressor.
    * **Logic:** Analyzes disposable income after rent, obligations, and living expenses.

## 🛠️ Features
* [cite_start]**Real-Time Interface:** A user-friendly web app built with **Streamlit** for instant predictions[cite: 13].
* [cite_start]**Experiment Tracking:** Integration with **MLflow** to track model versions and performance metrics[cite: 13].
* [cite_start]**Advanced Feature Engineering:** Processed over 400,000 records with 22+ financial variables including `Debt-to-Income Ratio` and `Credit Utilization`[cite: 13].

## 🚀 How to Run

### Prerequisites
* Python 3.8+
* Streamlit

### Installation
1.  **Install Requirements:**
    ```bash
    pip install pandas scikit-learn streamlit joblib
    ```

### Execution
1.  **Train the Models:**
    (This will generate the `emi_models.pkl` file)
    ```bash
    python model_training.py
    ```
2.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

## 📊 Dataset
The model is trained on `emi_prediction_dataset.csv`, containing 400,000+ records of applicant financial data, including:
* **Demographics:** Age, Education, Family Size.
* **Financials:** Income, Expenses, Credit Score, Existing Liabilities.
* [cite_start]**Loan Details:** Requested Amount, Tenure, Purpose[cite: 13].

---
