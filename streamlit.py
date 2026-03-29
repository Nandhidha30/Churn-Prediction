import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("Customer Churn Predictor")
st.markdown("Powered by Gradient Boosting — Best Model (ROC-AUC: 0.8466 | F1: 0.6241)")
st.markdown("---")


@st.cache_data
def train_model():
    df = pd.read_csv("telco_churn.csv")

    df_clean = df.copy()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].str.strip()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    df_clean = df_clean.drop(columns=['customerID'])

    df_encoded = df_clean.copy()
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines']
    le = LabelEncoder()
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    multi_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cols, drop_first=False)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_combined = pd.concat([X_train, y_train], axis=1)
    stayed  = X_train_combined[X_train_combined['Churn'] == 0]
    churned = X_train_combined[X_train_combined['Churn'] == 1]
    churned_upsampled = resample(churned, replace=True, n_samples=len(stayed), random_state=42)
    train_balanced = pd.concat([stayed, churned_upsampled])
    X_train_bal = train_balanced.drop(columns=['Churn'])
    y_train_bal  = train_balanced['Churn']

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    return model, X_train_bal


model, X_train_bal = train_model()

# ── Input Form ───────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.subheader("Phone & Internet")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Prediction Result")
    st.markdown("Click below to predict using Gradient Boosting.")

    predict_btn = st.button("Predict Churn", use_container_width=True)

    if predict_btn:
        total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

        input_data = {
            'SeniorCitizen': senior,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'gender': 1 if gender == 'Male' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
            'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        }

        all_multi_cols = {
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'No internet service', 'Yes'],
            'OnlineBackup': ['No', 'No internet service', 'Yes'],
            'DeviceProtection': ['No', 'No internet service', 'Yes'],
            'TechSupport': ['No', 'No internet service', 'Yes'],
            'StreamingTV': ['No', 'No internet service', 'Yes'],
            'StreamingMovies': ['No', 'No internet service', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)',
                              'Electronic check', 'Mailed check']
        }

        selected = {
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaymentMethod': payment_method
        }

        for col_name, options in all_multi_cols.items():
            for opt in options:
                input_data[f'{col_name}_{opt}'] = 1 if selected[col_name] == opt else 0

        input_df = pd.DataFrame([input_data])
        trained_cols = X_train_bal.columns.tolist()
        for c in trained_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[trained_cols]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        stay_prob = model.predict_proba(input_df)[0][0]

        st.markdown("---")

        if pred == 1:
            st.error(f"CHURN — This customer is likely to leave.")
        else:
            st.success(f"STAY — This customer is likely to stay.")

        st.markdown(f"**Churn Probability : {prob*100:.1f}%**")
        st.markdown(f"**Stay Probability  : {stay_prob*100:.1f}%**")

        st.progress(int(prob * 100))

        if prob >= 0.75:
            risk = "HIGH"
            st.error(f"Risk Level: {risk}")
        elif prob >= 0.45:
            risk = "MEDIUM"
            st.warning(f"Risk Level: {risk}")
        else:
            risk = "LOW"
            st.success(f"Risk Level: {risk}")