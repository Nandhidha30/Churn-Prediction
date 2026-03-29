import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction")
st.markdown("IBM Telco Dataset — Machine Learning + Rule-Based Analysis")
st.markdown("---")


@st.cache_data
def load_and_prepare():
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

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train_bal)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall':    round(recall_score(y_test, y_pred), 4),
            'F1-Score':  round(f1_score(y_test, y_pred), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
            'y_pred': y_pred, 'y_prob': y_prob
        }
        trained_models[name] = model

    return df, df_clean, X_test, y_test, results, trained_models, X_train_bal, scaler


df, df_clean, X_test, y_test, results, trained_models, X_train_bal, scaler = load_and_prepare()


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "EDA",
    "Model Comparison",
    "ROC Curves",
    "Feature Importance",
    "Rule-Based Predictor",
    "Predict a Customer"
])


# ── Page 1: Dataset Overview ─────────────────────────────────
if page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Churned", f"{(df['Churn'] == 'Yes').sum():,}")
    col4.metric("Churn Rate", "26.5%")

    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Data Types")
    dtype_df = pd.DataFrame(df.dtypes, columns=['Dtype']).astype(str)
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ── Page 2: EDA ──────────────────────────────────────────────
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle('EDA Overview', fontsize=15, fontweight='bold')

    churn_counts = df['Churn'].value_counts()
    bars = axes[0, 0].bar(churn_counts.index, churn_counts.values,
                          color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_ylabel('Count')
    for bar, val in zip(bars, churn_counts.values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, val + 40,
                        f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    axes[0, 0].set_ylim(0, max(churn_counts.values) * 1.2)

    contract_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    axes[0, 1].bar(contract_rate.index, contract_rate.values,
                   color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
    axes[0, 1].set_title('Churn Rate by Contract Type')
    axes[0, 1].set_ylabel('Churn Rate (%)')
    for i, val in enumerate(contract_rate.values):
        axes[0, 1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')

    internet_rate = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    axes[0, 2].bar(internet_rate.index, internet_rate.values,
                   color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
    axes[0, 2].set_title('Churn Rate by Internet Service')
    axes[0, 2].set_ylabel('Churn Rate (%)')
    for i, val in enumerate(internet_rate.values):
        axes[0, 2].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')

    axes[1, 0].hist(df[df['Churn'] == 'No']['tenure'], bins=30, alpha=0.65,
                    color='#2ecc71', label='Stayed', edgecolor='white')
    axes[1, 0].hist(df[df['Churn'] == 'Yes']['tenure'], bins=30, alpha=0.65,
                    color='#e74c3c', label='Churned', edgecolor='white')
    axes[1, 0].set_title('Tenure Distribution by Churn')
    axes[1, 0].set_xlabel('Tenure (months)')
    axes[1, 0].legend()

    axes[1, 1].hist(df[df['Churn'] == 'No']['MonthlyCharges'], bins=30, alpha=0.65,
                    color='#2ecc71', label='Stayed', edgecolor='white')
    axes[1, 1].hist(df[df['Churn'] == 'Yes']['MonthlyCharges'], bins=30, alpha=0.65,
                    color='#e74c3c', label='Churned', edgecolor='white')
    axes[1, 1].set_title('Monthly Charges by Churn')
    axes[1, 1].set_xlabel('Monthly Charges ($)')
    axes[1, 1].legend()

    pay_rate = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    short_labels = ['Bank\nTransfer', 'Credit\nCard', 'Electronic\nCheck', 'Mailed\nCheck']
    axes[1, 2].bar(range(len(pay_rate)), pay_rate.values,
                   color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], edgecolor='black')
    axes[1, 2].set_xticks(range(len(pay_rate)))
    axes[1, 2].set_xticklabels(short_labels, fontsize=8)
    axes[1, 2].set_title('Churn Rate by Payment Method')
    for i, val in enumerate(pay_rate.values):
        axes[1, 2].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=8)

    senior_rate = df.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    axes[2, 0].bar(['Non-Senior', 'Senior'], senior_rate.values,
                   color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[2, 0].set_title('Churn Rate by Senior Citizen')
    for i, val in enumerate(senior_rate.values):
        axes[2, 0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

    ts_rate = df.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    axes[2, 1].bar(ts_rate.index, ts_rate.values,
                   color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
    axes[2, 1].set_title('Churn Rate by Tech Support')
    for i, val in enumerate(ts_rate.values):
        axes[2, 1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

    os_rate = df.groupby('OnlineSecurity')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    axes[2, 2].bar(os_rate.index, os_rate.values,
                   color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
    axes[2, 2].set_title('Churn Rate by Online Security')
    for i, val in enumerate(os_rate.values):
        axes[2, 2].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


# ── Page 3: Model Comparison ─────────────────────────────────
elif page == "Model Comparison":
    st.header("Model Comparison")

    metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_data = {
        name: {k: v for k, v in vals.items() if k in metrics_plot}
        for name, vals in results.items()
    }
    metrics_df = pd.DataFrame(metrics_data).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
                 use_container_width=True)

    COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    x = np.arange(len(metrics_plot))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, color) in enumerate(zip(results.keys(), COLORS)):
        vals = [results[name][m] for m in metrics_plot]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_plot, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.legend()
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Confusion Matrices")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    from sklearn.metrics import ConfusionMatrixDisplay
    for i, (name, vals) in enumerate(results.items()):
        cm = confusion_matrix(y_test, vals['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(f'{name}\nF1={vals["F1-Score"]:.4f} | AUC={vals["ROC-AUC"]:.4f}',
                          fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)


# ── Page 4: ROC Curves ───────────────────────────────────────
elif page == "ROC Curves":
    st.header("ROC Curves")

    COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, vals), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, vals['y_prob'])
        ax.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {vals["ROC-AUC"]:.4f})', color=color)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All 4 Models', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    A higher ROC-AUC means the model is better at ranking churners above non-churners.
    Gradient Boosting leads with 0.8466, making it the most reliable model for deployment.
    """)


# ── Page 5: Feature Importance ───────────────────────────────
elif page == "Feature Importance":
    st.header("Feature Importance — Random Forest")

    rf_model = trained_models['Random Forest']
    feat_imp = pd.Series(rf_model.feature_importances_, index=X_train_bal.columns)
    feat_imp_top = feat_imp.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(feat_imp_top)))[::-1]
    ax.barh(feat_imp_top.index[::-1], feat_imp_top.values[::-1],
            color=colors_fi, edgecolor='black', linewidth=0.6)
    for i, (val, name) in enumerate(zip(feat_imp_top.values[::-1], feat_imp_top.index[::-1])):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importances', fontweight='bold')
    ax.set_xlim(0, feat_imp_top.max() * 1.2)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Top 5 Churn Predictors")
    for feat, imp in feat_imp_top.head(5).items():
        st.write(f"- {feat}: {imp:.4f}")


# ── Page 6: Rule-Based Predictor ─────────────────────────────
elif page == "Rule-Based Predictor":
    st.header("Rule-Based Churn Logic")

    st.markdown("""
    Each customer is scored against 6 business rules. Every condition satisfied adds 1 point.

    | Rule | Condition |
    |------|-----------|
    | 1 | Contract is Month-to-month |
    | 2 | Tenure less than 12 months |
    | 3 | Monthly Charges greater than 70 |
    | 4 | Internet Service is Fiber optic |
    | 5 | Payment Method is Electronic check |
    | 6 | No Tech Support with active internet |

    - Score 3 or more: HIGH risk
    - Score 2: MEDIUM risk
    - Score 1 or less: LOW risk
    """)

    def rule_based_churn(row):
        score = 0
        flags = []
        if row['Contract'] == 'Month-to-month':
            score += 1; flags.append('MonthToMonth')
        if row['tenure'] < 12:
            score += 1; flags.append('LowTenure')
        if row['MonthlyCharges'] > 70:
            score += 1; flags.append('HighCharges')
        if row['InternetService'] == 'Fiber optic':
            score += 1; flags.append('FiberOptic')
        if row['PaymentMethod'] == 'Electronic check':
            score += 1; flags.append('ElectronicCheck')
        if row['TechSupport'] == 'No' and row['InternetService'] != 'No':
            score += 1; flags.append('NoTechSupport')
        if score >= 3:
            return 1, 'HIGH', score, '|'.join(flags)
        elif score == 2:
            return 0, 'MEDIUM', score, '|'.join(flags)
        else:
            return 0, 'LOW', score, '|'.join(flags)

    df_rules = df_clean.copy()
    rule_results = df_rules.apply(rule_based_churn, axis=1)
    df_rules['Rule_Pred']  = [r[0] for r in rule_results]
    df_rules['Risk_Level'] = [r[1] for r in rule_results]
    df_rules['Risk_Score'] = [r[2] for r in rule_results]
    df_rules['Risk_Flags'] = [r[3] for r in rule_results]

    risk_counts = df_rules['Risk_Level'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("HIGH Risk", risk_counts.get('HIGH', 0))
    col2.metric("MEDIUM Risk", risk_counts.get('MEDIUM', 0))
    col3.metric("LOW Risk", risk_counts.get('LOW', 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    risk_order = ['HIGH', 'MEDIUM', 'LOW']
    values = [risk_counts.get(r, 0) for r in risk_order]
    axes[0].bar(risk_order, values, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
    axes[0].set_title('Risk Level Distribution')
    axes[0].set_ylabel('Customers')
    for i, val in enumerate(values):
        axes[0].text(i, val + 10, str(val), ha='center', fontweight='bold')

    all_flags = '|'.join(df_rules['Risk_Flags']).split('|')
    flag_freq = pd.Series(all_flags).value_counts().head(6)
    axes[1].barh(flag_freq.index, flag_freq.values, color='#3498db', edgecolor='black')
    axes[1].set_title('Top Rule Triggers')
    axes[1].set_xlabel('Count')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### High Risk Customers Sample")
    st.dataframe(df_rules[df_rules['Risk_Level'] == 'HIGH'][[
        'tenure', 'Contract', 'MonthlyCharges', 'InternetService',
        'PaymentMethod', 'TechSupport', 'Risk_Score', 'Risk_Flags', 'Churn'
    ]].head(10), use_container_width=True)


# ── Page 7: Predict a Customer ───────────────────────────────
elif page == "Predict a Customer":
    st.header("Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details below and get a churn prediction from all 4 models.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        senior = st.selectbox("Senior Citizen", [0, 1])
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

    if st.button("Predict Churn"):
        input_data = {
            'SeniorCitizen': senior, 'tenure': tenure,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
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
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaymentMethod': payment_method
        }

        for col, options in all_multi_cols.items():
            for opt in options:
                input_data[f'{col}_{opt}'] = 1 if selected[col] == opt else 0

        input_df = pd.DataFrame([input_data])
        trained_cols = X_train_bal.columns.tolist()
        for col in trained_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_cols]

        input_sc = scaler.transform(input_df)

        st.markdown("### Predictions")
        pred_cols = st.columns(4)
        COLORS_PRED = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

        for i, (name, model) in enumerate(trained_models.items()):
            if name == 'Logistic Regression':
                pred = model.predict(input_sc)[0]
                prob = model.predict_proba(input_sc)[0][1]
            else:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]

            label = "CHURN" if pred == 1 else "STAY"
            with pred_cols[i]:
                st.markdown(f"**{name}**")
                if pred == 1:
                    st.error(f"{label} ({prob*100:.1f}%)")
                else:
                    st.success(f"{label} ({prob*100:.1f}%)")

        st.markdown("### Rule-Based Risk Score")
        score = 0
        flags = []
        if contract == 'Month-to-month': score += 1; flags.append('Month-to-month contract')
        if tenure < 12: score += 1; flags.append('Low tenure')
        if monthly_charges > 70: score += 1; flags.append('High monthly charges')
        if internet_service == 'Fiber optic': score += 1; flags.append('Fiber optic internet')
        if payment_method == 'Electronic check': score += 1; flags.append('Electronic check payment')
        if tech_support == 'No' and internet_service != 'No': score += 1; flags.append('No tech support')

        risk = 'HIGH' if score >= 3 else 'MEDIUM' if score == 2 else 'LOW'
        if risk == 'HIGH':
            st.error(f"Risk Level: {risk} (Score: {score}/6)")
        elif risk == 'MEDIUM':
            st.warning(f"Risk Level: {risk} (Score: {score}/6)")
        else:
            st.success(f"Risk Level: {risk} (Score: {score}/6)")

        if flags:
            st.markdown("**Risk Factors Triggered:**")
            for f in flags:
                st.write(f"- {f}")