# Churn-Prediction

# Customer Churn Prediction

A machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project covers end-to-end data preprocessing, model training, evaluation, rule-based logic, and an interactive Streamlit dashboard.

---

## Project Structure

```
customer-churn-prediction/
│
├── telco_churn.csv               # Raw dataset
├── churn_notebook.ipynb          # Main Jupyter Notebook
├── streamlit_app.py              # Interactive Streamlit dashboard
├── model_comparison_results.csv  # Model metrics export
├── high_risk_customers.csv       # High-risk customer list from rule-based logic
└── README.md                     # Project documentation
```

---

## Dataset

- Source: IBM Telco Customer Churn (Kaggle)
- Records: 7,043 customers
- Features: 20 columns
- Target: Churn (Yes = Churned, No = Stayed)
- Churn Rate: 26.5% — realistic class imbalance

### Field Mapping to Task Requirements

| Task Field   | Telco Column                                      | Relation         |
|--------------|---------------------------------------------------|------------------|
| Age          | SeniorCitizen (0/1)                               | Age proxy        |
| Income       | MonthlyCharges, TotalCharges                      | Spending proxy   |
| Purchases    | StreamingTV, StreamingMovies, OnlineBackup, etc.  | Services bought  |
| Membership   | Contract (Month-to-month / One year / Two year)   | Direct match     |
| Churn        | Churn                                             | Direct match     |

---

## Tasks Completed

1. Load Dataset — CSV loaded, shape and types inspected
2. Clean Dataset — TotalCharges fixed, nulls handled, customerID dropped
3. Convert Categorical Data — Label Encoding for binary, One-Hot for multi-class
4. Train ML Models — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. Compare Algorithms — All 4 models compared across 5 metrics
6. Show Accuracy — Metrics table, bar chart, confusion matrices, ROC curves
7. Rule-Based Churn Logic — 6-rule scoring system with risk levels
8. Explain Best Model — Gradient Boosting selected with justification

---

## Models Used

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7317   | 0.4966    | 0.7781 | 0.6062   | 0.8412  |
| Decision Tree       | 0.7268   | 0.4903    | 0.7460 | 0.5917   | 0.7917  |
| Random Forest       | 0.7722   | 0.5718    | 0.5642 | 0.5680   | 0.8187  |
| Gradient Boosting   | 0.7495   | 0.5186    | 0.7834 | 0.6241   | 0.8466  |
| Rule-Based Logic    | 0.6842   | 0.4493    | 0.8411 | 0.5857   | N/A     |

Best Model: Gradient Boosting — highest ROC-AUC (0.8466) and F1-Score (0.6241)

---

## Rule-Based Churn Logic

Six conditions are checked per customer. Each condition satisfied adds 1 point to the risk score.

| Rule | Condition                          |
|------|------------------------------------|
| 1    | Contract is Month-to-month         |
| 2    | Tenure less than 12 months         |
| 3    | Monthly Charges greater than 70    |
| 4    | Internet Service is Fiber optic    |
| 5    | Payment Method is Electronic check |
| 6    | No Tech Support with internet      |

- Score 3 or more: HIGH risk — predicted as churn
- Score 2: MEDIUM risk — monitor
- Score 1 or less: LOW risk — predicted as stayed

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook churn_notebook.ipynb
```

### Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

---

## Key Findings

- Month-to-month contract customers churn at 43% versus 11% for two-year contracts
- Fiber optic internet users churn at nearly double the rate of DSL users
- Electronic check payment method has the highest churn rate among all payment types
- Senior citizens churn at almost twice the rate of non-senior customers
- Customers who churn have significantly lower average tenure (18 months versus 38 months)
- Higher monthly charges are consistently associated with higher churn probability

---
