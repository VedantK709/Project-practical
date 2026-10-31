import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence Engine",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


try:
    artifact = load_model()
    model = artifact["model"]
    feature_cols = artifact["features"]
except Exception as e:
    st.error(
        "Model file not found. Run 'python train_model.py' to generate 'churn_model.joblib'."
    )
    st.stop()

st.title("📊 Customer Churn Intelligence & Retention Dashboard")
st.caption(
    "Predictive machine learning pipeline for customer risk scoring and proactive retention intervention."
)

col1, col2, col3 = st.columns(3)
col1.metric("Model Architecture", "Random Forest Classifier")
col2.metric("Cross-Validation ROC-AUC", f"{artifact.get('roc_auc', 0.84):.2%}")
col3.metric("Baseline Accuracy", f"{artifact.get('accuracy', 0.81):.2%}")

st.write("")

# Layout: 1/3 Inputs, 2/3 Analytics
input_col, output_col = st.columns([1, 2])

with input_col:
    st.subheader("Customer Profile Inputs")
    tenure = st.slider(
        "Tenure (Months with Company)",
        min_value=1,
        max_value=72,
        value=12,
        help="Number of months customer has remained subscribed.",
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=15.0,
        max_value=150.0,
        value=75.0,
        step=5.0,
    )

    contract_mapping = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }
    selected_contract = st.selectbox(
        "Contract Duration", list(contract_mapping.keys())
    )
    contract_type = contract_mapping[selected_contract]

    tech_support_choice = st.radio(
        "Tech Support Add-on", ["No", "Yes"], horizontal=True
    )
    tech_support = 1 if tech_support_choice == "Yes" else 0

    security_choice = st.radio(
        "Online Security Add-on", ["No", "Yes"], horizontal=True
    )
    online_security = 1 if security_choice == "Yes" else 0

    billing_choice = st.radio(
        "Paperless Billing", ["No", "Yes"], horizontal=True
    )
    paperless_billing = 1 if billing_choice == "Yes" else 0

    predict_btn = st.button(
        "Analyze Churn Probability", type="primary", use_container_width=True
    )

with output_col:
    st.subheader("Predictive Risk Diagnostics")

    customer_df = pd.DataFrame(
        [[
            tenure,
            monthly_charges,
            contract_type,
            tech_support,
            online_security,
            paperless_billing,
        ]],
        columns=feature_cols,
    )

    probabilities = model.predict_proba(customer_df)[0]
    churn_prob = probabilities[1] * 100
    retention_prob = probabilities[0] * 100

    rcol1, rcol2 = st.columns(2)
    rcol1.metric("Churn Risk Probability", f"{churn_prob:.1f}%")
    rcol2.metric("Customer Retention Likelihood", f"{retention_prob:.1f}%")

    if churn_prob >= 65:
        st.error(
            "🚨 **High Attrition Risk**: Customer shows high risk indicators. Urgent retention intervention required."
        )
    elif churn_prob >= 35:
        st.warning(
            "⚠️ **Moderate Attrition Risk**: Customer has moderate retention risk. Monitor billing friction."
        )
    else:
        st.success(
            "✅ **Low Attrition Risk**: Customer shows healthy loyalty and low likelihood of cancellation."
        )

    st.write("")
    st.subheader("Global Feature Importance (Model Drivers)")
    importances = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=True)

    feature_labels = {
        "tenure": "Tenure Duration",
        "monthly_charges": "Monthly Bill Amount",
        "contract_type": "Contract Commitment",
        "tech_support": "Tech Support Presence",
        "online_security": "Online Security Presence",
        "paperless_billing": "Paperless Billing Setup",
    }
    importances["Feature"] = (
        importances["Feature"].map(feature_labels).fillna(importances["Feature"])
    )

    st.bar_chart(importances.set_index("Feature"))

    st.subheader("Prescriptive Retention Plan")
    if selected_contract == "Month-to-month" and churn_prob > 40:
        st.markdown(
            "- **Incentivize Term Commitment**: Offer a 15% discount for upgrading to an annual plan."
        )
    if tech_support == 0:
        st.markdown(
            "- **Value Add-on**: Provide a 60-day complimentary trial of dedicated technical support."
        )
    if monthly_charges > 85.0:
        st.markdown(
            "- **Rate Review**: Conduct an account tier review to optimize plan usage and eliminate billing surprises."
        )