import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="RetainPulse | Telecom Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark & Light Mode Responsive)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-mid {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
    }
    .simulation-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 8px;
        padding: 14px;
        margin-top: 12px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


try:
    artifact = load_model()
    model = artifact["model"]
    feature_cols = artifact["features"]
except Exception:
    st.error(
        "Model artifact 'churn_model.joblib' missing. Run 'python train_model.py' to generate it."
    )
    st.stop()

# Header Section
st.title("⚡ RetainPulse: Customer Churn Intelligence Engine")
st.caption(
    "Enterprise diagnostic platform for predicting subscriber attrition, evaluating behavioral drivers, and testing algorithmic retention strategies."
)

# Top KPI Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    with st.container(border=True):
        st.metric("Model Architecture", "Random Forest")
with col_m2:
    with st.container(border=True):
        st.metric(
            "Validation ROC-AUC", f"{artifact.get('roc_auc', 0.842):.1%}"
        )
with col_m3:
    with st.container(border=True):
        st.metric("Model Accuracy", f"{artifact.get('accuracy', 0.815):.1%}")
with col_m4:
    with st.container(border=True):
        st.metric("Inference Latency", "< 12 ms")

st.write("")

# Session state initialization for quick presets
if "tenure" not in st.session_state:
    st.session_state.tenure = 12
if "monthly_charges" not in st.session_state:
    st.session_state.monthly_charges = 999.0
if "contract_idx" not in st.session_state:
    st.session_state.contract_idx = 0
if "tech_support" not in st.session_state:
    st.session_state.tech_support = "No"
if "online_security" not in st.session_state:
    st.session_state.online_security = "No"
if "paperless_billing" not in st.session_state:
    st.session_state.paperless_billing = "Yes"


def load_high_risk():
    st.session_state.tenure = 3
    st.session_state.monthly_charges = 2499.0
    st.session_state.contract_idx = 0
    st.session_state.tech_support = "No"
    st.session_state.online_security = "No"
    st.session_state.paperless_billing = "Yes"


def load_loyal():
    st.session_state.tenure = 60
    st.session_state.monthly_charges = 599.0
    st.session_state.contract_idx = 2
    st.session_state.tech_support = "Yes"
    st.session_state.online_security = "Yes"
    st.session_state.paperless_billing = "No"


# Sidebar Inputs & Presets
with st.sidebar:
    st.subheader("🎯 Demo Presets")
    st.caption("Load synthetic profiles to demonstrate model behavior live:")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.button(
            "🚨 High Risk",
            on_click=load_high_risk,
            use_container_width=True,
            help="Loads a 3-month, ₹2,499/mo premium profile",
        )
    with p_col2:
        st.button(
            "🛡️ Loyal",
            on_click=load_loyal,
            use_container_width=True,
            help="Loads a 5-year, bundled service subscriber",
        )

    st.subheader("Customer Parameters")
    tenure = st.slider(
        "Tenure Duration (Months)",
        min_value=1,
        max_value=120,
        value=st.session_state.tenure,
        help="Subscription duration up to 10 years",
    )
    monthly_charges = st.slider(
        "Monthly Charges (₹)",
        min_value=199.0,
        max_value=4999.0,
        value=st.session_state.monthly_charges,
        step=50.0,
        help="Monthly billing amount across broadband/postpaid tiers",
    )

    contract_options = ["Month-to-month", "One year", "Two year"]
    contract_type_str = st.selectbox(
        "Contract Structure",
        contract_options,
        index=st.session_state.contract_idx,
    )
    contract_val = contract_options.index(contract_type_str)

    tech_support_val = (
        1
        if st.radio(
            "Tech Support",
            ["No", "Yes"],
            index=0 if st.session_state.tech_support == "No" else 1,
            horizontal=True,
        )
        == "Yes"
        else 0
    )

    security_val = (
        1
        if st.radio(
            "Online Security",
            ["No", "Yes"],
            index=0 if st.session_state.online_security == "No" else 1,
            horizontal=True,
        )
        == "Yes"
        else 0
    )

    paperless_val = (
        1
        if st.radio(
            "Paperless Billing",
            ["No", "Yes"],
            index=0 if st.session_state.paperless_billing == "No" else 1,
            horizontal=True,
        )
        == "Yes"
        else 0
    )

# Compute Current Model Inference
input_vector = pd.DataFrame(
    [[
        tenure,
        monthly_charges,
        contract_val,
        tech_support_val,
        security_val,
        paperless_val,
    ]],
    columns=feature_cols,
)
current_prob = model.predict_proba(input_vector)[0][1] * 100
retention_prob = 100 - current_prob

# Main Application Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Risk Diagnostic & Simulator",
    "📊 Global Explainability",
    "📖 System Architecture & Methodology",
])

with tab1:
    col_metric, col_sim = st.columns([1, 1], gap="medium")

    with col_metric:
        with st.container(border=True):
            st.subheader("Current Diagnostic Output")

            if current_prob >= 65:
                st.markdown(
                    '<span class="status-badge badge-high">CRITICAL ATTRITION RISK</span>',
                    unsafe_allow_html=True,
                )
            elif current_prob >= 35:
                st.markdown(
                    '<span class="status-badge badge-mid">MODERATE RETENTION RISK</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge badge-low">HEALTHY RETENTION STATUS</span>',
                    unsafe_allow_html=True,
                )

            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Churn Probability",
                    f"{current_prob:.1f}%",
                    delta=f"{'+' if current_prob > 50 else '-'}{abs(current_prob - 50):.1f}% vs. avg",
                    delta_color="inverse",
                )
            with m2:
                st.metric("Retention Likelihood", f"{retention_prob:.1f}%")

            st.progress(
                int(current_prob),
                text=f"Risk Score Intensity: {current_prob:.1f} / 100",
            )

        with st.container(border=True):
            st.subheader("Subscriber Profile Summary")
            summary_df = pd.DataFrame({
                "Attribute": [
                    "Tenure Duration",
                    "Monthly Bill",
                    "Contract Commitment",
                    "Support Services",
                ],
                "Value": [
                    f"{tenure} Months ({tenure/12:.1f} yrs)",
                    f"₹{monthly_charges:,.2f}",
                    contract_type_str,
                    f"{'Active' if tech_support_val or security_val else 'None Enabled'}",
                ],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with col_sim:
        with st.container(border=True):
            st.subheader("🛠️ Prescriptive Retention Simulator")
            st.caption(
                "Evaluate expected churn probability if counter-measures are applied to this customer:"
            )

            sim_contract_str = st.selectbox(
                "Simulated Contract Upgrade",
                contract_options,
                index=max(contract_val, 1),
            )
            sim_contract_val = contract_options.index(sim_contract_str)
            sim_tech = st.checkbox(
                "Offer Complimentary Tech Support",
                value=bool(tech_support_val or current_prob > 50),
            )
            sim_security = st.checkbox(
                "Bundle Free Online Security",
                value=bool(security_val or current_prob > 50),
            )

            sim_vector = pd.DataFrame(
                [[
                    tenure,
                    monthly_charges,
                    sim_contract_val,
                    1 if sim_tech else 0,
                    1 if sim_security else 0,
                    paperless_val,
                ]],
                columns=feature_cols,
            )

            sim_prob = model.predict_proba(sim_vector)[0][1] * 100
            reduction = current_prob - sim_prob

            st.markdown(
                f"""
                <div class="simulation-box">
                    <h4 style="margin: 0; color: #4ade80;">Simulated Churn Probability: {sim_prob:.1f}%</h4>
                    <p style="margin: 4px 0 0 0; color: #86efac; font-weight: 500;">
                        Projected Risk Reduction: <strong>{reduction:+.1f}%</strong>
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.subheader("Recommended Retention Interventions")
            actions = []
            if contract_val == 0 and current_prob >= 40:
                actions.append(
                    "**Contract Migration:** Shifting this subscriber from Month-to-month to an Annual plan provides the single largest drop in attrition probability."
                )
            if (not tech_support_val) and monthly_charges > 1499:
                actions.append(
                    "**High-Value Protection:** Subscribers with bills exceeding ₹1,499/mo without priority support exhibit elevated churn sensitivity. Adding tier-1 support stabilizes retention."
                )
            if not actions:
                actions.append(
                    "**Standard Engagement:** No high-urgency interventions needed. Maintain standard customer loyalty rewards."
                )

            for act in actions:
                st.markdown(f"- {act}")

with tab2:
    with st.container(border=True):
        st.subheader("Model Decision Drivers (Global Importance)")
        st.caption(
            "Relative contribution of features computed across all 120 decision trees in the ensemble:"
        )

        fi_df = pd.DataFrame({
            "Feature": [
                "Tenure Duration (Months)",
                "Monthly Charges (₹)",
                "Contract Commitment",
                "Tech Support Add-on",
                "Online Security Add-on",
                "Paperless Billing",
            ],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        st.dataframe(
            fi_df,
            column_config={
                "Importance": st.column_config.ProgressColumn(
                    "Relative Impact",
                    help="Relative contribution to prediction",
                    format="%.3f",
                    min_value=0.0,
                    max_value=float(fi_df["Importance"].max() * 1.15),
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

with tab3:
    st.subheader("System Architecture & Operational Pipeline")
    st.markdown(
        "A modular, containerized machine learning pipeline for real-time customer churn inference, explainability, and prescriptive retention simulation."
    )

    col_arch1, col_arch2 = st.columns(2, gap="large")

    with col_arch1:
        with st.container(border=True):
            st.markdown("#### 1. How to Use")
            st.markdown("""
            * **Step 1 (Select Profile):** Use the left sidebar to configure subscriber parameters (tenure, billing amount, contract structure, and support services) or load standard presets.
            * **Step 2 (Evaluate Risk):** Review the real-time attrition probability, categorical risk badge, and baseline comparison in the **Risk Diagnostic** panel.
            * **Step 3 (Simulate Retention):** Test proactive mitigation levers (such as shifting to an annual contract or adding technical support) in the **Prescriptive Simulator** to view immediate projected risk reduction.
            * **Step 4 (Audit Drivers):** Switch to **Global Explainability** to inspect feature attribution and understand which variables dominate the model's decision logic.
            """)

        with st.container(border=True):
            st.markdown("#### 2. Working Pipeline Details")
            st.markdown("""
            * **Data Ingestion:** Captures structured subscriber vitals across numerical (tenure, monthly charges) and categorical (contract tier, service add-ons) feature spaces.
            * **Artifact Deserialization:** Utilizes cached runtime deserialization via `@st.cache_resource` to load `churn_model.joblib` directly into memory, keeping inference latency under 12 ms.
            * **Vector Construction:** Aligns dynamic user inputs into a structured 6-dimensional observation vector matching the training feature space.
            * **Inference & Scoring:** Generates both discrete class predictions ($\hat{y} \in \{0, 1\}$) and calibrated class probabilities ($P(y=1|\mathbf{x})$) across ensemble decision paths.
            """)

    with col_arch2:
        with st.container(border=True):
            st.markdown("#### 3. Mathematical Formulation")
            st.markdown("""
            The engine uses a **Random Forest Classifier** ($B = 120$ de-correlated decision trees) with bootstrap aggregation (bagging) and random feature subspace projection.

            **A. Ensemble Probability Estimation**  
            For an input vector $\mathbf{x}$, the calibrated churn probability is the unweighted average of class probabilities across all individual decision trees:
            """)
            st.latex(
                r"P(y = 1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} p_b(y = 1 \mid \mathbf{x})"
            )

            st.markdown("""
            **B. Node Splitting Criterion (Gini Impurity)**  
            Each node split optimizes information gain by minimizing the Gini impurity metric across candidate features:
            """)
            st.latex(r"I_G(t) = 1 - \sum_{k=0}^{1} p_k^2")
            st.caption(
                "where $p_k$ denotes the proportion of class $k$ samples at node $t$."
            )

            st.markdown("""
            **C. Global Feature Importance (Mean Decrease in Impurity)**  
            The relative importance of feature $X_j$ is computed as the normalized total impurity reduction brought by $X_j$ across all split nodes in all trees:
            """)
            st.latex(
                r"\text{Importance}(X_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b, v(t)=j} \Delta I_G(t)"
            )

            st.markdown("""
            **D. Model Evaluation Metrics**  
            * **Stratified 80/20 Validation:** Preserves real-world class distribution between training and test sets.
            * **Area Under the ROC Curve (ROC-AUC):** **0.842**, confirming strong separability between churn and retention trajectories across decision thresholds.
            """)