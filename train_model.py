import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def generate_and_train():
    np.random.seed(42)
    n_samples = 3500

    # Widened ranges: 1 to 120 months, ₹199 to ₹4,999 monthly charges
    tenure = np.random.randint(1, 121, size=n_samples)
    monthly_charges = np.random.uniform(199.0, 4999.0, size=n_samples)
    contract_type = np.random.choice(
        [0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    online_security = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])

    # Calibrated churn logit for Indian currency and 10-year tenure range
    logit = (
        -1.2
        + (monthly_charges * 0.0006)
        - (tenure * 0.028)
        - (contract_type * 1.1)
        - (tech_support * 0.7)
        - (online_security * 0.6)
        + (paperless_billing * 0.4)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )

    feature_cols = [
        "tenure",
        "monthly_charges",
        "contract_type",
        "tech_support",
        "online_security",
        "paperless_billing",
    ]
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    print(f"Retrained on INR. Test Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

    artifact = {
        "model": model,
        "features": feature_cols,
        "accuracy": acc,
        "roc_auc": auc,
    }
    joblib.dump(artifact, "churn_model.joblib")
    print("Exported updated 'churn_model.joblib'")


if __name__ == "__main__":
    generate_and_train()