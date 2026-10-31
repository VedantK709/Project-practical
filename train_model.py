import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def generate_and_train():
    np.random.seed(42)
    n_samples = 4000

    # 1. Expanded tenure (3 to 60 months)
    tenure = np.random.randint(3, 61, size=n_samples)

    # 2. Expanded monthly charges (₹150 to ₹50,000)
    monthly_charges = np.random.uniform(150.0, 50000.0, size=n_samples)

    # 3. Contract commitment in months (1, 3, 6, 12, 24, 36, etc.)
    contract_months = np.random.choice(
        [1, 3, 6, 12, 24], size=n_samples, p=[0.40, 0.20, 0.15, 0.15, 0.10]
    )

    # 4. Five distinct service/variety fields
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    online_security = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    device_protection = np.random.choice(
        [0, 1], size=n_samples, p=[0.65, 0.35]
    )
    streaming_bundle = np.random.choice([0, 1], size=n_samples, p=[0.50, 0.50])
    paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])

    # Calibrated risk logit across 8 dimensions
    logit = (
        -0.85
        + (monthly_charges * 0.000075)
        - (tenure * 0.035)
        - (contract_months * 0.075)
        - (tech_support * 0.55)
        - (online_security * 0.45)
        - (device_protection * 0.35)
        - (streaming_bundle * 0.30)
        + (paperless_billing * 0.30)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

    feature_cols = [
        "tenure",
        "monthly_charges",
        "contract_months",
        "tech_support",
        "online_security",
        "device_protection",
        "streaming_bundle",
        "paperless_billing",
    ]

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_months": contract_months,
            "tech_support": tech_support,
            "online_security": online_security,
            "device_protection": device_protection,
            "streaming_bundle": streaming_bundle,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )

    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=7, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    print(
        f"Model successfully retrained on 8 features. Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}"
    )

    artifact = {
        "model": model,
        "features": feature_cols,
        "accuracy": acc,
        "roc_auc": auc,
    }
    joblib.dump(artifact, "churn_model.joblib")
    print("Saved updated artifact: 'churn_model.joblib'")


if __name__ == "__main__":
    generate_and_train()