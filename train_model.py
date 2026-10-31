import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def generate_and_train():
    np.random.seed(42)
    n_samples = 3000

    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    contract_type = np.random.choice(
        [0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]
    )  # 0: Month-to-month, 1: One year, 2: Two year
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    online_security = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])

    # Realistic churn probability logic: high charges + month-to-month = higher churn risk
    logit = (
        -1.5
        + (monthly_charges * 0.025)
        - (tenure * 0.045)
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

    print("=== Model Training Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    artifact = {
        "model": model,
        "features": feature_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    joblib.dump(artifact, "churn_model.joblib")
    print("Model artifact successfully saved as 'churn_model.joblib'")


if __name__ == "__main__":
    generate_and_train()