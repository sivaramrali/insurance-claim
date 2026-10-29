
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Resolve paths from this script's location.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "vehicle-claims"
    / "synthetic_vehicle_insurance_claims.csv"
)

# Keep the existing health model untouched.
MODEL_PATH = APP_DIR / "vehicle_claim_model.joblib"

NUMERIC_FEATURES = [
    "vehicle_age",
    "driver_age",
    "vehicle_value",
    "claim_amount",
    "repair_cost",
    "previous_claims",
    "policy_duration_months",
    "documents_complete",
    "police_report_available",
]

CATEGORICAL_FEATURES = ["accident_severity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "claim_status"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    df = df[required_columns].dropna().copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    allowed_labels = {"approved", "denied"}
    unexpected_labels = set(df[TARGET].unique()) - allowed_labels

    if unexpected_labels:
        raise ValueError(f"Unexpected claim labels: {unexpected_labels}")

    print("Dataset:", DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nClaim status counts:")
    print(df[TARGET].value_counts())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=5,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    classes = list(model.named_steps["classifier"].classes_)
    denied_index = classes.index("denied")

    print("\nTraining records:", len(X_train))
    print("Test records:", len(X_test))
    print("\nTest-set evaluation")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 3),
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["approved", "denied"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["approved", "denied"],
    ))

    y_denied = (y_test == "denied").astype(int)

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_denied, probabilities[:, denied_index]), 3),
    )

    # Save only to the separate vehicle model path.
    joblib.dump(model, MODEL_PATH)

    print("\nVehicle model saved to:", MODEL_PATH)
    print(
        "Reminder: these metrics evaluate synthetic labels, "
        "not real-world insurance decisions."
    )


if __name__ == "__main__":
    main()
