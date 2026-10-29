
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "health-claims" / "claims.csv"
MODEL_PATH = APP_DIR / "claim_model.joblib"

NUMERIC_FEATURES = ["professional_fee", "facility_fee"]
CATEGORICAL_FEATURES = ["service_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "adjudication_status"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep finalized paid and denied claims.
    df = df[df[TARGET].isin(["paid", "denied"])].copy()

    # A service category is required for this model.
    df = df.dropna(subset=CATEGORICAL_FEATURES)

    # Preserve missing fees for the pipeline's imputer.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    X = df[FEATURES].copy()
    y = df[TARGET].map({"paid": 0, "denied": 1})

    if y.isna().any():
        raise ValueError("Unexpected target labels found.")

    if y.nunique() != 2:
        raise ValueError("Both paid and denied claims are required.")

    return X, y


def build_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Training records:", len(X_train))
    print("Test records:", len(X_test))
    print("\nTest-set evaluation")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(
        "Balanced accuracy:",
        f"{balanced_accuracy_score(y_test, predictions):.3f}",
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["paid", "denied"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
