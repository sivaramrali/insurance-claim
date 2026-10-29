
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "health-claims" / "claims.csv"
MODEL_PATH = Path(__file__).resolve().parent / "claim_model.joblib"

TARGET = "adjudication_status"

NUMERIC_FEATURES = [
    "professional_fee",
    "facility_fee",
]

CATEGORICAL_FEATURES = [
    "service_category",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only finalized binary outcomes.
    df = df[df[TARGET].isin(["paid", "denied"])].copy()

    # Exclude records without the basic inputs needed here.
    df = df.dropna(subset=CATEGORICAL_FEATURES)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "The dataset must contain both paid and denied claims."
        )

    if df[TARGET].value_counts().min() < 5:
        raise ValueError(
            "There are too few examples in one class for a "
            "stratified demonstration split."
        )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].map({"denied": 1, "paid": 0})

    print("Records used:", len(df))
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

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

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nEvaluation results")
    print("------------------")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["paid", "denied"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    if y_test.nunique() == 2:
        print(
            "\nROC-AUC:",
            round(roc_auc_score(y_test, probabilities), 4),
        )

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    print(
        "\nNote: This small synthetic dataset is for a project "
        "demonstration, not real-world insurance decisions."
    )


if __name__ == "__main__":
    main()
