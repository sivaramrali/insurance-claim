
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "health-claims" / "claims.csv"

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

    # Evaluate finalized paid and denied claims only.
    df = df[df[TARGET].isin(["paid", "denied"])].copy()

    # Keep rows with a known service category.
    df = df.dropna(subset=["service_category"])

    # Convert fees to numeric values.
    # Missing or invalid values become NaN and are imputed
    # inside each cross-validation training fold.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[TARGET].nunique() != 2:
        raise ValueError("Both paid and denied records are required.")

    counts = df[TARGET].value_counts()

    if counts.min() < 5:
        raise ValueError(
            "At least five records per class are required "
            "for five-fold cross-validation."
        )

    X = df[FEATURES].copy()
    y = df[TARGET].map({"paid": 0, "denied": 1})

    print("Dataset:", DATA_PATH)
    print("Original CSV rows:", len(pd.read_csv(DATA_PATH)))
    print("Eligible records:", len(df))

    print("\nClass distribution:")
    print(df[TARGET].value_counts().to_string())

    print("\nMissing feature values:")
    print(X.isna().sum().to_string())

    print("\nLabel mapping: paid=0, denied=1")

    return X, y


def make_pipeline(classifier):
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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate_model(name, model, X, y, cv):
    # Out-of-fold predictions: each record is evaluated by a model
    # that was not trained on that record.
    predictions = cross_val_predict(
        model, X, y, cv=cv, method="predict"
    )

    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    print(f"Accuracy:          {accuracy_score(y, predictions):.3f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.3f}"
    )
    print(
        f"Denied precision:  "
        f"{precision_score(y, predictions, zero_division=0):.3f}"
    )
    print(
        f"Denied recall:     "
        f"{recall_score(y, predictions, zero_division=0):.3f}"
    )
    print(
        f"Denied F1:         "
        f"{f1_score(y, predictions, zero_division=0):.3f}"
    )
    print(
        f"Average precision: "
        f"{average_precision_score(y, probabilities):.3f}"
    )
    print(f"ROC-AUC:           {roc_auc_score(y, probabilities):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["paid", "denied"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y, predictions, labels=[0, 1]))


def main():
    X, y = load_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = [
        (
            "Majority-class baseline",
            make_pipeline(
                DummyClassifier(strategy="most_frequent")
            ),
        ),
        (
            "Logistic Regression",
            make_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )
            ),
        ),
        (
            "Random Forest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
    ]

    print("\nFive-fold stratified cross-validation")
    print("Results are exploratory estimates on synthetic data.")

    for name, model in models:
        evaluate_model(name, model, X, y, cv)

    print(
        "\nInterpretation:"
        "\n- Compare models against the majority-class baseline."
        "\n- Examine denied-class precision, recall and F1."
        "\n- Examine balanced accuracy and ROC-AUC."
        "\n- Do not judge performance using accuracy alone."
        "\n- These results do not establish real-world insurance accuracy."
    )


if __name__ == "__main__":
    main()
