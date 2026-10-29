
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "health-claims" / "claims.csv"

NUMERIC = ["professional_fee", "facility_fee"]
CATEGORICAL = ["service_category"]
TARGET = "adjudication_status"


def main():
    df = pd.read_csv(DATA)
    df = df[df[TARGET].isin(["paid", "denied"])].copy()
    df = df.dropna(subset=CATEGORICAL)

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].map({"paid": 0, "denied": 1})

    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Cross-validated model results")
    print("-----------------------------")

    predictions = cross_val_predict(model, X, y, cv=cv)
    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    print("Balanced accuracy:", round(
        balanced_accuracy_score(y, predictions), 3
    ))
    print("ROC-AUC:", round(roc_auc_score(y, probabilities), 3))
    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["paid", "denied"],
        zero_division=0,
    ))

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X, y)
    baseline_predictions = baseline.predict(X)

    print("Majority-class baseline accuracy:", round(
        accuracy_score(y, baseline_predictions), 3
    ))


if __name__ == "__main__":
    main()
