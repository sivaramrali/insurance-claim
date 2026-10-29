
from pathlib import Path

import dice_ml
import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "health-claims" / "claims.csv"
MODEL_PATH = APP_DIR / "claim_model.joblib"

NUMERIC_FEATURES = [
    "professional_fee",
    "facility_fee",
]

CATEGORICAL_FEATURES = [
    "service_category",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "adjudication_status"


def load_reference_data():
    """Load eligible claims and prepare complete reference data for DiCE."""

    if not DATA_PATH.exists():
        raise FileNotFoundError("Claims dataset was not found.")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[df[TARGET].isin(["paid", "denied"])].copy()
    df = df.dropna(subset=CATEGORICAL_FEATURES)

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.empty:
        raise ValueError("No eligible paid or denied claims were found.")

    # Use reference-data medians only to prepare DiCE's reference table.
    # The loaded model retains its own fitted preprocessing pipeline.
    medians = df[NUMERIC_FEATURES].median()

    if medians.isna().any():
        raise ValueError(
            "At least one fee column has no usable numeric values."
        )

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(medians)

    df["denied_flag"] = (
        df[TARGET].eq("denied").astype(int)
    )

    training_data = df[FEATURES + ["denied_flag"]].copy()

    known_categories = sorted(
        training_data["service_category"].astype(str).unique().tolist()
    )

    return training_data, known_categories


def generate_counterfactuals(
    professional_fee: float,
    facility_fee: float,
    service_category: str,
    total_cfs: int = 3,
) -> dict:
    """Generate and validate hypothetical fee-based counterfactuals."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model file was not found.")

    if total_cfs < 1 or total_cfs > 5:
        raise ValueError("total_cfs must be between 1 and 5.")

    try:
        professional_fee = float(professional_fee)
        facility_fee = float(facility_fee)
    except (TypeError, ValueError):
        raise ValueError("Fees must be valid numeric values.") from None

    if not np.isfinite(professional_fee) or not np.isfinite(facility_fee):
        raise ValueError("Fees must be finite numeric values.")

    if professional_fee < 0 or facility_fee < 0:
        raise ValueError("Fees cannot be negative.")

    training_data, known_categories = load_reference_data()

    if service_category not in known_categories:
        raise ValueError(
            "Unknown service category. Choose a category present "
            "in the training dataset."
        )

    model = joblib.load(MODEL_PATH)

    query = pd.DataFrame([{
        "professional_fee": professional_fee,
        "facility_fee": facility_fee,
        "service_category": service_category,
    }], columns=FEATURES)

    original_prediction = int(model.predict(query)[0])
    desired_prediction = 1 - original_prediction

    data_interface = dice_ml.Data(
        dataframe=training_data,
        continuous_features=NUMERIC_FEATURES,
        outcome_name="denied_flag",
    )

    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",
    )

    explainer = dice_ml.Dice(
        data_interface,
        dice_model,
        method="random",
    )

    # Keep proposed fee changes within the observed reference-data range.
    permitted_range = {
        feature: [
            float(training_data[feature].min()),
            float(training_data[feature].max()),
        ]
        for feature in NUMERIC_FEATURES
    }

    result = explainer.generate_counterfactuals(
        query,
        total_CFs=total_cfs,
        desired_class=desired_prediction,
        features_to_vary=NUMERIC_FEATURES,
        permitted_range=permitted_range,
        verbose=False,
    )

    cf = result.cf_examples_list[0]
    valid_counterfactuals = []

    if cf.final_cfs_df is not None and not cf.final_cfs_df.empty:
        for _, row in cf.final_cfs_df.iterrows():
            try:
                candidate = {
                    "professional_fee": float(row["professional_fee"]),
                    "facility_fee": float(row["facility_fee"]),
                    # Service category is deliberately kept unchanged.
                    "service_category": service_category,
                }
            except (KeyError, TypeError, ValueError):
                continue

            fee_values = [
                candidate["professional_fee"],
                candidate["facility_fee"],
            ]

            if not all(np.isfinite(value) for value in fee_values):
                continue

            if any(value < 0 for value in fee_values):
                continue

            candidate_df = pd.DataFrame([candidate], columns=FEATURES)

            # Verify the actual loaded model predicts the desired class.
            candidate_prediction = int(model.predict(candidate_df)[0])

            if candidate_prediction != desired_prediction:
                continue

            # Avoid duplicate counterfactuals.
            if any(
                existing["professional_fee"] == candidate["professional_fee"]
                and existing["facility_fee"] == candidate["facility_fee"]
                for existing in valid_counterfactuals
            ):
                continue

            candidate["predicted_decision"] = (
                "denied" if candidate_prediction == 1 else "paid"
            )

            valid_counterfactuals.append(candidate)

            if len(valid_counterfactuals) >= total_cfs:
                break

    response = {
        "success": True,
        "original_prediction": (
            "denied" if original_prediction == 1 else "paid"
        ),
        "counterfactuals": valid_counterfactuals,
        "disclaimer": (
            "These are hypothetical fee changes that alter this model's "
            "prediction while keeping the service category fixed. They "
            "are not insurance rules, proof of causation, or guarantees "
            "of an actual claim outcome. The model has shown weak "
            "predictive performance on the available synthetic data."
        ),
    }

    if not valid_counterfactuals:
        response["message"] = (
            "No valid counterfactuals were found for this input. "
            "The model may not identify a fee change within the observed "
            "data ranges that reverses its prediction."
        )

    return response
