
from pathlib import Path

import dice_ml
import joblib
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


def generate_counterfactuals(
    professional_fee: float,
    facility_fee: float,
    service_category: str,
    total_cfs: int = 3,
) -> dict:
    """Generate hypothetical counterfactuals for the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model file was not found.")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Claims dataset was not found.")

    if professional_fee < 0 or facility_fee < 0:
        raise ValueError("Fees cannot be negative.")

    if total_cfs < 1 or total_cfs > 5:
        raise ValueError("total_cfs must be between 1 and 5.")

    df = pd.read_csv(DATA_PATH)

    df = df[
        df["adjudication_status"].isin(["paid", "denied"])
    ].copy()

    df = df.dropna(subset=CATEGORICAL_FEATURES)

    # Match the numeric labels used by the trained classifier:
    # paid = 0, denied = 1.
    df["denied_flag"] = (
        df["adjudication_status"].eq("denied").astype(int)
    )

    training_data = df[FEATURES + ["denied_flag"]].copy()

    known_categories = set(
        training_data["service_category"].dropna().astype(str)
    )

    if service_category not in known_categories:
        raise ValueError(
            "Unknown service category. Choose a category present "
            "in the training dataset."
        )

    model = joblib.load(MODEL_PATH)

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

    query = pd.DataFrame([{
        "professional_fee": float(professional_fee),
        "facility_fee": float(facility_fee),
        "service_category": service_category,
    }])

    # Keep numerical values within observed training-data ranges.
    permitted_range = {
        feature: [
            float(training_data[feature].min()),
            float(training_data[feature].max()),
        ]
        for feature in NUMERIC_FEATURES
    }

    permitted_range["service_category"] = sorted(known_categories)

    result = explainer.generate_counterfactuals(
        query,
        total_CFs=total_cfs,
        desired_class="opposite",
        features_to_vary=FEATURES,
        permitted_range=permitted_range,
        verbose=False,
    )

    cf = result.cf_examples_list[0]

    if cf.final_cfs_df is None or cf.final_cfs_df.empty:
        return {
            "success": True,
            "message": (
                "DiCE could not find a counterfactual for this input. "
                "Try a different input within the observed data ranges."
            ),
            "counterfactuals": [],
        }

    counterfactuals = (
        cf.final_cfs_df[FEATURES + ["denied_flag"]]
        .to_dict(orient="records")
    )

    for row in counterfactuals:
        row["predicted_decision"] = (
            "denied" if int(row.pop("denied_flag")) == 1 else "paid"
        )

    original_prediction = int(model.predict(query)[0])

    return {
        "success": True,
        "original_prediction": (
            "denied" if original_prediction == 1 else "paid"
        ),
        "counterfactuals": counterfactuals,
        "disclaimer": (
            "These are hypothetical changes that affect this model's "
            "prediction. They are not insurance rules or guarantees "
            "of an actual claim outcome."
        ),
    }
