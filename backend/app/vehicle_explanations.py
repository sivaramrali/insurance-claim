
from pathlib import Path

import dice_ml
import joblib
import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "vehicle-claims"
    / "synthetic_vehicle_insurance_claims.csv"
)

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

BINARY_FEATURES = [
    "documents_complete",
    "police_report_available",
]

FEATURES_TO_VARY = [
    "claim_amount",
    "repair_cost",
]


class BinaryDecisionAdapter:
    """Map approved/denied model predictions to numeric classes."""

    def __init__(self, fitted_model):
        self.model = fitted_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probabilities = self.model.predict_proba(X)
        model_classes = list(self.model.classes_)

        approved_index = model_classes.index("approved")
        denied_index = model_classes.index("denied")

        return probabilities[:, [approved_index, denied_index]]

    def predict(self, X):
        predictions = self.model.predict(X)

        return np.asarray(
            [
                1 if value == "denied" else 0
                for value in predictions
            ],
            dtype=int,
        )


def load_vehicle_reference_data():
    """Load and validate vehicle-claim reference data for DiCE."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Vehicle dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [
        column
        for column in FEATURES + ["claim_status"]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Vehicle dataset is missing columns: {missing}"
        )

    df = df[
        df["claim_status"].isin(["approved", "denied"])
    ].copy()

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=FEATURES)

    if df.empty:
        raise ValueError(
            "No complete approved/denied vehicle claims were found."
        )

    if df["claim_status"].nunique() != 2:
        raise ValueError(
            "Reference data must contain approved and denied claims."
        )

    for column in BINARY_FEATURES:
        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"{column} must contain only 0 or 1."
            )

        df[column] = df[column].astype(int)

    df["accident_severity"] = (
        df["accident_severity"].astype(str).str.strip()
    )

    if (df["accident_severity"] == "").any():
        raise ValueError(
            "The dataset contains empty accident severity values."
        )

    df["denied_flag"] = (
        df["claim_status"].eq("denied").astype(int)
    )

    return df[FEATURES + ["denied_flag"]].copy()


def generate_vehicle_counterfactuals(
    claim: dict,
    total_cfs: int = 3,
) -> dict:
    """Generate hypothetical vehicle-claim counterfactuals."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Vehicle model not found: {MODEL_PATH}"
        )

    if (
        not isinstance(total_cfs, int)
        or isinstance(total_cfs, bool)
        or not 1 <= total_cfs <= 5
    ):
        raise ValueError(
            "total_cfs must be an integer from 1 to 5."
        )

    if not isinstance(claim, dict):
        raise ValueError("claim must be a dictionary.")

    reference_data = load_vehicle_reference_data()

    missing = [
        feature
        for feature in FEATURES
        if feature not in claim
    ]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    query_values = {}

    for feature in NUMERIC_FEATURES:
        try:
            value = float(claim[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid number."
            ) from None

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        if value < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

        query_values[feature] = value

    for column in BINARY_FEATURES:
        if query_values[column] not in (0.0, 1.0):
            raise ValueError(
                f"{column} must be either 0 or 1."
            )

        query_values[column] = int(query_values[column])

    severity_input = str(
        claim["accident_severity"]
    ).strip().lower()

    severity_lookup = {
        value.lower(): value
        for value in reference_data["accident_severity"].unique()
    }

    if severity_input not in severity_lookup:
        raise ValueError(
            "Unknown accident severity. Available values: "
            + ", ".join(sorted(severity_lookup.values()))
        )

    # Preserve the category spelling used in the dataset.
    query_values["accident_severity"] = (
        severity_lookup[severity_input]
    )

    query = pd.DataFrame(
        [query_values],
        columns=FEATURES,
    )

    for column in BINARY_FEATURES:
        query[column] = query[column].astype(int)

    fitted_model = joblib.load(MODEL_PATH)
    adapter = BinaryDecisionAdapter(fitted_model)

    original_class = int(adapter.predict(query)[0])
    desired_class = 1 - original_class

    dice_data = dice_ml.Data(
        dataframe=reference_data,
        continuous_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        outcome_name="denied_flag",
    )

    dice_model = dice_ml.Model(
        model=adapter,
        backend="sklearn",
        model_type="classifier",
    )

    explainer = dice_ml.Dice(
        dice_data,
        dice_model,
        method="random",
    )

    permitted_range = {
        "claim_amount": [
            0.0,
            min(
                float(reference_data["claim_amount"].max()),
                float(query_values["vehicle_value"]),
            ),
        ],
        "repair_cost": [
            0.0,
            min(
                float(reference_data["repair_cost"].max()),
                float(query_values["claim_amount"]),
                float(query_values["vehicle_value"]),
            ),
        ],
    }

    # DiCE can raise this exception when it finds no alternatives.
    try:
        result = explainer.generate_counterfactuals(
            query,
            total_CFs=total_cfs,
            desired_class=desired_class,
            features_to_vary=FEATURES_TO_VARY,
            permitted_range=permitted_range,
            verbose=False,
        )
    except UserConfigValidationException:
        result = None

    cf_example = (
        result.cf_examples_list[0]
        if result is not None
        else None
    )

    valid_counterfactuals = []

    if (
        cf_example is not None
        and cf_example.final_cfs_df is not None
        and not cf_example.final_cfs_df.empty
    ):
        for _, row in cf_example.final_cfs_df.iterrows():
            candidate = query.iloc[0].to_dict()

            try:
                candidate["claim_amount"] = float(
                    row["claim_amount"]
                )
                candidate["repair_cost"] = float(
                    row["repair_cost"]
                )
            except (KeyError, TypeError, ValueError):
                continue

            amount = candidate["claim_amount"]
            repair = candidate["repair_cost"]

            if not np.isfinite(amount) or not np.isfinite(repair):
                continue

            if amount < 0 or repair < 0:
                continue

            # Enforce the amount constraints.
            if repair > amount:
                continue

            if amount > candidate["vehicle_value"]:
                continue

            candidate_df = pd.DataFrame(
                [candidate],
                columns=FEATURES,
            )

            for column in BINARY_FEATURES:
                candidate_df[column] = candidate_df[
                    column
                ].astype(int)

            # Verify the actual trained model's prediction.
            predicted_class = int(
                adapter.predict(candidate_df)[0]
            )

            if predicted_class != desired_class:
                continue

            duplicate = any(
                abs(
                    existing["claim_amount"] - amount
                ) < 0.01
                and abs(
                    existing["repair_cost"] - repair
                ) < 0.01
                for existing in valid_counterfactuals
            )

            if duplicate:
                continue

            valid_counterfactuals.append(
                {
                    "claim_amount": round(amount, 2),
                    "repair_cost": round(repair, 2),
                    "predicted_decision": (
                        "denied"
                        if predicted_class == 1
                        else "approved"
                    ),
                }
            )

            if len(valid_counterfactuals) >= total_cfs:
                break

    response = {
        "success": True,
        "original_prediction": (
            "denied"
            if original_class == 1
            else "approved"
        ),
        "counterfactuals": valid_counterfactuals,
        "disclaimer": (
            "This demonstration uses synthetic vehicle-claim data. "
            "Counterfactuals are hypothetical input changes, not proof "
            "of causation, actual insurer rules, or guarantees of a "
            "claim outcome. Claim and repair amounts may not be "
            "controllable by a claimant. The model has limited ability "
            "to identify denied claims and is not validated for real "
            "insurance decisions."
        ),
    }

    if not valid_counterfactuals:
        response["message"] = (
            "No valid counterfactual was found for this input. "
            "The model may not find an opposite prediction within "
            "the permitted ranges and consistency constraints."
        )

    return response
