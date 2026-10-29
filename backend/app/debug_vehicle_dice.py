
import joblib
import pandas as pd
import dice_ml

from vehicle_explanations import (
    APP_DIR,
    FEATURES,
    FEATURES_TO_VARY,
    load_vehicle_reference_data,
    BinaryDecisionAdapter,
)

claim = {
    "vehicle_age": 1,
    "driver_age": 21,
    "vehicle_value": 1_000_000,
    "claim_amount": 300_000,
    "repair_cost": 250_000,
    "previous_claims": 0,
    "policy_duration_months": 24,
    "documents_complete": 0,
    "police_report_available": 0,
    "accident_severity": "minor",
}

reference_data = load_vehicle_reference_data()
query = pd.DataFrame([claim], columns=FEATURES)

for column in ["documents_complete", "police_report_available"]:
    query[column] = query[column].astype(int)

model = joblib.load(APP_DIR / "vehicle_claim_model.joblib")
adapter = BinaryDecisionAdapter(model)

original_class = int(adapter.predict(query)[0])
desired_class = 1 - original_class

print("Original prediction:", "denied" if original_class else "approved")
print("Desired prediction:", "denied" if desired_class else "approved")

dice_data = dice_ml.Data(
    dataframe=reference_data,
    continuous_features=FEATURES[:len(FEATURES) - 1],
    categorical_features=["accident_severity"],
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
    feature: [
        float(reference_data[feature].min()),
        float(reference_data[feature].max()),
    ]
    for feature in FEATURES_TO_VARY
}

result = explainer.generate_counterfactuals(
    query,
    total_CFs=3,
    desired_class=desired_class,
    features_to_vary=FEATURES_TO_VARY,
    permitted_range=permitted_range,
    verbose=False,
)

example = result.cf_examples_list[0]
print("\nRaw DiCE counterfactuals:")

if example.final_cfs_df is None:
    print("DiCE returned None.")
elif example.final_cfs_df.empty:
    print("DiCE returned an empty dataframe.")
else:
    print(example.final_cfs_df.to_string(index=False))

    candidates = example.final_cfs_df[FEATURES].copy()
    for column in ["documents_complete", "police_report_available"]:
        candidates[column] = candidates[column].astype(int)

    print("\nPredictions from the original model:")
    print(adapter.predict(candidates))
