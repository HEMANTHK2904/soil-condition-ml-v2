
from pathlib import Path

import joblib
import pandas as pd

base = Path(__file__).resolve().parent

model = joblib.load(
    base / "artifacts" / "soil_condition_model.joblib"
)
df = pd.read_csv(base / "soil_dataset.csv")

# The exact readings from your Postman test
readings = {
    "nitrogen_mg_kg": 157,
    "phosphorus_mg_kg": 64,
    "potassium_mg_kg": 197,
    "ph": 5.8,
    "electrical_conductivity_ds_m": 1.73,
    "soil_temperature_c": 26.9,
    "moisture_percent": 56.1,
    "crop_selected": "Banana",
    "soil_type": "Loamy",
}

X = pd.DataFrame([readings])
prediction = model.predict(X)[0]

print("\n========== MODEL PREDICTION ==========")
print("Prediction:", prediction)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[0]
    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")

print("\n========== DATASET LABELS ==========")

# Show how labels are distributed for Banana
banana = df[
    df["crop_selected"].astype(str).str.strip().str.lower()
    == "banana"
]

print("Banana rows:", len(banana))

if len(banana):
    print("\nLabels for Banana:")
    print(banana["soil_condition"].value_counts())

    if "soil_type" in banana.columns:
        loamy = banana[
            banana["soil_type"].astype(str).str.strip().str.lower()
            == "loamy"
        ]
        print("\nBanana + Loamy rows:", len(loamy))
        if len(loamy):
            print(loamy["soil_condition"].value_counts())

print("\n========== NEAREST TRAINING EXAMPLES ==========")

numeric = [
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "ph",
    "electrical_conductivity_ds_m",
    "soil_temperature_c",
    "moisture_percent",
]

# Compare rows on numeric features after scaling by dataset spread.
# This is diagnostic only, not an agronomic suitability score.
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

subset = df.dropna(subset=numeric + ["soil_condition"]).copy()

if len(subset):
    medians = subset[numeric].median()
    scales = subset[numeric].quantile(0.75) - subset[numeric].quantile(0.25)
    scales = scales.mask(scales == 0, 1)

    distances = (
        (subset[numeric] - pd.Series(readings)[numeric]) / scales
    ).pow(2).sum(axis=1).pow(0.5)

    subset["distance"] = distances

    print(
        subset.sort_values("distance")[
            ["crop_selected", "soil_type"] + numeric
            + ["soil_condition", "distance"]
        ].head(10).to_string(index=False)
    )
else:
    print("No complete numeric rows available for comparison.")
