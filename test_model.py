
import joblib
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

model = joblib.load(
    base / "artifacts" / "soil_condition_model.joblib"
)

readings = pd.DataFrame([{
    "nitrogen_mg_kg": 157,
    "phosphorus_mg_kg": 64,
    "potassium_mg_kg": 197,
    "ph": 5.8,
    "electrical_conductivity_ds_m": 1.73,
    "soil_temperature_c": 26.9,
    "moisture_percent": 56.1,
    "crop_selected": "Banana",
    "soil_type": "Loamy",
}])

print("Prediction:", model.predict(readings)[0])

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(readings)[0]
    print("\nClass probabilities:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")
