import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "soil_condition_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.joblib"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

NUMERIC_FEATURES = [
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "ph",
    "electrical_conductivity_ds_m",
    "soil_temperature_c",
    "moisture_percent",
]

CATEGORICAL_FEATURES = ["crop_selected", "soil_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Ensure the API uses the same input schema as training.
if metadata.get("features") != FEATURES:
    raise ValueError(
        "Feature mismatch. Check train_model.py and model_metadata.joblib."
    )

# Set FRONTEND_ORIGIN to your Lovable website's origin(s), comma-separated.
# During local development, allow the Vite default origin.
frontend_origin = os.environ.get(
    "FRONTEND_ORIGIN", "http://localhost:5173"
)

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in frontend_origin.split(",")
    if origin.strip()
]

# Apply CORS to all routes (including / health check and /predict)
CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
)

# Provisional bounds for rejecting obviously invalid inputs.
NUMERIC_BOUNDS = {
    "nitrogen_mg_kg": (0, 10000),
    "phosphorus_mg_kg": (0, 10000),
    "potassium_mg_kg": (0, 10000),
    "ph": (0, 14),
    "electrical_conductivity_ds_m": (0, 1000),
    "soil_temperature_c": (-50, 100),
    "moisture_percent": (0, 100),
}


# --------------------------------------------------
# Agronomic Diagnostics Helpers
# --------------------------------------------------

def evaluate_nutrients(row):
    """
    Evaluates individual soil metrics against general agronomic thresholds.
    Labels supported by the frontend UI: "Optimal", "Good", "Low", "High", "Moderate", "Bad".
    """
    n = row["nitrogen_mg_kg"]
    p = row["phosphorus_mg_kg"]
    k = row["potassium_mg_kg"]
    ph = row["ph"]
    ec = row["electrical_conductivity_ds_m"]
    moisture = row["moisture_percent"]

    # Nitrogen (mg/kg or ppm)
    n_label = "Low" if n < 50 else ("Optimal" if n <= 140 else "High")

    # Phosphorus (mg/kg)
    p_label = "Low" if p < 25 else ("Optimal" if p <= 80 else "High")

    # Potassium (mg/kg)
    k_label = "Low" if k < 100 else ("Optimal" if k <= 250 else "High")

    # pH
    if 6.0 <= ph <= 7.5:
        ph_label = "Optimal"
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.2:
        ph_label = "Moderate"
    elif ph < 5.5:
        ph_label = "Low"
    else:
        ph_label = "High"

    # Electrical Conductivity (salinity in dS/m)
    if ec <= 1.5:
        ec_label = "Optimal"
    elif ec <= 2.5:
        ec_label = "Moderate"
    else:
        ec_label = "Bad"

    # Moisture (%)
    if 40 <= moisture <= 75:
        moisture_label = "Optimal"
    elif 25 <= moisture < 40:
        moisture_label = "Moderate"
    elif moisture < 25:
        moisture_label = "Low"
    else:
        moisture_label = "High"

    return [
        {"name": "Nitrogen", "value": round(n, 1), "label": n_label},
        {"name": "Phosphorus", "value": round(p, 1), "label": p_label},
        {"name": "Potassium", "value": round(k, 1), "label": k_label},
        {"name": "pH", "value": round(ph, 2), "label": ph_label},
        {"name": "Electrical conductivity", "value": round(ec, 2), "label": ec_label},
        {"name": "Moisture", "value": round(moisture, 1), "label": moisture_label},
    ]


def generate_recommendations(row, crop):
    """
    Builds tailored advisory messages according to nutrient deficiencies or soil stress.
    """
    n = row["nitrogen_mg_kg"]
    p = row["phosphorus_mg_kg"]
    k = row["potassium_mg_kg"]
    ph = row["ph"]
    ec = row["electrical_conductivity_ds_m"]
    moisture = row["moisture_percent"]

    recs = []

    # Salinity
    if ec > 2.5:
        recs.append(
            f"High salinity detected (EC {ec:.1f} dS/m): leach the field with clean water and improve sub-surface drainage before planting."
        )
    elif ec > 1.5:
        recs.append(
            "Mild salinity levels: monitor irrigation water salinity and avoid over-fertilizing with chloride-based salts."
        )

    # pH
    if ph < 5.5:
        recs.append(
            f"Soil is acidic (pH {ph:.1f}): apply agricultural lime or dolomite to raise soil pH toward the optimal range (6.0 - 7.0)."
        )
    elif ph > 8.0:
        recs.append(
            f"Soil is alkaline (pH {ph:.1f}): incorporate gypsum and organic matter (farmyard manure) to reduce alkalinity and increase micronutrient uptake."
        )

    # Nitrogen
    if n < 50:
        recs.append(
            f"Nitrogen is deficient ({n:.0f} mg/kg): apply urea, ammonium sulfate, or well-rotted compost in split doses for {crop}."
        )
    elif n > 140:
        recs.append(
            f"Excess nitrogen detected ({n:.0f} mg/kg): withhold synthetic nitrogen fertilizers to prevent vegetative overgrowth and lodging."
        )

    # Phosphorus
    if p < 25:
        recs.append(
            f"Phosphorus is low ({p:.0f} mg/kg): apply single super phosphate (SSP) or DAP as basal fertilizer to promote root establishment."
        )

    # Potassium
    if k < 100:
        recs.append(
            f"Potassium is low ({k:.0f} mg/kg): apply muriate of potash (MOP) to boost disease resistance and grain/fruit filling."
        )

    # Moisture
    if moisture < 30:
        recs.append(
            f"Soil moisture is low ({moisture:.0f}%): schedule an irrigation cycle to prevent moisture stress in {crop}."
        )
    elif moisture > 80:
        recs.append(
            f"Soil moisture is saturated ({moisture:.0f}%): inspect drainage channels to prevent waterlogging and root rot."
        )

    if not recs:
        recs.append(
            f"Soil conditions and nutrient levels are well balanced for growing {crop}."
        )

    return recs


# --------------------------------------------------
# Authentication
# --------------------------------------------------

def verify_firebase_token():
    if os.environ.get("AUTH_REQUIRED", "true").lower() == "false":
        app.logger.warning(
            "Authentication disabled. Local testing only."
        )
        return None

    from firebase_admin import auth, credentials, initialize_app
    import firebase_admin

    if not firebase_admin._apps:
        service_account_json = os.environ.get(
            "FIREBASE_SERVICE_ACCOUNT_JSON"
        )

        if not service_account_json:
            raise RuntimeError(
                "Firebase service account is not configured."
            )

        import json

        certificate = credentials.Certificate(
            json.loads(service_account_json)
        )
        initialize_app(certificate)

    header = request.headers.get("Authorization", "")

    if not header.startswith("Bearer "):
        return False

    token = header.split(" ", 1)[1].strip()

    if not token:
        return False

    try:
        return auth.verify_id_token(token)
    except Exception:
        return False


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "Soil Condition ML API is running",
        "model_version": metadata.get("version", "unknown"),
        "label_type": metadata.get(
            "label_type", "provisional"
        ),
    })


@app.post("/predict")
def predict():
    try:
        user = verify_firebase_token()
    except RuntimeError:
        app.logger.exception("Authentication configuration error")
        return jsonify({
            "error": "Server authentication is not configured."
        }), 500

    if user is False:
        return jsonify({
            "error": "A valid Firebase ID token is required."
        }), 401

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request must contain a JSON object."
        }), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "fields": missing,
        }), 400

    unexpected = [key for key in data if key not in FEATURES]

    if unexpected:
        return jsonify({
            "error": "Unexpected fields provided.",
            "fields": unexpected,
        }), 400

    row = {}

    try:
        for feature in NUMERIC_FEATURES:
            value = data[feature]

            if isinstance(value, bool):
                raise ValueError(f"{feature} must be numeric.")

            number = float(value)

            if not np.isfinite(number):
                raise ValueError(f"{feature} must be finite.")

            low, high = NUMERIC_BOUNDS[feature]

            if not low <= number <= high:
                raise ValueError(
                    f"{feature} must be between {low} and {high}."
                )

            row[feature] = number

        for feature in CATEGORICAL_FEATURES:
            value = data[feature]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{feature} must be a non-empty string."
                )

            row[feature] = value.strip()

        input_df = pd.DataFrame([row], columns=FEATURES)

        # Predict condition from your trained model
        prediction = str(model.predict(input_df)[0]).strip().title()

        valid_labels = {"Good", "Moderate", "Bad"}

        if prediction not in valid_labels:
            app.logger.error("Model returned an unexpected label")
            return jsonify({
                "error": "Model returned an invalid prediction."
            }), 500

        # Generate nutrient breakdown and agronomic recommendations
        nutrients = evaluate_nutrients(row)
        recommendations = generate_recommendations(row, row["crop_selected"])

        response = {
            "prediction": prediction,
            "model_version": metadata.get("version", "unknown"),
            "label_type": metadata.get(
                "label_type", "provisional"
            ),
            "status_note": (
                "Prototype prediction based on provisional "
                "dataset labels; not a validated soil diagnosis."
            ),
            "nutrients": nutrients,
            "recommendations": recommendations,
        }

        if isinstance(user, dict):
            response["request_user_uid"] = user.get("uid")

        return jsonify(response), 200

    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed due to a server error."
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "accuracy": 0.8079,
        "precision": 0.8081,
        "recall": 0.8079,
        "f1_score": 0.8078,
        "model_version": "3.0-prototype",
        "evaluated_on": "20% held-out test set"
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
