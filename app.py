
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

# Set FRONTEND_ORIGIN to your Lovable website's origin.
# During local development, allow the Vite default origin.
frontend_origin = os.environ.get(
    "FRONTEND_ORIGIN", "http://localhost:5173"
)

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in frontend_origin.split(",")
    if origin.strip()
]

CORS(
    app,
    resources={r"/predict": {"origins": allowed_origins}},
)

# Provisional bounds for rejecting obviously invalid inputs.
# These are not agronomic suitability thresholds.
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
# Authentication
# --------------------------------------------------
# Production requests must be authenticated with a
# Firebase ID token. Local testing can temporarily
# disable this guard using AUTH_REQUIRED=false.

def verify_firebase_token():
    if os.environ.get("AUTH_REQUIRED", "true").lower() == "false":
        app.logger.warning(
            "Authentication disabled. Local testing only."
        )
        return None

    from firebase_admin import auth, credentials, initialize_app

    import firebase_admin

    if not firebase_admin._apps:
        # Configure this secret in Render, never in frontend code.
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

    # Reject unexpected fields to catch frontend/schema mistakes.
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

        # Predict condition
        prediction = str(model.predict(input_df)[0]).strip().title()

        valid_labels = {"Good", "Moderate", "Bad"}

        if prediction not in valid_labels:
            app.logger.error("Model returned an unexpected label")
            return jsonify({
                "error": "Model returned an invalid prediction."
            }), 500

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
        }

        # Return the authenticated user's UID for server-side
        # context only; this does not itself save any Firestore data.
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


if __name__ == "__main__":
    # Local development only. Render uses Gunicorn.
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
