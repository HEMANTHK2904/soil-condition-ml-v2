
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "soil_dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

# These feature names MUST match app.py.
NUMERIC_FEATURES = [
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "ph",
    "electrical_conductivity_ds_m",
    "soil_temperature_c",
    "moisture_percent",
]

CATEGORICAL_FEATURES = [
    "crop_selected",
    "soil_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "soil_condition"

EXPECTED_LABELS = {"Good", "Moderate", "Bad"}
LABEL_ORDER = ["Bad", "Good", "Moderate"]

# ============================================================
# 2. LOAD AND MAP THE NEW DATASET
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place the new CSV in the project folder as soil_dataset.csv."
    )

raw_df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", raw_df.shape)
print("Original columns:", raw_df.columns.tolist())

# Map the new dataset's columns to the existing API schema.
COLUMN_MAP = {
    "Crop": "crop_selected",
    "Soil_Type": "soil_type",
    "Nitrogen": "nitrogen_mg_kg",
    "Phosphorus": "phosphorus_mg_kg",
    "Potassium": "potassium_mg_kg",
    "pH": "ph",
    "EC": "electrical_conductivity_ds_m",
    "Temperature": "soil_temperature_c",
    "Moisture": "moisture_percent",
    "Soil_Condition": "soil_condition",
}

missing_columns = [
    column for column in COLUMN_MAP
    if column not in raw_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected dataset columns: {missing_columns}\n"
        f"Found columns: {raw_df.columns.tolist()}"
    )

df = raw_df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()

# ============================================================
# 3. CLEAN AND VALIDATE DATA
# ============================================================

for column in NUMERIC_FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

for column in CATEGORICAL_FEATURES + [TARGET]:
    df[column] = df[column].astype("string").str.strip()

df = df.replace([np.inf, -np.inf], np.nan)

missing_counts = df[FEATURES + [TARGET]].isna().sum()
if missing_counts.any():
    print("\nMissing values:")
    print(missing_counts[missing_counts > 0])
    raise ValueError(
        "Missing or invalid values found. Correct the CSV before training."
    )

for column in CATEGORICAL_FEATURES:
    if (df[column] == "").any():
        raise ValueError(f"Empty values found in {column}.")

df[TARGET] = df[TARGET].str.title()

found_labels = set(df[TARGET].unique())
if found_labels != EXPECTED_LABELS:
    raise ValueError(
        f"Expected labels {EXPECTED_LABELS}, found {found_labels}"
    )

# Basic physical/input sanity checks.
# These are broad data validation limits, not crop suitability thresholds.
bounds = {
    "nitrogen_mg_kg": (0, 10000),
    "phosphorus_mg_kg": (0, 10000),
    "potassium_mg_kg": (0, 10000),
    "ph": (0, 14),
    "electrical_conductivity_ds_m": (0, 1000),
    "soil_temperature_c": (-50, 100),
    "moisture_percent": (0, 100),
}

for column, (low, high) in bounds.items():
    invalid = ~df[column].between(low, high)

    if invalid.any():
        raise ValueError(
            f"{column} has {invalid.sum()} values outside "
            f"the allowed input range {low} to {high}."
        )

# Check for exact duplicate records.
duplicate_count = int(df.duplicated().sum())
print("Exact duplicate rows:", duplicate_count)

if duplicate_count:
    print(
        "Warning: duplicates exist. If these are repeated or synthetic "
        "records, random splitting may overestimate generalization."
    )

X = df[FEATURES]
y = df[TARGET]

print("\nCleaned dataset shape:", df.shape)
print("\nOverall label distribution:")
print(y.value_counts().reindex(LABEL_ORDER, fill_value=0))

print("\nBanana label distribution:")
banana_mask = df["crop_selected"].str.casefold() == "banana"
banana_df = df.loc[banana_mask]

print("Banana rows:", len(banana_df))
print(
    banana_df[TARGET].value_counts().reindex(
        LABEL_ORDER, fill_value=0
    )
)

if banana_df.empty:
    raise ValueError("No Banana records were found in the dataset.")

# ============================================================
# 4. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))

# ============================================================
# 5. PREPROCESSING
# ============================================================

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numbers",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

# ============================================================
# 6. COMPARE MODELS USING TRAINING DATA ONLY
# ============================================================

candidates = {
    "Dummy baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
}

minimum_class_count = int(y_train.value_counts().min())
n_splits = min(5, minimum_class_count)

if n_splits < 2:
    raise ValueError(
        "Not enough training records per class for cross-validation."
    )

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42,
)

cv_results = {}

print("\n========== CROSS-VALIDATION ==========")

for name, classifier in candidates.items():
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise",
    )

    cv_results[name] = scores.mean()

    print(
        f"{name}: macro F1 = {scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

# Exclude the dummy baseline from final model selection.
eligible_models = {
    name: score
    for name, score in cv_results.items()
    if name != "Dummy baseline"
}

best_name = max(eligible_models, key=eligible_models.get)
print("\nSelected model:", best_name)

# ============================================================
# 7. TRAIN THE SELECTED MODEL
# ============================================================

classifiers = {
    "Random Forest": candidates["Random Forest"],
    "Extra Trees": candidates["Extra Trees"],
}

final_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", classifiers[best_name]),
])

final_model.fit(X_train, y_train)

# ============================================================
# 8. EVALUATE ON THE HELD-OUT TEST SET
# ============================================================

predictions = final_model.predict(X_test)

print("\n========== HELD-OUT TEST RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro"), 4),
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=LABEL_ORDER,
    )
)

# ============================================================
# 9. TEST THE EXACT BANANA READINGS
# ============================================================

banana_test = pd.DataFrame([{
    "nitrogen_mg_kg": 157,
    "phosphorus_mg_kg": 64,
    "potassium_mg_kg": 197,
    "ph": 5.8,
    "electrical_conductivity_ds_m": 1.73,
    "soil_temperature_c": 26.9,
    "moisture_percent": 56.1,
    "crop_selected": "Banana",
    "soil_type": "Loamy",
}])[FEATURES]

banana_prediction = final_model.predict(banana_test)[0]

print("\n========== BANANA TEST ==========")
print("Input:", banana_test.to_dict(orient="records")[0])
print("Predicted condition:", banana_prediction)

if hasattr(final_model, "predict_proba"):
    probabilities = final_model.predict_proba(banana_test)[0]

    print("Predicted class probabilities:")
    for label, probability in zip(
        final_model.classes_, probabilities
    ):
        print(f"{label}: {probability:.2%}")

# ============================================================
# 10. SAVE MODEL AND METADATA
# ============================================================

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "soil_condition_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.joblib"

joblib.dump(final_model, MODEL_PATH)

metadata = {
    "model_name": best_name,
    "version": "3.0-prototype",
    "features": FEATURES,
    "classes": LABEL_ORDER,
    "label_type": "provisional",
    "dataset_rows": int(len(df)),
    "dataset_columns": list(df.columns),
    "cv_macro_f1": {
        name: float(score)
        for name, score in cv_results.items()
    },
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, predictions)
    ),
    "test_macro_f1": float(
        f1_score(y_test, predictions, average="macro")
    ),
}

joblib.dump(metadata, METADATA_PATH)

print("\n========== SAVED FILES ==========")
print("Model:", MODEL_PATH)
print("Metadata:", METADATA_PATH)
print("\nTraining completed successfully.")
print(
    "Reminder: metrics measure agreement with the dataset labels, "
    "not validated real-world soil suitability."
)
