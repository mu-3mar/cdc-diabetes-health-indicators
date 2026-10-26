
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"

# Load the saved artifact once when this module is imported
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
THRESHOLD = artifact["threshold"]
FEATURES = artifact["features"]


def predict_diabetes(input_data: dict) -> dict:
    missing_features = set(FEATURES) - set(input_data)

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    input_df = pd.DataFrame(
        [{feature: input_data[feature] for feature in FEATURES}]
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": THRESHOLD,
    }
