
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "prepared"
MODEL_DIR = PROJECT_ROOT / "models"

# Configuration
NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
THRESHOLD = 0.6

# Load training data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ],
    remainder="passthrough",
)

# Build pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=1,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

# Train
model_pipeline.fit(X_train, y_train)

# Save model and threshold
MODEL_DIR.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model_pipeline,
    "threshold": THRESHOLD,
    "features": X_train.columns.tolist(),
}

model_path = MODEL_DIR / "diabetes_model.joblib"
joblib.dump(artifact, model_path)

print(f"Training completed.")
print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
print(f"Threshold: {THRESHOLD}")
print(f"Model saved to: {model_path}")
