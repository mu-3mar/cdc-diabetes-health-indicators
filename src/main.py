from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_diabetes, FEATURES

app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for diabetes risk prediction",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_diabetes(request.features)
        return result
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
