
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import predict_diabetes, FEATURES

app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for diabetes risk prediction",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_diabetes(request.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
