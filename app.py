from contextlib import asynccontextmanager
import os

import numpy as np
import xgboost as xgb
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model = None

FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = os.getenv("MODEL_PATH", "./model")
    model = mlflow.xgboost.load_model(model_path)
    print("Model loaded successfully!")
    yield


app = FastAPI(title="California Housing API", lifespan=lifespan)


class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(input_data: HousingInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = input_data.model_dump()
        row = np.array([[data[f] for f in FEATURE_ORDER]], dtype=float)
        raw_prediction = float(model.predict(xgb.DMatrix(row))[0])
        actual_price = raw_prediction * 100000.0
        return {"prediction": f"${actual_price:,.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))