from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import yaml
import os
import joblib
import pathlib as pl
from ml_serving.config import LOGS_DIR, MODEL_DIR
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(
        MODEL_DIR / "model.pkl"
    )
    yield


app = FastAPI(lifespan=lifespan)


class ModelPredictRequest(BaseModel):
    data: list[float] | list[int]


@app.post("/predict")
async def predict(model_input: ModelPredictRequest, request: Request):
    try:
        model = request.app.state.model
        predictions = model.predict(
            np.array(model_input.data).reshape(-1, 1)
        )
        return {"predictions": predictions.ravel().tolist()}
    except Exception as e:
        return {"error": str(e)}
