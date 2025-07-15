from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import joblib
from ml_serving.config import MODEL_DIR
import numpy as np
from .utils import setup_logging
import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.model = joblib.load(MODEL_DIR / "model.pkl")
    app.state.logger = logging.getLogger(__name__)
    yield


app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    app.state.logger.warning(f"Validation failed for request: {await request.body()}")
    app.state.logger.warning(f"Validation errors: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)

class ModelPredictRequest(BaseModel):
    data: list[float] | list[int]

    @field_validator("data")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("Input data must not be empty.")
        return v


@app.post("/predict")
async def predict(model_input: ModelPredictRequest, request: Request):
    try:
        model = request.app.state.model
        predictions = model.predict(np.array(model_input.data).reshape(-1, 1))
        app.state.logger.info(f"Input: {model_input}, Output: {predictions}")
        return {"predictions": predictions.ravel().tolist()}
    except Exception as e:
        app.state.logger.info(
            f"Predict failed for input: {model_input}. "
            f"Error: {str(e)}"
            )
        return {"error": str(e)}
