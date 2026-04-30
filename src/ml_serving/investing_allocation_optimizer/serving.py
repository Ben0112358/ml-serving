from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import cloudpickle
from ml_serving.config import MODEL_DIR, ENV_VAR_OUTPUT_SUFFIX
import numpy as np
from ml_serving.utils import setup_logging
import ml_training
import logging
from typing import Callable, Union, Optional


# ---- STATIC MODEL FOR SWAGGER (schema visible) ---- #
class PredictArgs(BaseModel):
    metric: str = Field("mean", description="Metric name (internally uses np.mean)")
    p_1_constraint: Optional[float] = Field(None, description="Optional p1 constraint")
    p_5_constraint: Optional[float] = Field(None, description="Optional p5 constraint")
    max_std: Optional[float] = Field(None, description="Optional max std value")
    n_trials: int = Field(100, description="Number of trials")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    bootstrap_block_size: Union[int, str] = Field(
        "cube root", description="Bootstrap block size (integer or 'cube root')"
    )
    bootstrap_path_length: int = Field(100, description="Length of bootstrap path")
    n_bootstrap_paths: int = Field(1000, description="Number of bootstrap paths")


class PredictRequest(BaseModel):
    data: PredictArgs


# ---- APP LIFESPAN ---- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    with open(MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl", "rb") as f:
        _, app.state.model = cloudpickle.load(f)
    app.state.logger = logging.getLogger(__name__)
    yield


app = FastAPI(lifespan=lifespan)


# ---- EXCEPTION HANDLER ---- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    app.state.logger.error(f"Validation failed for request: {await request.body()}")
    app.state.logger.error(f"Validation errors: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)


# ---- MAIN ENDPOINT ---- #
@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    model = request.app.state.model
    args = body.data

    try:
        # Replace "metric" string with actual callable
        metric_fn = np.mean if args.metric == "mean" else np.mean

        predictions = model.predict(
            metric=metric_fn,
            p_1_constraint=args.p_1_constraint,
            p_5_constraint=args.p_5_constraint,
            max_std=args.max_std,
            n_trials=args.n_trials,
            random_seed=args.random_seed,
            bootstrap_block_size=args.bootstrap_block_size,
            bootstrap_path_length=args.bootstrap_path_length,
            n_bootstrap_paths=args.n_bootstrap_paths,
        )

        request.app.state.logger.info(f"Input: {body}, Output: {predictions}")
        return {"predictions": predictions}

    except Exception as e:
        request.app.state.logger.error(f"Prediction failed: {str(e)}")
        return {"error": str(e)}
