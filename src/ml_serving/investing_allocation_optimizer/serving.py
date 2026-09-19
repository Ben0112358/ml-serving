from contextlib import asynccontextmanager
import logging
import threading
from typing import Any, Optional, Union

import cloudpickle
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, Field

from ml_serving.config import ENV_VAR_OUTPUT_SUFFIX, MODEL_DIR
from ml_serving.investing_allocation_optimizer.jobs import JobStore
from ml_serving.utils import setup_logging


class WeightBound(BaseModel):
    min: float = Field(0.0, ge=0.0, le=1.0)
    max: float = Field(1.0, ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    metric: str = Field("mean_return")
    metric_params: Optional[dict[str, Any]] = None
    p_1_constraint: Optional[float] = None
    p_5_constraint: Optional[float] = None
    max_std: Optional[float] = None
    n_trials: int = Field(100, ge=1, le=5000)
    random_seed: Optional[int] = None
    bootstrap_block_size: Union[int, str] = Field("cube root")
    horizon_years: float = Field(10.0, gt=0, le=100)
    n_bootstrap_paths: int = Field(1000, ge=10, le=50000)
    weight_bounds: Optional[dict[str, WeightBound]] = None


class PredictRequestBody(BaseModel):
    data: OptimizeRequest


class JobCreateResponse(BaseModel):
    job_id: str


def _metrics_for_api() -> list[dict[str, Any]]:
    try:
        from ml_training.investing_allocation_optimizer.utils.metrics import (
            metrics_for_api,
        )

        return metrics_for_api()
    except ImportError:
        return [
            {
                "key": "mean_return",
                "label": "Mean annualized return",
                "description": "",
            },
            {"key": "sharpe", "label": "Sharpe ratio", "description": ""},
        ]


def _weight_bounds_dict(
    req: OptimizeRequest,
) -> dict[str, tuple[float, float]] | None:
    if not req.weight_bounds:
        return None
    out: dict[str, tuple[float, float]] = {}
    for asset, wb in req.weight_bounds.items():
        if wb.min > wb.max:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid bounds for {asset}",
            )
        out[asset] = (wb.min, wb.max)
    return out


def _run_predict(
    model: Any,
    req: OptimizeRequest,
    progress_callback: Any | None = None,
    cancel_event: threading.Event | None = None,
    weight_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if weight_bounds is None:
        weight_bounds = _weight_bounds_dict(req)
    return model.predict(
        metric=req.metric,
        metric_params=req.metric_params,
        p_1_constraint=req.p_1_constraint,
        p_5_constraint=req.p_5_constraint,
        max_std=req.max_std,
        n_trials=req.n_trials,
        random_seed=req.random_seed,
        bootstrap_block_size=req.bootstrap_block_size,
        horizon_years=req.horizon_years,
        n_bootstrap_paths=req.n_bootstrap_paths,
        weight_bounds=weight_bounds,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


job_store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    with open(MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl", "rb") as f:
        _, app.state.model = cloudpickle.load(f)
    app.state.logger = logging.getLogger(__name__)
    app.state.job_store = job_store
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    app.state.logger.error(
        "Validation failed for request: %s",
        (await request.body()).decode(errors="replace"),
    )
    app.state.logger.error("Validation errors: %s", exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.get("/options")
async def options(request: Request):
    model = request.app.state.model
    assets = getattr(model, "assets_", list(getattr(model, "df", []).columns))
    return {
        "metrics": _metrics_for_api(),
        "assets": assets,
        "benchmark_label": "total-world",
        "defaults": {
            "metric": "mean_return",
            "n_trials": 100,
            "horizon_years": 10.0,
            "n_bootstrap_paths": 1000,
            "bootstrap_block_size": "cube root",
        },
        "sample_window": getattr(model, "sample_window_", None),
        "periods_per_year": getattr(model, "periods_per_year_", None),
    }


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(request: Request, body: OptimizeRequest):
    store: JobStore = request.app.state.job_store
    model = request.app.state.model
    logger = request.app.state.logger

    try:
        bounds = _weight_bounds_dict(body)
    except HTTPException:
        raise

    rec = store.create(n_trials=body.n_trials)

    def worker() -> None:
        rec.status = "running"
        lock = threading.Lock()

        def progress(trial: int, value: float, best: float) -> None:
            with lock:
                rec.trials_done = trial + 1
                rec.best_value = best
                rec.history.append(
                    {
                        "trial": float(trial),
                        "value": value,
                        "best_value": best,
                    }
                )
            if rec.cancel_event.is_set():
                return

        try:
            result = _run_predict(
                model,
                body,
                progress_callback=progress,
                cancel_event=rec.cancel_event,
                weight_bounds=bounds,
            )
            if rec.cancel_event.is_set():
                rec.status = "cancelled"
            else:
                rec.result = result
                rec.best_value = result.get("metric_value")
                rec.history = result.get("history", rec.history)
                rec.trials_done = result.get(
                    "trials_completed", rec.trials_done
                )
                rec.status = "completed"
        except Exception as exc:
            logger.exception("Job %s failed", rec.job_id)
            rec.status = "failed"
            rec.error = str(exc)

    store.submit(rec.job_id, worker)
    return JobCreateResponse(job_id=rec.job_id)


@app.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    store: JobStore = request.app.state.job_store
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return store.to_public(rec)


@app.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str):
    store: JobStore = request.app.state.job_store
    if not store.cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/predict")
async def predict(request: Request, body: PredictRequestBody):
    model = request.app.state.model
    args = body.data
    try:
        predictions = _run_predict(model, args)
        request.app.state.logger.info("Predict success metric=%s", args.metric)
        return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        request.app.state.logger.error("Prediction failed: %s", e)
        return {"error": str(e)}
