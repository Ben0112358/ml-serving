import base64
import logging
from contextlib import asynccontextmanager
from typing import Literal

import cloudpickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_serving.board_meeting_protocol.backends import (
    CursorLocalModel,
    OpenAICompatibleModel,
    cursor_model_ids,
)
from ml_serving.board_meeting_protocol.documents import (
    build_document,
    extract_text,
    suffix_of,
)
from ml_serving.board_meeting_protocol.jobs import JobStore
from ml_serving.config import ENV_VAR_OUTPUT_SUFFIX, MODEL_DIR
from ml_serving.utils import setup_logging

MAX_FILE_BYTES = 5_000_000
MAX_ROWS = 3
job_store = JobStore()


class FilePart(BaseModel):
    filename: str
    content_base64: str


class RowIn(BaseModel):
    left: FilePart | None = None
    middle: FilePart | None = None
    right: FilePart | None = None
    note: str = ""
    feedback: str = ""


class JobIn(BaseModel):
    mode: Literal["writing", "reviewing"]
    backend: Literal["openai", "cursor"]
    row_model: str | None = None
    orchestrator_model: str | None = None
    orchestrator_note: str = ""
    rows: list[RowIn] = Field(default_factory=list)
    new_input: FilePart


def _decode(part: FilePart) -> bytes:
    try:
        data = base64.b64decode(part.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {part.filename}",
        ) from exc
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File is too large")
    try:
        suffix_of(part.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return data


def _text_of(part: FilePart | None) -> str:
    if part is None:
        return ""
    return extract_text(part.filename, _decode(part))


def _parse_row(row: RowIn) -> dict | None:
    left = _text_of(row.left)
    middle = _text_of(row.middle)
    right = row.feedback.strip()
    note = row.note.strip()
    if not any((left, middle, right, note)):
        return None
    if not left or not middle:
        raise HTTPException(
            status_code=400,
            detail="Each history row needs its first two documents",
        )
    return {
        "left": left,
        "middle": middle,
        "right": right,
        "note": note,
    }


def _style_template(
    rows: list[RowIn],
    output_name: str,
) -> tuple[str | None, bytes | None]:
    output_suffix = suffix_of(output_name)
    parts = [row.middle for row in rows if row.middle is not None]
    if not parts:
        return None, None
    same_type = [
        part for part in parts if suffix_of(part.filename) == output_suffix
    ]
    chosen = (same_type or parts)[-1]
    return chosen.filename, _decode(chosen)


def _backend(name: str):
    if name == "openai":
        model = OpenAICompatibleModel()
        if not model.available():
            raise HTTPException(
                status_code=400,
                detail="OpenAI-compatible backend is not configured",
            )
        return model
    model = CursorLocalModel()
    if not model.available():
        raise HTTPException(
            status_code=400,
            detail="Cursor backend is not configured",
        )
    return model


def _options() -> dict:
    openai = OpenAICompatibleModel()
    cursor = CursorLocalModel()
    return {
        "modes": ["writing", "reviewing"],
        "max_rows": MAX_ROWS,
        "backends": [
            {
                "id": "openai",
                "label": "OpenAI-compatible",
                "available": openai.available(),
                "models": [],
            },
            {
                "id": "cursor",
                "label": "Cursor",
                "available": cursor.available(),
                "models": cursor_model_ids() if cursor.available() else [],
            },
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    path = MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl"
    with open(path, "rb") as handle:
        _, app.state.writer = cloudpickle.load(handle)
    app.state.logger = logging.getLogger(__name__)
    app.state.job_store = job_store
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/options")
async def options():
    return _options()


@app.post("/jobs")
async def create_job(body: JobIn):
    if len(body.rows) > MAX_ROWS:
        raise HTTPException(status_code=400, detail="At most 3 history rows")
    writer = app.state.writer
    logger = app.state.logger
    backend = _backend(body.backend)
    rows = []
    for row in body.rows:
        parsed = _parse_row(row)
        if parsed is not None:
            rows.append(parsed)
    new_text = _text_of(body.new_input)
    record = job_store.create()

    def progress(event: dict) -> None:
        if record.cancel_event.is_set():
            raise RuntimeError("cancelled")
        job_store.note(record, event)

    def worker() -> None:
        record.status = "running"
        try:
            document_body = writer.predict(
                backend,
                mode=body.mode,
                rows=rows,
                new_text=new_text,
                orchestrator_note=body.orchestrator_note,
                row_model=body.row_model,
                orchestrator_model=body.orchestrator_model,
                progress=progress,
            )
            if record.cancel_event.is_set():
                record.status = "cancelled"
                return
            if body.mode == "reviewing":
                record.result = {"kind": "text", "text": document_body}
            else:
                template_name, template = _style_template(
                    body.rows,
                    body.new_input.filename,
                )
                raw, media = build_document(
                    body.new_input.filename,
                    document_body,
                    template_name,
                    template,
                )
                record.result = {
                    "kind": "file",
                    "filename": body.new_input.filename,
                    "media_type": media,
                    "content_base64": base64.b64encode(raw).decode("ascii"),
                }
            record.status = "completed"
            record.message = "Done"
        except Exception as exc:
            logger.exception("Job %s failed", record.job_id)
            if record.status != "cancelled":
                record.status = "failed"
                record.error = str(exc)

    job_store.submit(worker)
    return {"job_id": record.job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store.to_public(record)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if not job_store.cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": "cancelled"}
