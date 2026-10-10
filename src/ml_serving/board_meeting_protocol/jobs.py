import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    message: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def create(self) -> JobRecord:
        record = JobRecord(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, fn: Any) -> None:
        self._executor.submit(fn)

    def cancel(self, job_id: str) -> bool:
        record = self.get(job_id)
        if record is None:
            return False
        record.cancel_event.set()
        if record.status in {"queued", "running"}:
            record.status = "cancelled"
        return True

    def note(self, record: JobRecord, event: dict[str, Any]) -> None:
        with self._lock:
            _apply_trace(record, event)

    def to_public(self, record: JobRecord) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": record.job_id,
                "status": record.status,
                "message": record.message,
                "trace": [dict(step) for step in record.trace],
                "result": record.result,
                "error": record.error,
            }


def _apply_trace(record: JobRecord, event: dict[str, Any]) -> None:
    step_id = str(event.get("step") or "")
    current = next(
        (step for step in record.trace if step["id"] == step_id), None
    )
    kind = event.get("kind")
    if kind == "start":
        title = str(event.get("title") or "Model")
        record.trace.append(
            {
                "id": step_id,
                "title": title,
                "model": str(event.get("model") or ""),
                "state": "running",
                "thinking": "",
                "reply": "",
                "received": str(event.get("received") or ""),
            }
        )
        record.message = f"{title} — calling the model"
        return
    if current is None:
        return
    if kind == "delta":
        channel = str(event.get("channel") or "")
        text = str(event.get("text") or "")
        if channel in {"thinking", "reply"} and text:
            current[channel] += text
            activity = "thinking" if channel == "thinking" else "writing"
            record.message = f"{current['title']} — {activity}"
        return
    if kind == "done":
        current["state"] = "done"
        current["reply"] = str(event.get("reply") or current["reply"])
        record.message = f"{current['title']} — done"
