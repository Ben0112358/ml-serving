import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    n_trials: int = 0
    trials_done: int = 0
    best_value: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def create(self, n_trials: int) -> JobRecord:
        job_id = str(uuid.uuid4())
        rec = JobRecord(job_id=job_id, n_trials=n_trials)
        with self._lock:
            self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job_id: str, fn: Any) -> None:
        self._executor.submit(fn)

    def to_public(self, rec: JobRecord) -> dict[str, Any]:
        return {
            "job_id": rec.job_id,
            "status": rec.status,
            "n_trials": rec.n_trials,
            "trials_done": rec.trials_done,
            "best_value": rec.best_value,
            "history": list(rec.history),
            "result": rec.result,
            "error": rec.error,
        }

    def cancel(self, job_id: str) -> bool:
        rec = self.get(job_id)
        if rec is None:
            return False
        rec.cancel_event.set()
        if rec.status in ("queued", "running"):
            rec.status = "cancelled"
        return True
