import logging

from ml_serving.board_meeting_protocol.jobs import JobStore, _MAX_JOBS


def _run(record, writer) -> None:
    from ml_serving.board_meeting_protocol.serving import (
        FilePart,
        JobIn,
        _run_job,
    )

    body = JobIn(
        mode="reviewing",
        backend="openai",
        new_input=FilePart(filename="notes.docx", content_base64=""),
    )
    _run_job(
        record,
        writer,
        object(),
        body,
        [],
        "",
        logging.getLogger("board-meeting-protocol-test"),
    )


def test_cancelled_queued_job_does_not_call_the_model(ml_serving_env):
    record = JobStore().create()
    record.cancel_event.set()
    record.status = "cancelled"
    calls: list[int] = []

    class Writer:
        def predict(self, *args, **kwargs):
            calls.append(1)
            return "nope"

    _run(record, Writer())
    assert calls == []
    assert record.status == "cancelled"
    assert record.error is None


def test_cancel_during_predict_stays_cancelled(ml_serving_env):
    record = JobStore().create()

    class Writer:
        def predict(self, *args, **kwargs):
            record.cancel_event.set()
            kwargs["progress"](
                {"kind": "start", "step": "row", "title": "Row"}
            )
            return "nope"

    _run(record, Writer())
    assert record.status == "cancelled"
    assert record.error is None


def test_predict_error_is_stored_as_failed(ml_serving_env):
    record = JobStore().create()

    class Writer:
        def predict(self, *args, **kwargs):
            raise RuntimeError("boom")

    _run(record, Writer())
    assert record.status == "failed"
    assert record.error == "boom"


def test_create_drops_oldest_finished_job_at_the_cap():
    store = JobStore()
    records = [store.create() for _ in range(_MAX_JOBS)]
    for record in records:
        record.status = "completed"
    records[1].status = "running"
    newest = store.create()
    assert store.get(records[0].job_id) is None
    assert store.get(records[1].job_id) is records[1]
    assert store.get(newest.job_id) is newest
    assert len(store._jobs) == _MAX_JOBS


def test_create_keeps_queued_jobs_past_the_cap():
    store = JobStore()
    records = [store.create() for _ in range(_MAX_JOBS)]
    extra = store.create()
    assert store.get(records[0].job_id) is records[0]
    assert store.get(extra.job_id) is extra
    assert len(store._jobs) == _MAX_JOBS + 1
