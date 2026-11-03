import logging

from ml_serving.investing_allocation_optimizer.jobs import JobStore


def _run(record, model) -> None:
    from ml_serving.investing_allocation_optimizer.serving import (
        OptimizeRequest,
        _run_job,
    )

    _run_job(
        record,
        model,
        OptimizeRequest(n_trials=5),
        None,
        logging.getLogger("investing-job-cancel-test"),
    )


def test_cancelled_queued_job_does_not_call_the_model(ml_serving_env):
    record = JobStore().create(n_trials=5)
    record.cancel_event.set()
    record.status = "cancelled"
    calls: list[int] = []

    class Model:
        def predict(self, **kwargs):
            calls.append(1)
            return {
                "metric_value": 1.0,
                "history": [],
                "trials_completed": 0,
            }

    _run(record, Model())
    assert calls == []
    assert record.status == "cancelled"
    assert record.error is None


def test_cancel_during_predict_stays_cancelled(ml_serving_env):
    record = JobStore().create(n_trials=5)

    class Model:
        def predict(self, **kwargs):
            record.cancel_event.set()
            raise RuntimeError("stopped")

    _run(record, Model())
    assert record.status == "cancelled"
    assert record.error is None
