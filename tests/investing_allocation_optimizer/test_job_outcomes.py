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
        OptimizeRequest(n_trials=3),
        None,
        logging.getLogger("investing-job-outcomes-test"),
    )


def test_completed_job_records_the_model_result(ml_serving_env):
    record = JobStore().create(n_trials=3)
    result = {
        "metric_value": 0.25,
        "history": [{"trial": 0.0, "value": 0.25, "best_value": 0.25}],
        "trials_completed": 3,
    }

    class Model:
        def predict(self, **kwargs):
            return result

    _run(record, Model())
    assert record.status == "completed"
    assert record.result is result
    assert record.best_value == 0.25
    assert record.history == result["history"]
    assert record.trials_done == 3
    assert record.error is None


def test_progress_is_kept_when_result_omits_history(ml_serving_env):
    record = JobStore().create(n_trials=3)

    class Model:
        def predict(self, progress_callback=None, **kwargs):
            progress_callback(0, 0.1, 0.1)
            progress_callback(1, 0.3, 0.3)
            return {"metric_value": 0.3}

    _run(record, Model())
    assert record.status == "completed"
    assert record.trials_done == 2
    assert record.history == [
        {"trial": 0.0, "value": 0.1, "best_value": 0.1},
        {"trial": 1.0, "value": 0.3, "best_value": 0.3},
    ]


def test_cancel_during_successful_predict_discards_result(ml_serving_env):
    record = JobStore().create(n_trials=3)

    class Model:
        def predict(self, **kwargs):
            record.cancel_event.set()
            return {"metric_value": 1.0, "trials_completed": 3}

    _run(record, Model())
    assert record.status == "cancelled"
    assert record.result is None
    assert record.error is None


def test_failed_predict_marks_job_failed(ml_serving_env):
    record = JobStore().create(n_trials=3)

    class Model:
        def predict(self, **kwargs):
            raise ValueError("synthetic failure")

    _run(record, Model())
    assert record.status == "failed"
    assert record.error == "synthetic failure"
    assert record.result is None
