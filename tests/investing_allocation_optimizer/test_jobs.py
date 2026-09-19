from ml_serving.investing_allocation_optimizer.jobs import JobStore


def test_job_store_create_and_cancel():
    store = JobStore(max_workers=1)
    rec = store.create(n_trials=10)
    assert rec.status == "queued"
    assert store.cancel(rec.job_id)
    assert rec.cancel_event.is_set()
    pub = store.to_public(rec)
    assert pub["job_id"] == rec.job_id
