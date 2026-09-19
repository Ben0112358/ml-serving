def test_setup_logging_runs(ml_serving_env):
    from ml_serving.utils.logging import setup_logging

    setup_logging()
