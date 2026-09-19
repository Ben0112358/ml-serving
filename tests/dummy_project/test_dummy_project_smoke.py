def test_config_paths_resolve(ml_serving_env):
    import ml_serving.config as config

    assert config.MODEL_DIR.name == "models"
