import importlib


def _reload_config():
    import ml_serving.config as config

    return importlib.reload(config)


def test_empty_config_file_falls_back_to_env_vars(
    ml_serving_env, monkeypatch, tmp_path
):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    config = _reload_config()

    assert config.MODEL_DIR == tmp_path / "models"
    assert config.LOGS_DIR == tmp_path / "logs" / "serving"
