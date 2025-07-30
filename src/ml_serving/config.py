import pathlib as pl
import yaml
import os

ENV_VAR_ML_HOMELAB_ROOT = pl.Path(os.environ["ML_HOMELAB_ROOT"])
ENV_VAR_PROJECT_NAME = os.environ["PROJECT_NAME"]
ENV_VAR_MODE = os.environ["MODE"]
ENV_VAR_DOCKER_NETWORK_NAME = os.environ["DOCKER_NETWORK_NAME"]
ENV_VAR_TIMESTAMP = os.environ["TIMESTAMP"]
ENV_VAR_OUTPUT_SUFFIX = os.environ["OUTPUT_SUFFIX"]
ENV_VAR_CONFIG_PATH = pl.Path(os.environ["CONFIG_PATH"])


if not ENV_VAR_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"{ENV_VAR_CONFIG_PATH} was expected, but does not exist. "
        f"See the README.md how this is created in a different infra repo."
    )

with open(ENV_VAR_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

MODEL_DIR = pl.Path(CONFIG["paths"]["models"])
LOGS_DIR = pl.Path(CONFIG["paths"]["serving_logs"])
