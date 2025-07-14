import pathlib as pl
import yaml
import os

ENV_VAR_ML_HOMELAB_ROOT = "ML_HOMELAB_ROOT"

try:
    ML_HOMELAB_ROOT_DIR = pl.Path(os.environ[ENV_VAR_ML_HOMELAB_ROOT])
except KeyError:
    raise EnvironmentError(
        f"{ENV_VAR_ML_HOMELAB_ROOT} is a required environment variable. "
        f"Please export it before continuing."
    )

CONFIG_PATH = ML_HOMELAB_ROOT_DIR / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"{CONFIG_PATH} was expected, but does not exist. "
        f"See the README.md how this is created in a different infra repo."
    )

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


MODEL_DIR = pl.Path(CONFIG["paths"]["models"])
LOGS_DIR = pl.Path(CONFIG["paths"]["serving_logs"])
