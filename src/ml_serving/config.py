import pathlib as pl
import yaml
import os

ENV_VAR_ML_HOMELAB_ROOT = pl.Path(os.environ["ML_HOMELAB_ROOT"])
ENV_VAR_OUTPUT_SUFFIX = os.environ["OUTPUT_SUFFIX"]


ENV_VAR_CONFIG_PATH = os.environ.get("CONFIG_PATH")

CONFIG = {}
if ENV_VAR_CONFIG_PATH and pl.Path(ENV_VAR_CONFIG_PATH).exists():
    with open(ENV_VAR_CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)

MODEL_DIR = pl.Path(
    CONFIG.get("paths", {}).get("models")
    or (ENV_VAR_ML_HOMELAB_ROOT / os.environ.get("MODEL_DIR", ""))
)
LOGS_DIR = pl.Path(
    CONFIG.get("paths", {}).get("data_logs")
    or (ENV_VAR_ML_HOMELAB_ROOT / os.environ.get("LOGS_DIR", ""))
)

for var_name, path in [
    ("MODEL_DIR", MODEL_DIR),
    ("LOGS_DIR", LOGS_DIR),
]:
    if path.name == "":
        raise RuntimeError(
            f"{var_name} must be set in CONFIG_PATH or "
            f"as an environment variable."
        )
