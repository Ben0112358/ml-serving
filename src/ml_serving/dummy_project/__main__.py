from ml_serving.config import LOGS_DIR
import logging
from datetime import datetime
import subprocess
import os


def setup_logging():
    log_file_path = LOGS_DIR / f"{datetime.today()}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(" "message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s -" " %(levelname)s - %(" "message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    logger = logging.getLogger(__name__)
    logger.info("Running docker according to the docker-compose.yaml file.")

    env = os.environ.copy()
    if "ML_HOMELAB_ROOT" not in env:
        raise RuntimeError("ML_HOMELAB_ROOT is not set in the environment.")

    subprocess.run(["docker", "compose", "up", "--build"], env=env, check=True)


if __name__ == "__main__":
    logger = setup_logging()
    main()
