import logging
import subprocess
import os
from ml_serving.utils import setup_logging


def main():
    logger = logging.getLogger(__name__)
    logger.info("Running docker according to the docker-compose.yaml file.")

    env = os.environ.copy()
    if "ML_HOMELAB_ROOT" not in env:
        raise RuntimeError("ML_HOMELAB_ROOT is not set in the environment.")

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.dummy_project.yaml",
            "up",
            "--build",
            "-d"
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    logger = setup_logging()
    main()
