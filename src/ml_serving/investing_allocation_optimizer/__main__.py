import subprocess


def main():
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.investing_allocation_optimizer.yaml",
            "up",
            "--build",
            "-d",
        ],
    )


if __name__ == "__main__":
    main()
