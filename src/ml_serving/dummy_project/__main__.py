import subprocess


def main():
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
    )


if __name__ == "__main__":
    main()