import subprocess


def main():
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.board_meeting_protocol.yaml",
            "up",
            "--build",
        ],
    )


if __name__ == "__main__":
    main()
