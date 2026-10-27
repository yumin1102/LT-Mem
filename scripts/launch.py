"""Run LT-Mem commands in Docker from the host checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = {
    "setup": ["python", "scripts/setup_models.py"],
    "run": ["python", "run.py"],
    "qa": ["python", "scripts/qa.py"],
    "parking": ["python", "scripts/parking.py"],
    "check": ["python", "scripts/check_runtime.py"],
    "shell": ["bash"],
}


def docker_command(command, arguments, *, image, models_dir, env_file):
    qa_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    qa_parser.add_argument("--provider", choices=("gemini", "qwen"))
    qa_parser.add_argument("--mode", choices=("natural", "events"), default="natural")
    options = qa_parser.parse_known_args(arguments)[0]
    provider = options.provider or ("gemini" if options.mode == "natural" else "qwen")
    cloud_qa = command == "qa" and provider == "gemini"
    raw_input = command == "run" and any(
        arg == "--input" or arg.startswith("--input=") for arg in arguments
    )
    image = image or ("lt-mem:full" if raw_input else "lt-mem:latest")
    result = ["docker", "run", "--rm", "-i"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        result.append("-t")
    if command != "setup":
        result += ["--runtime=nvidia", "--gpus", "all"]
    result += [
        "--shm-size=4g" if raw_input else "--shm-size=1g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "--mount",
        f"type=bind,src={ROOT},dst=/app",
        "-w",
        "/app",
    ]
    if command != "check":
        suffix = "" if command == "setup" else ",readonly"
        result += ["--mount", f"type=bind,src={models_dir},dst=/models{suffix}"]
        if command == "setup":
            if os.environ.get("HF_TOKEN"):
                result += ["-e", "HF_TOKEN"]
        else:
            result += ["--env-file", str(env_file)]
    if cloud_qa:
        result += ["-e", "LT_MEM_ALLOW_CLOUD_PIPELINE_EGRESS=true"]
        for name in (
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ):
            if os.environ.get(name):
                result += ["-e", name]
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials and not os.environ.get("GEMINI_API_KEY"):
            credentials_path = Path(credentials).expanduser().resolve(strict=True)
            result += [
                "--mount",
                f"type=bind,src={credentials_path},dst=/tmp/gemini-credentials.json,readonly",
                "-e",
                "GOOGLE_APPLICATION_CREDENTIALS=/tmp/gemini-credentials.json",
            ]
    return result + [image, *COMMANDS[command], *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--image", help="Image override; raw --input selects lt-mem:full by default"
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--env-file", type=Path, default=ROOT / "qwen.env")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without running Docker"
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Arguments passed to the selected command"
    )
    args = parser.parse_args()
    models_dir = args.models_dir.expanduser().resolve()
    command = docker_command(
        args.command,
        args.arguments,
        image=args.image,
        models_dir=models_dir,
        env_file=args.env_file.expanduser().resolve(),
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    if args.command == "setup":
        models_dir.mkdir(parents=True, exist_ok=True)
        (ROOT / "data").mkdir(exist_ok=True)
        (ROOT / "results").mkdir(exist_ok=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.exit(getattr(exc, "returncode", 1), f"Container command failed: {exc}\n")


if __name__ == "__main__":
    main()
