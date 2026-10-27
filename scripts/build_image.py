"""Check the Linux x86_64 host, then build the aligned or full Docker image."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent.parent
X86_MACHINES = {"x86_64", "amd64"}


def build_command(target: str, tag: str | None) -> list[str]:
    default_tag = f"lt-mem:{'latest' if target == 'aligned' else 'full'}"
    command = [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "--target",
        target,
        "-f",
        "docker/Dockerfile",
        "-t",
        tag or default_tag,
    ]
    return command + ["."]


def check_host(*, skip_gpu: bool) -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in X86_MACHINES:
        raise ValueError(
            f"Build on a Linux x86_64 host; detected {platform.system()} {platform.machine()}"
        )
    result = subprocess.run(
        ["docker", "info", "--format", "{{.Architecture}}"],
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip().lower() not in X86_MACHINES:
        raise ValueError(f"Docker daemon architecture {result.stdout.strip()!r} is not x86_64")
    if not skip_gpu:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            text=True,
            capture_output=True,
            check=True,
        )
        if "nvidia" not in json.loads(result.stdout):
            raise ValueError(
                "Docker's nvidia runtime is not configured; complete NVIDIA Container Toolkit "
                "setup before using the documented GPU commands"
            )
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        )
        versions = result.stdout.splitlines()
        if not versions or any(int(version.strip().split(".")[0]) < 580 for version in versions):
            raise ValueError("CUDA 13 requires an NVIDIA driver from R580 or newer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("aligned", "full"), default="aligned")
    parser.add_argument("--tag")
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Build-only host; skip NVIDIA runtime and driver checks",
    )
    parser.add_argument("--check-only", action="store_true", help="Check the host without building")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without contacting Docker"
    )
    args = parser.parse_args()
    try:
        command = build_command(args.target, args.tag)
        if not args.dry_run:
            check_host(skip_gpu=args.skip_gpu_check)
        print(f"Target: {args.target}", flush=True)
        print(shlex.join(command), flush=True)
        if not (args.dry_run or args.check_only):
            subprocess.run(command, cwd=ROOT, check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        parser.exit(1, f"Build stopped: {detail.strip() if detail else exc}\n")


if __name__ == "__main__":
    main()
