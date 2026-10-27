#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltmem.local_model_integrity import MANIFEST_FILENAME, create_model_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a pinned Hugging Face snapshot for local-only runtime use"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face source repository",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable source commit SHA to download",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local runtime directory for the downloaded model",
    )
    parser.add_argument(
        "--env-prefix",
        default="QWEN",
        help="Environment-variable prefix printed after download (for example QWEN or BGE)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision.strip()):
        parser.error("--revision must be an immutable 40-character commit SHA")
    env_prefix = args.env_prefix.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env_prefix):
        parser.error("--env-prefix must contain only uppercase letters, digits, and underscores")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model}@{args.revision} to {output}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=str(output),
        )
    except Exception as exc:
        print(f"Error: {exc}")
        print("Tip: Set HF_TOKEN only for this explicit download step if the model is gated.")
        raise

    payload = create_model_manifest(str(output))
    print(f"Manifest: {output / MANIFEST_FILENAME}")
    print(f"export {env_prefix}_MODEL_PATH={output}")
    print(f"export {env_prefix}_MODEL_SHA256={payload['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
