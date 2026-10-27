"""Download the reference models and write qwen.env inside the setup container."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.write_model_env import write_model_environment

MODELS = (
    ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", "qwen"),
    ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "bge"),
)
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def prepare_models(models_dir: Path, output: Path) -> None:
    from huggingface_hub import get_token, hf_hub_download

    if output.exists():
        raise FileExistsError(f"{output} already exists. Reuse it or choose a new --output path.")
    if not get_token():
        subprocess.run(["hf", "auth", "login"], check=True)
    hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        revision=SAM3_REVISION,
        local_dir=str(models_dir),
    )
    for model, revision, name in MODELS:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/download_hf_model.py"),
                "--model",
                model,
                "--revision",
                revision,
                "--output-dir",
                str(models_dir / name),
                "--env-prefix",
                name.upper(),
            ],
            check=True,
        )
    write_model_environment(models_dir, output)
    print(f"Models ready. Configuration saved to {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("/models"))
    parser.add_argument("--output", type=Path, default=Path("qwen.env"))
    args = parser.parse_args()
    prepare_models(args.models_dir.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
