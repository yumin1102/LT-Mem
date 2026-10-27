from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltmem.local_model_integrity import MANIFEST_FILENAME, verify_local_model


def write_model_environment(models_dir: Path, output: Path) -> None:
    values = {"LLM_PROVIDER": "qwen"}
    for name in ("qwen", "bge"):
        model_dir = models_dir / name
        manifest = json.loads((model_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        _, digest = verify_local_model(str(model_dir), manifest["model_sha256"])
        prefix = name.upper()
        values[f"{prefix}_MODEL_PATH"] = f"/models/{name}"
        values[f"{prefix}_MODEL_SHA256"] = digest
        values[f"{prefix}_MODEL_MANIFEST"] = f"/models/{name}/{MANIFEST_FILENAME}"
    with (models_dir / "sam3.pt").open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    values.update(
        {
            "LT_SAM3_CKPT": "/models/sam3.pt",
            "LT_SAM3_CKPT_SHA256": checkpoint_digest,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "LT_MEM_REQUIRE_READ_ONLY_MODELS": "true",
            "USE_4BIT": "true",
        }
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write("".join(f"{key}={value}\n" for key, value in values.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify prepared models and write an env file for a Docker /models mount"
    )
    parser.add_argument("--models-dir", type=Path, default=Path("/models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen.env"),
        help="New env file; existing files are preserved",
    )
    args = parser.parse_args()
    write_model_environment(args.models_dir.expanduser(), args.output.expanduser())
    print(f"Wrote {args.output}. Mount {args.models_dir} read-only at /models for inference.")


if __name__ == "__main__":
    main()
