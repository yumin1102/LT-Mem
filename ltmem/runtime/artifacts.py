from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
import subprocess
from time import monotonic

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _source_identity():
    files = [ROOT / "run.py"]
    files += sorted((ROOT / "ltmem").rglob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": digest.hexdigest()}


@contextmanager
def record_run(output_dir: Path, config: Path, *, mode: str, seed=None, **options):
    config_bytes = config.read_bytes()
    raw = yaml.safe_load(config_bytes)
    packages = {}
    for name in ("torch", "transformers", "sam3", "numpy", "open3d", "chromadb"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "seed": seed,
        "config": {
            "path": str(config.resolve()),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "values": {
                key: raw[key]
                for key in (
                    "data_root",
                    "env",
                    "sessions",
                    "objects",
                    "structural_anchors",
                    "alignment_gate_enabled",
                    "session_times",
                )
                if key in raw
            },
        },
        "options": {
            key: str(value) if isinstance(value, Path) else value for key, value in options.items()
        },
        "code": _source_identity(),
        "python": platform.python_version(),
        "packages": packages,
        "models": {
            "provider": os.getenv("LLM_PROVIDER", "qwen") if mode != "parking" else None,
            "use_4bit": os.getenv("USE_4BIT", "true") if mode != "parking" else None,
            "configured_sha256": {
                name: os.getenv(f"{prefix}_SHA256")
                for name, prefix in (
                    ("qwen", "QWEN_MODEL"),
                    ("bge", "BGE_MODEL"),
                    ("sam3", "LT_SAM3_CKPT"),
                )
                if mode != "parking" or name == "sam3"
            },
        },
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new --output_dir."
        ) from None
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    started = monotonic()
    try:
        yield
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(monotonic() - started, 3),
        )
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
