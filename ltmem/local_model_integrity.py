from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple


MANIFEST_FILENAME = "lt_mem_model_manifest.json"
SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def capture_artifact_fingerprint(artifact_path: Path) -> tuple:
    root = Path(artifact_path).resolve(strict=True)
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    records = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(f"Model artifact must not contain a symbolic link: {candidate}")
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        relative = candidate.name if root.is_file() else candidate.relative_to(root).as_posix()
        records.append((relative, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(records)


def assert_artifact_unchanged(
    artifact_path: Path,
    expected_fingerprint: tuple,
    *,
    artifact_label: str,
) -> None:
    current = capture_artifact_fingerprint(artifact_path)
    if current != expected_fingerprint:
        raise RuntimeError(
            f"Local {artifact_label} changed after integrity verification; "
            "refusing to continue with a mutable model artifact"
        )


def verify_local_file(
    file_path: str,
    expected_sha256: str,
    *,
    file_label: str = "file",
    path_env: str = "FILE_PATH",
    sha_env: str = "FILE_SHA256",
) -> Tuple[Path, str]:
    if not isinstance(file_path, str) or not file_path.strip():
        raise RuntimeError(f"{path_env} must point to an existing local {file_label}")

    candidate = Path(file_path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{path_env} must point to an existing local {file_label}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{path_env} must point to a file")

    expected = str(expected_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise RuntimeError(f"{sha_env} must be a 64-character SHA-256 digest")

    actual = _sha256_file(resolved)
    if actual != expected:
        raise RuntimeError(
            f"Local {file_label} checksum mismatch: {resolved}. "
            f"Refusing to deserialize a file that does not match {sha_env}"
        )
    return resolved, actual


def _resolve_model_directory(
    model_path: str,
    *,
    path_env: str = "MODEL_PATH",
) -> Path:
    if not isinstance(model_path, str) or not model_path.strip():
        raise RuntimeError(f"{path_env} must point to an existing local model directory")
    candidate = Path(model_path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{path_env} must point to an existing local model directory; "
            "Hugging Face Hub model IDs are not accepted"
        ) from exc
    if not resolved.is_dir():
        raise RuntimeError(f"{path_env} must point to a directory")
    return resolved


def enforce_read_only_model_policy(
    artifact_path: Path,
    *,
    specific_env: str,
) -> None:
    master_env = "LT_MEM_REQUIRE_READ_ONLY_MODELS"
    master_required = os.getenv(master_env, "false").strip().lower() == "true"
    specific_required = os.getenv(specific_env, "false").strip().lower() == "true"
    if not (master_required or specific_required):
        return
    requirement_env = master_env if master_required else specific_env
    try:
        flags = os.statvfs(artifact_path).f_flag
    except OSError as exc:
        raise RuntimeError(
            f"Unable to verify read-only model artifact for {artifact_path}"
        ) from exc
    if not flags & os.ST_RDONLY:
        raise RuntimeError(
            f"{requirement_env}=true requires the local model artifact to be "
            "mounted on a read-only filesystem"
        )


def _iter_model_files(root: Path, manifest_path: Path) -> Iterable[Tuple[str, Path]]:
    manifest_absolute = manifest_path.absolute()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if candidate.is_symlink():
            raise RuntimeError(
                f"Local model must not contain a symbolic link: {relative.as_posix()}"
            )
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if candidate.absolute() == manifest_absolute:
            continue
        if candidate.is_file():
            yield relative.as_posix(), candidate


def _aggregate_sha256(entries: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def create_model_manifest(
    model_path: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    root = _resolve_model_directory(model_path)
    destination = (
        Path(manifest_path).expanduser().resolve() if manifest_path else root / MANIFEST_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        {"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)}
        for relative, path in sorted(_iter_model_files(root, destination))
    ]
    if not entries:
        raise RuntimeError("Local model directory contains no model files")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "model_sha256": _aggregate_sha256(entries),
        "files": entries,
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return payload


def verify_local_model(
    model_path: str,
    expected_sha256: str,
    manifest_path: Optional[str] = None,
    *,
    model_label: str = "model",
    path_env: str = "MODEL_PATH",
    sha_env: str = "MODEL_SHA256",
) -> Tuple[Path, str]:
    root = _resolve_model_directory(model_path, path_env=path_env)
    expected = str(expected_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise RuntimeError(f"{sha_env} must be a 64-character SHA-256 digest")

    manifest = (
        Path(manifest_path).expanduser().resolve() if manifest_path else root / MANIFEST_FILENAME
    )
    if not manifest.is_file():
        raise RuntimeError(
            f"Local {model_label} checksum manifest not found: {manifest}. "
            "Create it with scripts/download_hf_model.py"
        )

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid local {model_label} checksum manifest: {manifest}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported local {model_label} checksum manifest schema")
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("files"), list):
        raise RuntimeError(f"Invalid local {model_label} checksum manifest contract")

    entries: List[Dict[str, Any]] = []
    seen = set()
    for raw_entry in payload["files"]:
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"Invalid local {model_label} checksum entry")
        relative_text = raw_entry.get("path")
        size = raw_entry.get("size")
        digest = raw_entry.get("sha256")
        if not isinstance(relative_text, str):
            raise RuntimeError("Model checksum paths must be strings")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"Unsafe model checksum path: {relative_text!r}")
        if relative_text in seen:
            raise RuntimeError(f"Duplicate model checksum path: {relative_text}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"Invalid model file size for {relative_text}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"Invalid model file digest for {relative_text}")
        seen.add(relative_text)
        entries.append({"path": relative_text, "size": size, "sha256": digest})

    entries.sort(key=lambda entry: entry["path"])
    actual_files = {relative for relative, _path in _iter_model_files(root, manifest)}
    if actual_files != seen:
        missing = sorted(seen - actual_files)
        extra = sorted(actual_files - seen)
        raise RuntimeError(
            f"Local {model_label} file set differs from manifest; missing={missing}, extra={extra}"
        )

    for entry in entries:
        path = root.joinpath(*PurePosixPath(entry["path"]).parts)
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Local {model_label} checksum mismatch: {entry['path']}")

    aggregate = _aggregate_sha256(entries)
    if payload.get("model_sha256") != aggregate:
        raise RuntimeError(f"Local {model_label} manifest aggregate checksum is invalid")
    if aggregate != expected:
        raise RuntimeError(f"Local {model_label} does not match {sha_env}")
    return root, aggregate
