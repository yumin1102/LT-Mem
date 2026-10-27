from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RuntimeObjectConfig:
    name: str
    object_classes: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path
    env: str
    sessions: dict[str, str]
    objects: tuple[RuntimeObjectConfig, ...]
    structural_anchors: tuple[str, ...]
    alignment_gate_enabled: bool = False


def load_runtime_config(
    path: str | Path,
    *,
    require_data: bool = False,
) -> RuntimeConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))

    root_value = raw["data_root"]
    if not root_value:
        raise ValueError(f"missing required 'data_root' in {source}")
    data_root = (source.parent / Path(root_value).expanduser()).resolve()

    env = raw["env"].strip()
    if not env:
        raise ValueError(f"missing required 'env' in {source}")

    alignment_gate_enabled = raw.get("alignment_gate_enabled", False)
    structural_anchors = ()
    if alignment_gate_enabled:
        anchors = raw["structural_anchors"]
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"'structural_anchors' must be a non-empty list in {source}")
        structural_anchors = tuple(value.strip() for value in anchors)
        if any(not value for value in structural_anchors):
            raise ValueError("'structural_anchors' must contain non-empty strings")

    sessions = {}
    for index, (folder, session_id) in enumerate(raw["sessions"]):
        if not isinstance(session_id, str) or not re.fullmatch(r"Session[1-9][0-9]*", session_id):
            raise ValueError(f"sessions[{index}] label must be SessionN (N >= 1): {session_id!r}")
        folder = folder.strip()
        if not folder:
            raise ValueError(f"sessions[{index}] requires a folder name")
        if session_id in sessions:
            raise ValueError("duplicate session label in sessions")
        if folder in sessions.values():
            raise ValueError("duplicate folder in sessions")
        sessions[session_id] = folder
    if not sessions:
        raise ValueError("'sessions' must not be empty")

    session_numbers = [int(label.removeprefix("Session")) for label in sessions]
    if any(current <= previous for previous, current in zip(session_numbers, session_numbers[1:])):
        raise ValueError("sessions must be in strictly increasing session order")

    if alignment_gate_enabled and next(iter(sessions)) != "Session1":
        raise ValueError("sessions must start with Session1 for the alignment reference")

    objects = []
    for index, item in enumerate(raw["objects"]):
        classes_raw = item["object_classes"]
        if not isinstance(classes_raw, list) or not classes_raw:
            raise ValueError(f"objects[{index}] requires non-empty object_classes")
        object_classes = tuple(value.strip() for value in classes_raw)
        if any(not value for value in object_classes):
            raise ValueError(f"objects[{index}] object_classes must contain non-empty strings")
        name = item["name"].strip()
        if not name:
            raise ValueError(f"objects[{index}] requires a canonical name")
        objects.append(RuntimeObjectConfig(name=name, object_classes=object_classes))
    if not objects:
        raise ValueError("'objects' must not be empty")
    normalized_names = [obj.name.casefold() for obj in objects]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("duplicate object name in objects")

    if require_data and not data_root.is_dir():
        raise ValueError(f"dataset directory does not exist: {data_root}")
    return RuntimeConfig(
        data_root=data_root,
        env=env,
        sessions=sessions,
        objects=tuple(objects),
        structural_anchors=structural_anchors,
        alignment_gate_enabled=alignment_gate_enabled,
    )


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
