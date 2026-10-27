import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_sim3(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Invalid pose shape or nonfinite pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous pose row")
    scale = float(np.linalg.norm(pose[:3, 0]))
    if scale <= 1e-6 or np.linalg.det(pose[:3, :3]) <= 0:
        raise ValueError("Nonpositive/degenerate Sim3 scale")
    rotation = pose[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("Pose is not a uniform-scale proper rotation")
    return scale


def loader_scale(pose):
    scale = validate_sim3(pose)
    return scale if abs(scale - 1.0) > 1e-4 else 1.0


def selected_global_pose(trajectory, g2o_idx, frame_count):
    trajectory = np.asarray(trajectory)
    if trajectory.shape != (frame_count + 1, 4, 4) or not 1 <= g2o_idx <= frame_count:
        raise ValueError("Unexpected optimizer seed/index contract")
    pose = trajectory[g2o_idx]
    validate_sim3(pose)
    return pose


def validate_keyframe_threshold(value):
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Keyframe match fraction must be numeric") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Keyframe match fraction must be finite and strictly between 0 and 1")
    return threshold
