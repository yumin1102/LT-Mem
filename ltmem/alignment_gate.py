from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Protocol

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

ALIGNMENT_THRESHOLD_M = 0.3
FUSION_VOXEL_SIZE_M = 0.04
OUTLIER_NB_POINTS = 4
OUTLIER_RADIUS_M = 0.10
DBSCAN_EPS_M = 0.12
DBSCAN_MIN_POINTS = 15
ICP_MIN_MODEL_POINTS = 100
ICP_TRIM_FRACTION = 0.70
ICP_SCHEDULE = ((0.60, 80), (0.30, 60), (0.15, 50), (0.08, 40))
ICP_MINIMUM_FITNESS = 0.50


class AlignmentGateStatus(Enum):
    PROCEED = "PROCEED"
    GLOBAL_HOLD = "GLOBAL_HOLD"


@dataclass
class AlignmentGateResult:
    status: AlignmentGateStatus
    distances_m: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentRequest:
    session_id: str
    structural_anchors: tuple[str, ...]


@dataclass(frozen=True)
class AnchorModel:
    frame_points: tuple[np.ndarray, ...]


class AlignmentPerceptionPort(Protocol):
    def observe_structural_anchor_points(
        self,
        request: AlignmentRequest,
    ) -> Mapping[str, tuple[np.ndarray, ...]]: ...


class AlignmentGate:
    def check(self, distances: List[float]) -> AlignmentGateResult:
        return AlignmentGateResult(
            AlignmentGateStatus.PROCEED
            if sum(distances) / len(distances) <= ALIGNMENT_THRESHOLD_M
            else AlignmentGateStatus.GLOBAL_HOLD
        )


class SessionAlignmentRegistry:
    def __init__(self, gate: Optional[AlignmentGate] = None):
        self.gate = gate or AlignmentGate()
        self.reference_models: Dict[str, np.ndarray] = {}
        self.reference_centers: Dict[str, np.ndarray] = {}
        self.results: Dict[str, AlignmentGateResult] = {}

    def initialize(self, session_id: str, models: Dict[str, AnchorModel]) -> AlignmentGateResult:
        reference_models = {}
        for label, model in models.items():
            supported = _supported_anchor_model(model.frame_points)
            if supported is None:
                result = AlignmentGateResult(AlignmentGateStatus.GLOBAL_HOLD)
                self.reference_models = {}
                self.reference_centers = {}
                self.results = {session_id: result}
                return result
            reference_models[label] = supported
        reference_centers = {label: model.mean(axis=0) for label, model in reference_models.items()}
        result = AlignmentGateResult(AlignmentGateStatus.PROCEED)
        self.reference_models = reference_models
        self.reference_centers = reference_centers
        self.results = {session_id: result}
        return result

    def evaluate(
        self,
        session_id: str,
        models: Dict[str, AnchorModel],
        registered_centers: Mapping[str, np.ndarray],
    ) -> AlignmentGateResult:
        distances: Dict[str, float] = {}
        if not self.reference_models:
            result = AlignmentGateResult(AlignmentGateStatus.GLOBAL_HOLD)
            self.results[session_id] = result
            return result
        for label, reference in self.reference_models.items():
            model = models.get(label)
            supported = _supported_anchor_model(model.frame_points) if model is not None else None
            center = _register_anchor_model(supported, reference) if supported is not None else None
            if center is None:
                result = AlignmentGateResult(AlignmentGateStatus.GLOBAL_HOLD, distances)
                self.results[session_id] = result
                return result
            distances[label] = float(np.linalg.norm(center - registered_centers[label]))
        result = self.gate.check(list(distances.values()))
        result.distances_m = distances
        self.results[session_id] = result
        return result


def _fuse_anchor_points(points: np.ndarray) -> Optional[np.ndarray]:
    if len(points) == 0 or not np.isfinite(points).all():
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(FUSION_VOXEL_SIZE_M)
    cloud, _ = cloud.remove_radius_outlier(nb_points=OUTLIER_NB_POINTS, radius=OUTLIER_RADIUS_M)
    if len(cloud.points) == 0:
        return None
    labels = np.asarray(
        cloud.cluster_dbscan(eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_POINTS, print_progress=False)
    )
    if not np.any(labels >= 0):
        return None
    largest = int(np.argmax(np.bincount(labels[labels >= 0])))
    return np.asarray(cloud.points)[labels == largest]


def _supported_anchor_model(frames: tuple[np.ndarray, ...]) -> Optional[np.ndarray]:
    if len(frames) < 2 or any(not np.isfinite(points).all() for points in frames):
        return None
    chunks = []
    for points in frames:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        chunks.append(np.asarray(cloud.voxel_down_sample(FUSION_VOXEL_SIZE_M).points))
    supported = []
    for index, chunk in enumerate(chunks):
        others = np.vstack([p for j, p in enumerate(chunks) if j != index])
        tree = cKDTree(others)
        distance, _ = tree.query(chunk, k=1, distance_upper_bound=OUTLIER_RADIUS_M)
        supported.append(chunk[np.isfinite(distance)])
    return _fuse_anchor_points(np.vstack(supported))


def _register_anchor_model(current: np.ndarray, reference: np.ndarray) -> Optional[np.ndarray]:
    if min(len(current), len(reference)) < ICP_MIN_MODEL_POINTS:
        return None
    tree = cKDTree(reference)
    transform = np.eye(4)
    for threshold, iterations in ICP_SCHEDULE:
        for _ in range(iterations):
            moved = current @ transform[:3, :3].T + transform[:3, 3]
            distance, target_index = tree.query(moved)
            candidates = np.flatnonzero(distance < threshold)
            count = int(len(candidates) * ICP_TRIM_FRACTION)
            if count < ICP_MIN_MODEL_POINTS:
                return None
            selected = candidates[np.argsort(distance[candidates])[:count]]
            source_points = moved[selected]
            target_points = reference[target_index[selected]]
            source_center = source_points.mean(axis=0)
            target_center = target_points.mean(axis=0)
            u, _, vt = np.linalg.svd(
                (target_points - target_center).T @ (source_points - source_center)
            )
            sign = np.ones(3)
            sign[-1] = np.linalg.det(u @ vt)
            delta = np.eye(4)
            delta[:3, :3] = u @ np.diag(sign) @ vt
            delta[:3, 3] = target_center - delta[:3, :3] @ source_center
            transform = delta @ transform
            if np.linalg.norm(delta - np.eye(4)) < 1e-7:
                break
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(current)
    target.points = o3d.utility.Vector3dVector(reference)
    final = o3d.pipelines.registration.evaluate_registration(
        source, target, ICP_SCHEDULE[-1][0], transform
    )
    if (
        not np.isfinite(transform).all()
        or not np.isfinite([final.fitness, final.inlier_rmse]).all()
        or final.fitness < ICP_MINIMUM_FITNESS
    ):
        return None
    inverse = np.linalg.inv(transform)
    return inverse[:3, :3] @ reference.mean(axis=0) + inverse[:3, 3]


def observe_anchor_models(
    request: AlignmentRequest,
    perception_port: AlignmentPerceptionPort,
) -> Dict[str, AnchorModel]:
    observed = perception_port.observe_structural_anchor_points(request)
    return {label: AnchorModel(observed.get(label, ())) for label in request.structural_anchors}


def run_alignment_stage(
    request: AlignmentRequest,
    perception_port: AlignmentPerceptionPort,
    registry: SessionAlignmentRegistry,
    registered_centers: Mapping[str, np.ndarray],
) -> AlignmentGateResult:
    models = observe_anchor_models(request, perception_port)
    return registry.evaluate(request.session_id, models, registered_centers)
