from pathlib import Path
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sklearn.cluster import DBSCAN

from ltmem.local_model_integrity import (
    assert_artifact_unchanged,
    capture_artifact_fingerprint,
    enforce_read_only_model_policy,
    verify_local_file,
)
from ltmem.models import ObjectCluster, SessionPerception


class MAST3RDataLoader:
    def __init__(self, data_root: str, session_dirs: Mapping[str, str]):
        self.data_root = Path(data_root).expanduser().resolve()
        self.session_dirs = session_dirs

    def list_frames(self, session_id: str) -> List[int]:
        ply_dir = self.data_root / self.session_dirs[session_id] / "ply"
        frames = sorted(int(path.stem) for path in ply_dir.glob("*.npy"))
        if not frames:
            raise FileNotFoundError(f"No aligned frames for {session_id}: {ply_dir}")
        return frames

    def load_frame(self, session_id: str, frame_idx: int) -> Dict:
        session_dir = self.data_root / self.session_dirs[session_id]

        frame_str = f"{frame_idx:06d}"
        image_path = session_dir / "image" / f"{frame_str}.png"
        ply_path = session_dir / "ply" / f"{frame_str}.npy"
        pose_path = session_dir / "pose" / f"{frame_str}.txt"
        pointcloud = np.load(ply_path, allow_pickle=False)
        pose = np.loadtxt(pose_path)

        return {
            "image_path": str(image_path),
            "pointcloud": pointcloud,
            "pose": pose,
        }


def project_mask_to_3d(
    mask: np.ndarray,
    pointmap: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    height, width = mask.shape

    pointmap_2d = pointmap.reshape(height, width, 3)

    valid_pixels = mask & np.isfinite(pointmap_2d).all(axis=2)
    points_3d = np.asarray(pointmap_2d[valid_pixels], dtype=float)

    camera_to_world = np.asarray(pose, dtype=float)
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return (rotation @ points_3d.T).T + translation


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SAM3Segmenter:
    def __init__(self):
        configured_model = os.getenv(
            "LT_SAM3_CKPT",
            str(_PROJECT_ROOT / "models" / "sam3" / "sam3.pt"),
        )
        checkpoint, _ = verify_local_file(
            configured_model,
            os.getenv("LT_SAM3_CKPT_SHA256", ""),
            file_label="SAM3 checkpoint",
            path_env="LT_SAM3_CKPT",
            sha_env="LT_SAM3_CKPT_SHA256",
        )
        model_fingerprint = capture_artifact_fingerprint(checkpoint)
        enforce_read_only_model_policy(
            checkpoint,
            specific_env="LT_SAM3_REQUIRE_READ_ONLY",
        )
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            device="cuda",
            load_from_HF=False,
        )
        assert_artifact_unchanged(checkpoint, model_fingerprint, artifact_label="SAM3 checkpoint")
        self.processor = Sam3Processor(model)
        self.current_state: Optional[Dict[str, Any]] = None
        self.current_frame: Optional[Tuple[str, int]] = None

    def segment(
        self,
        image: np.ndarray,
        prompt: str,
        session_id: str,
        frame_idx: int,
    ) -> List[Dict]:
        frame = (session_id, frame_idx)
        if self.current_frame != frame:
            self.current_state = self.processor.set_image(Image.fromarray(image))
            self.current_frame = frame

        output = self.processor.set_text_prompt(
            state=self.current_state,
            prompt=prompt,
        )
        results = []
        for mask_tensor, score in zip(output["masks"], output["scores"]):
            mask = mask_tensor.bool().cpu().numpy().squeeze()
            if mask.any():
                results.append(
                    {
                        "mask": mask,
                        "feature": self._extract_mask_feature(mask),
                        "detection_score": float(score.item()),
                        "detection_id": (session_id, frame_idx, prompt.strip(), len(results)),
                    }
                )
        return results

    def _extract_mask_feature(self, mask: np.ndarray) -> np.ndarray:
        state = cast(Dict[str, Any], self.current_state)
        features = state["backbone_out"]["vision_features"][0]
        resized_mask = torch.nn.functional.adaptive_max_pool2d(
            torch.from_numpy(mask).to(device=features.device, dtype=torch.float32)[None, None],
            output_size=features.shape[-2:],
        )[0, 0].bool()
        feature = features[:, resized_mask].mean(dim=1)
        return feature.detach().cpu().numpy().astype(np.float32)


class PerceptionStage:
    _OBSERVED_POINT_STRIDE = 64
    _OBJECT_VOXEL_SIZE_M = 0.02
    _OBJECT_CLUSTER_EPS_M = 0.10
    _OBJECT_CLUSTER_MIN_SAMPLES = 30
    _DUPLICATE_MASK_IOU = 0.5

    def __init__(self, data_root: str, session_dirs: Mapping[str, str]) -> None:
        self.data_loader = MAST3RDataLoader(data_root, session_dirs)
        self.segmenter = SAM3Segmenter()
        self._fragments_by_label: Dict[str, List[Dict[str, Any]]] = {}
        self._current_observed_point_chunks: List[np.ndarray] = []

    def _merge_duplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        selected = {}
        for index in sorted(
            range(len(detections)),
            key=lambda index: detections[index]["detection_score"],
            reverse=True,
        ):
            mask = detections[index]["mask"]
            for other in selected:
                reference = detections[other]["mask"]
                intersection = np.count_nonzero(mask & reference)
                union = np.count_nonzero(mask | reference)
                if intersection > self._DUPLICATE_MASK_IOU * union:
                    selected[other] = {
                        **selected[other],
                        "mask": selected[other]["mask"] | mask,
                    }
                    break
            else:
                selected[index] = detections[index]
        return [selected[index] for index in sorted(selected)]

    def cluster_and_merge_objects(
        self,
        object_class: str,
    ) -> List[ObjectCluster]:
        fragments = self._fragments_by_label.get(object_class, [])
        if not fragments:
            return []

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.vstack([fragment["points_3d"] for fragment in fragments])
        )
        voxel_size = self._OBJECT_VOXEL_SIZE_M
        traced_cloud, _, source_indices = cloud.voxel_down_sample_and_trace(
            voxel_size,
            cloud.get_min_bound() - voxel_size / 2,
            cloud.get_max_bound() + voxel_size / 2,
        )
        points = np.asarray(traced_cloud.points)
        labels = DBSCAN(
            eps=self._OBJECT_CLUSTER_EPS_M,
            min_samples=self._OBJECT_CLUSTER_MIN_SAMPLES,
        ).fit_predict(points)
        point_fragments = np.repeat(
            np.arange(len(fragments)),
            [len(fragment["points_3d"]) for fragment in fragments],
        )
        components = [
            np.flatnonzero(labels == cluster_id) for cluster_id in np.unique(labels[labels >= 0])
        ]
        if not components:
            return []
        support = np.stack(
            [
                np.bincount(
                    point_fragments[np.concatenate([source_indices[index] for index in indices])],
                    minlength=len(fragments),
                )
                for indices in components
            ]
        )
        representatives = np.unique(support[:, support.any(axis=0)].argmax(axis=0))
        clusters = []
        for component in representatives:
            cluster_points = points[components[component]]
            detections = {
                fragments[index]["detection_id"]: fragments[index]
                for index in np.flatnonzero(support[component])
            }
            contributors = [detections[detection_id] for detection_id in sorted(detections)]
            observation_confidence = float(
                np.mean([fragment["detection_score"] for fragment in contributors])
            )
            features = [fragment["feature"] for fragment in contributors]
            embedding = np.stack(features).mean(axis=0, dtype=np.float32)
            cluster = ObjectCluster(
                points=cluster_points,
                observation_confidence=observation_confidence,
                observation_count=len(contributors),
                embedding=embedding,
            )
            clusters.append(cluster)
        return clusters

    def process_frame(
        self,
        frame_idx: int,
        session_id: str,
        object_classes: List[str],
        structural_alignment: bool = False,
    ) -> None:
        frame_data = self.data_loader.load_frame(session_id, frame_idx)
        image = cv2.imread(frame_data["image_path"])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camera_points = frame_data["pointcloud"].reshape(-1, 3)[:: self._OBSERVED_POINT_STRIDE]
        camera_points = np.asarray(camera_points, dtype=float)
        camera_points = camera_points[np.isfinite(camera_points).all(axis=1)]
        pose = np.asarray(frame_data["pose"], dtype=float)
        world_points = (pose[:3, :3] @ camera_points.T).T + pose[:3, 3]
        self._current_observed_point_chunks.append(np.asarray(world_points, dtype=np.float32))
        for obj_class in object_classes:
            fragments = self._fragments_by_label.setdefault(obj_class, [])
            detections = self.segmenter.segment(image_rgb, obj_class, session_id, frame_idx)
            if not structural_alignment:
                detections = self._merge_duplicate_detections(detections)
            for detection in detections:
                points = project_mask_to_3d(
                    detection["mask"], frame_data["pointcloud"], frame_data["pose"]
                )
                if len(points):
                    fragments.append(
                        {
                            "points_3d": points,
                            "detection_id": detection["detection_id"],
                            "detection_score": detection["detection_score"],
                            "feature": detection["feature"],
                        }
                    )

    def process_session(
        self,
        session_id: str,
        object_classes: List[str],
        structural_alignment: bool = False,
    ) -> SessionPerception:
        self._fragments_by_label = {}
        self._current_observed_point_chunks = []
        frames = self.data_loader.list_frames(session_id)

        for index, frame_idx in enumerate(frames, 1):
            self.process_frame(
                frame_idx,
                session_id,
                object_classes,
                structural_alignment=structural_alignment,
            )
            if index % 50 == 0 or index == len(frames):
                print(f"    Frames {index}/{len(frames)}", flush=True)

        scene_points = np.vstack(self._current_observed_point_chunks)
        self._current_observed_point_chunks = []
        if structural_alignment:
            return SessionPerception(
                session_id,
                [],
                scene_points,
                self._collect_anchor_points(object_classes),
            )
        clusters = [
            cluster for label in object_classes for cluster in self.cluster_and_merge_objects(label)
        ]
        return SessionPerception(session_id, clusters, scene_points)

    def _collect_anchor_points(
        self,
        labels: List[str],
    ) -> Dict[str, Tuple[np.ndarray, ...]]:
        observed = {}
        for label in labels:
            frames: Dict[int, List[np.ndarray]] = {}
            for fragment in self._fragments_by_label.get(label, []):
                frame_id = fragment["detection_id"][1]
                frames.setdefault(frame_id, []).append(fragment["points_3d"])
            if frames:
                observed[label] = tuple(np.vstack(frames[frame]) for frame in sorted(frames))
        return observed
