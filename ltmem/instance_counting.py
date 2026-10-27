from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

VOXEL_SIZE_M = 0.02
MIN_VOXEL_POINTS = 50
MASK_IOU_THRESHOLD = 0.5
SIFT_FEATURES = 2000
MATCH_RATIO = 0.75
MIN_SCENE_MATCHES = 12
MIN_MASK_MATCHES = 4
MIN_MATCH_PURITY = 0.5
SURFACE_DISTANCE_M = 0.35
MIN_SURFACE_COVERAGE = 0.5
MAX_ASSOCIATION_POINTS = 1024


@dataclass
class MaskObservation:
    frame: int
    mask_index: int
    prompt: str
    points: np.ndarray


@dataclass
class FrameFeatures:
    xy: np.ndarray
    descriptors: np.ndarray | None
    owners: np.ndarray


@dataclass
class CountedInstance:
    observations: list[MaskObservation]
    voxel_points: int


class _Groups:
    def __init__(self, frames: list[set[int]]):
        self.parents = list(range(len(frames)))
        self.frames = [set(values) for values in frames]

    def find(self, index: int) -> int:
        while index != self.parents[index]:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first != second and self.frames[first].isdisjoint(self.frames[second]):
            self.parents[second] = first
            self.frames[first].update(self.frames[second])

    def collect(self, items: list[list[int]]) -> list[list[int]]:
        groups = {}
        for index, values in enumerate(items):
            groups.setdefault(self.find(index), []).extend(values)
        return list(groups.values())


def _collect_observations(loader, segmenter, session_id, object_classes):
    from ltmem.perception import project_mask_to_3d

    observations = []
    views = []
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    for frame_index in loader.list_frames(session_id):
        frame = loader.load_frame(session_id, frame_index)
        image = cv2.cvtColor(cv2.imread(frame["image_path"]), cv2.COLOR_BGR2RGB)
        detections = [
            (prompt, index, detection)
            for prompt in object_classes
            for index, detection in enumerate(
                segmenter.segment(image, prompt, session_id, frame_index)
            )
        ]
        if not detections:
            continue
        gray = cv2.imread(frame["image_path"], cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        xy = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        masks, indices = [], []
        for prompt, mask_index, detection in sorted(
            detections, key=lambda item: item[2]["detection_score"], reverse=True
        ):
            mask = detection["mask"]
            if any(
                np.count_nonzero(mask & other) > MASK_IOU_THRESHOLD * np.count_nonzero(mask | other)
                for other in masks
            ):
                continue
            masks.append(mask)
            indices.append(len(observations))
            observations.append(
                MaskObservation(
                    frame_index,
                    mask_index,
                    prompt,
                    project_mask_to_3d(mask, frame["pointcloud"], frame["pose"]),
                )
            )
        owners = np.full(len(keypoints), -1, dtype=int)
        for mask, index in reversed(list(zip(masks, indices))):
            interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            pixels = np.rint(xy).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, mask.shape[1] - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, mask.shape[0] - 1)
            owners[interior[pixels[:, 1], pixels[:, 0]] > 0] = index
        views.append(FrameFeatures(xy, descriptors, owners))
    return observations, views


def _match_views(first: FrameFeatures, second: FrameFeatures, matcher):
    if (
        first.descriptors is None
        or second.descriptors is None
        or len(first.descriptors) < 2
        or len(second.descriptors) < 2
    ):
        return []
    forward = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    backward = matcher.knnMatch(second.descriptors, first.descriptors, k=2)
    reverse = {
        (pair[0].trainIdx, pair[0].queryIdx)
        for pair in backward
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    }
    matches = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < MATCH_RATIO * pair[1].distance
        and (pair[0].queryIdx, pair[0].trainIdx) in reverse
    ]
    if len(matches) < MIN_SCENE_MATCHES:
        return []
    left = np.asarray([match.queryIdx for match in matches])
    right = np.asarray([match.trainIdx for match in matches])
    _, inliers = cv2.findFundamentalMat(first.xy[left], second.xy[right], cv2.FM_RANSAC, 1.5, 0.99)
    if inliers is None or inliers.sum() < MIN_SCENE_MATCHES:
        return []
    keep = inliers.ravel().astype(bool)
    a, b = first.owners[left[keep]], second.owners[right[keep]]
    valid = (a >= 0) & (b >= 0)
    pairs, counts = np.unique(np.column_stack((a[valid], b[valid])), axis=0, return_counts=True)
    edges = []
    for (first_id, second_id), count in zip(pairs, counts):
        purity = count / min(np.count_nonzero(a == first_id), np.count_nonzero(b == second_id))
        if count >= MIN_MASK_MATCHES and purity >= MIN_MATCH_PURITY:
            edges.append((int(count), float(purity), int(first_id), int(second_id)))
    return edges


def _visual_groups(observations, views):
    cv2.setRNGSeed(0)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    edges = []
    for index, view in enumerate(views):
        for previous in views[:index]:
            edges.extend(_match_views(view, previous, matcher))
    groups = _Groups([{observation.frame} for observation in observations])
    for _, _, first, second in sorted(edges, reverse=True):
        groups.join(first, second)
    return groups.collect([[index] for index in range(len(observations))])


def _merge_surfaces(observations, tracks):
    frames = [{observations[index].frame for index in track} for track in tracks]
    ordered = sorted(
        (
            observations[index].frame,
            observations[index].prompt,
            observations[index].mask_index,
            track_id,
            index,
        )
        for track_id, track in enumerate(tracks)
        for index in track
    )
    clouds = []
    for _, _, _, track_id, index in ordered:
        points = observations[index].points
        if not len(points):
            continue
        points = points[:: max(1, int(np.ceil(len(points) / MAX_ASSOCIATION_POINTS)))]
        clouds.append((track_id, points, cKDTree(points), points.min(0), points.max(0)))
    pairs = {}
    for index, (first, points, tree, minimum, maximum) in enumerate(clouds):
        for second, other, other_tree, other_min, other_max in clouds[:index]:
            if first == second or not frames[first].isdisjoint(frames[second]):
                continue
            if np.any(minimum > other_max + SURFACE_DISTANCE_M) or np.any(
                other_min > maximum + SURFACE_DISTANCE_M
            ):
                continue
            coverage = max(
                np.mean(other_tree.query(points)[0] <= SURFACE_DISTANCE_M),
                np.mean(tree.query(other)[0] <= SURFACE_DISTANCE_M),
            )
            if coverage >= MIN_SURFACE_COVERAGE:
                key = tuple(sorted((first, second)))
                pairs[key] = max(pairs.get(key, 0), float(coverage))
    groups = _Groups(frames)
    for (first, second), _ in sorted(pairs.items(), key=lambda item: (-item[1], item[0])):
        groups.join(first, second)
    return groups.collect(tracks)


def count_session_instances(loader, segmenter, session_id, object_classes):
    observations, views = _collect_observations(loader, segmenter, session_id, object_classes)
    tracks = _merge_surfaces(observations, _visual_groups(observations, views))
    instances = []
    for track in tracks:
        members = [observations[index] for index in track]
        points = np.vstack([observation.points for observation in members])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        count = len(cloud.voxel_down_sample(VOXEL_SIZE_M).points) if len(points) else 0
        instances.append(CountedInstance(members, count))
    return instances
