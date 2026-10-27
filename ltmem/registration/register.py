import argparse
import itertools
import json
import os
from pathlib import Path
import shutil

import cv2
import lietorch
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from .contracts import loader_scale, selected_global_pose, validate_sim3
from cores.dataloader import load_keyframe
from cores.slam_backend import SLAMBackend

ROOT = Path(os.environ["MRSCALEMASTER_ROOT"])


def frame_to(frame, device):
    for name in ("img", "img_shape", "img_true_shape", "X_canon", "C", "feat", "pos", "K"):
        value = getattr(frame, name, None)
        if isinstance(value, torch.Tensor):
            setattr(frame, name, value.to(device))
    frame.T_WC = lietorch.Sim3(frame.T_WC.data.to(device))


class PagedBackend(SLAMBackend):
    def compute_sim3_edge(self, query_frame, target_frame):
        frame_to(query_frame, self.device)
        frame_to(target_frame, self.device)
        try:
            return super().compute_sim3_edge(query_frame, target_frame)
        finally:
            if target_frame is not query_frame:
                frame_to(target_frame, "cpu")

    def process_frame(self, *args, **kwargs):
        result = super().process_frame(*args, **kwargs)
        frame_to(self.global_frame_list[-1]["frame"], "cpu")
        return result


def connected_components(robot_ids, constraints):
    adjacency = {rid: set() for rid in robot_ids}
    for edge in constraints:
        a, b = edge["query_robot"], edge["target_robot"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(robot_ids)
    while unseen:
        todo, component = [min(unseen)], set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(adjacency[node] - component)
        unseen -= component
        components.append(sorted(component))
    return components


def load_frontend_sessions(source, expected_sessions):
    frames = {}
    for folder in sorted(source.glob("robot_*")):
        rid = int(folder.name.split("_")[-1])
        manifest = json.loads((folder / "manifest.json").read_text())
        frames_processed = manifest.get("frames_processed")
        frames_total = manifest.get("frames_total")
        complete = (
            manifest.get("status") == "complete"
            and manifest.get("full_session_processed") is True
            and isinstance(frames_processed, int)
            and not isinstance(frames_processed, bool)
            and isinstance(frames_total, int)
            and not isinstance(frames_total, bool)
            and frames_processed >= 0
            and frames_processed == frames_total
        )
        if not complete:
            raise ValueError(f"Incomplete frontend session: {folder}")
        entries = manifest["keyframes"]
        if not entries:
            raise ValueError("Empty keyframe session")
        frames[rid] = []
        resolved_folder = folder.resolve()
        for entry in entries:
            bundle = (folder / entry["dir"]).resolve()
            if not bundle.is_relative_to(resolved_folder):
                raise ValueError("Keyframe bundle escapes session directory")
            frames[rid].append((bundle, entry))
    if set(frames) != set(range(1, expected_sessions + 1)):
        raise ValueError("Missing/extra robot sessions")
    return frames


def export_aligned(backend, graph, bundles_by_gid, counts, output):
    output.mkdir(parents=True, exist_ok=False)
    for rid in counts:
        robot = output / f"robot_{rid}"
        for part in ("image", "ply", "pose", "timestamp"):
            (robot / part).mkdir(parents=True)
    for record in backend.global_frame_list:
        rid, idx, gid = record["robot_id"], record["g2o_idx"], record["gid"]
        folder, entry = bundles_by_gid[gid]
        global_pose = selected_global_pose(graph[rid], idx, counts[rid])
        original_pose = np.load(folder / "pose_4x4.npy", allow_pickle=False)
        factor = loader_scale(original_pose)
        image = cv2.imread(str(folder / "image.png"))
        if image is None:
            raise ValueError("Missing keyframe image")
        h, w = image.shape[:2]
        with np.load(folder / "pointcloud_local.npz", allow_pickle=False) as data:
            points = data["points"].reshape(-1, 3).astype(np.float32) * factor
            valid = data["valid"].reshape(-1).astype(bool) & np.isfinite(points).all(axis=1)
        if points.shape != (h * w, 3) or not valid.any():
            raise ValueError("Dense export shape/validity failure")
        points[~valid] = np.nan
        robot = output / f"robot_{rid}"
        name = f"{record['seq_id']:06d}"
        shutil.copyfile(folder / "image.png", robot / "image" / f"{name}.png")
        np.save(robot / "ply" / f"{name}.npy", points)
        np.savetxt(robot / "pose" / f"{name}.txt", global_pose, fmt="%.12g")
        (robot / "timestamp" / f"{name}.txt").write_text(f"{entry['timestamp']:.12f}\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Original MR registration plus dense camera-local export; no upstream edits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned-output", type=Path)
    parser.add_argument("--expected-sessions", type=int, default=10)
    parser.add_argument("--page-inactive-frames", action="store_true")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    aligned_output = args.aligned_output.resolve() if args.aligned_output is not None else None
    if aligned_output is not None and aligned_output.exists():
        raise FileExistsError("Refusing to overwrite aligned dataset")
    if output.exists():
        raise FileExistsError("Refusing to overwrite registration output")
    output.mkdir(parents=True)
    frames = load_frontend_sessions(source, args.expected_sessions)
    os.chdir(ROOT)
    cfg = yaml.safe_load((ROOT / "config/default.yaml").read_text())
    cfg["paths"]["loop_vis_save_dir"] = str(output / "loop_closures")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cls = PagedBackend if args.page_inactive_frames else SLAMBackend
    backend = cls(str(output / "config.yaml"))
    backend.visualize_loop_pair = lambda *args, **kwargs: None
    backend.model.eval()
    if not backend.keyframes.uimg.is_shared():
        raise RuntimeError("Original shared keyframe store not preserved")
    constraints = []
    bundles_by_gid = {}
    counts = {rid: len(items) for rid, items in frames.items()}
    total = sum(counts.values())
    streams = [[(rid, *item) for item in frames[rid]] for rid in sorted(frames)]
    for batch in itertools.zip_longest(*streams):
        for item in batch:
            if item is None:
                continue
            rid, folder, entry = item
            loaded = load_keyframe(str(folder), str(rid))
            if loaded is None:
                raise ValueError("Missing/unreadable keyframe bundle")
            _, bgr, pose, points, colors, timestamp = loaded
            validate_sim3(pose)
            if (
                points is None
                or colors is None
                or len(points) < 100
                or not np.isfinite(points).all()
            ):
                raise ValueError(f"Invalid/empty filtered cloud: {folder}")
            spread = np.median(np.linalg.norm(points - points.mean(axis=0), axis=1))
            if not np.isfinite(spread) or spread < 1e-6:
                raise ValueError("Degenerate cloud cannot yield scale ratio")
            loops, _ = backend.process_frame(
                rid, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pose, points, colors, timestamp
            )
            current = backend.global_frame_list[-1]
            gid = current["gid"]
            if current["g2o_idx"] != current["seq_id"] + 1:
                raise RuntimeError("Unexpected optimizer removal/index change")
            bundles_by_gid[gid] = (folder, entry)
            for match in loops:
                target = backend.global_frame_list[match["idx"]]
                constraints.append({"query_robot": rid, "target_robot": target["robot_id"]})
            if loops:
                backend.optimize_and_get_graph()
            processed = len(bundles_by_gid)
            if processed % 25 == 0 or processed == total:
                print("REGISTER", processed, "/", total, flush=True)
    if len(bundles_by_gid) != total:
        raise RuntimeError("Incomplete keyframe coverage")
    graph = backend.optimize_and_get_graph()
    for rid, traj in graph.items():
        for idx in range(1, counts[rid] + 1):
            selected_global_pose(traj, idx, counts[rid])
    components = connected_components(frames, constraints)
    if len(components) != 1:
        raise RuntimeError(f"Disconnected sessions {components}; aligned export prohibited")
    if aligned_output is not None:
        export_aligned(backend, graph, bundles_by_gid, counts, aligned_output)
    print(f"REGISTRATION COMPLETE: {len(frames)} sessions, {total} keyframes", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
