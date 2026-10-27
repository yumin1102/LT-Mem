import argparse
import os
from pathlib import Path
import runpy
import sys

import av
import cv2
import numpy as np
import yaml

from .contracts import validate_keyframe_threshold, validate_sim3, write_json
from mast3r_slam.config import load_config
from mast3r_slam.dataloader import MonocularDataset
import mast3r_slam.dataloader as dataloader
from mast3r_slam.lietorch_utils import as_SE3

ROOT = Path(os.environ["MAST3R_SLAM_DIR"])


def video_timestamps(path):
    with av.open(str(path), options={"ignore_editlist": "1"}) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        times = [
            float(frame.pts * frame.time_base)
            if frame.pts is not None and frame.time_base is not None
            else None
            for frame in container.decode(video=0)
        ]
        if not times:
            raise ValueError(f"Empty video: {path}")
        if all(value is None for value in times):
            rate = float(stream.average_rate) if stream.average_rate else 0.0
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"Video has no frame times or valid frame rate: {path}")
            times = [index / rate for index in range(len(times))]
        elif any(value is None for value in times):
            raise ValueError(f"Video has incomplete frame times: {path}")
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Video frame times must be finite and increasing: {path}")
        start = times[0]
        return [value - start for value in times]


class StoredVideoDataset(MonocularDataset):
    def __init__(self, source):
        super().__init__()
        self.dataset_path = Path(source)
        video_path = self.dataset_path / "rgb.mp4"
        self.timestamps = video_timestamps(video_path)
        self.count = len(self.timestamps)
        self.rgb_files = range(self.count)
        self.use_calibration = False
        self.save_results = False
        self.container = av.open(str(video_path), options={"ignore_editlist": "1"})
        self.stream = iter(self.container.decode(video=0))
        self.last_index = -1
        self.last_rgb = None

    def subsample(self, stride):
        if stride != 1:
            raise ValueError("Full-data runner prohibits input subsampling")

    def read_img(self, index):
        if index == self.last_index:
            return self.last_rgb
        if index != self.last_index + 1:
            raise ValueError(f"Nonsequential frame request: {self.last_index} -> {index}")
        try:
            frame = next(self.stream)
        except StopIteration as exc:
            raise RuntimeError(f"Video ended before scanned frame {index}") from exc
        self.last_rgb = frame.to_ndarray(format="rgb24")
        self.last_index = index
        if index % 100 == 0 or index + 1 == self.count:
            print(f"PROGRESS {self.dataset_path.name} {index + 1}/{self.count}", flush=True)
        return self.last_rgb

    def verify_eof(self):
        if self.last_index + 1 != self.count:
            raise RuntimeError("Not all source frames were consumed")
        try:
            next(self.stream)
        except StopIteration:
            return
        raise RuntimeError("Video has more frames than the initial scan")


def export_keyframes(keyframes, dataset, output):
    entries = []
    for index in range(len(keyframes)):
        frame = keyframes[index]
        source_id = frame.frame_id
        if not 0 <= source_id < dataset.count:
            raise ValueError("Keyframe source ID is out of range")
        timestamp = dataset.timestamps[source_id]
        folder = output / f"kf_{index:06d}"
        folder.mkdir(exist_ok=False)
        rgb = (frame.uimg.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        points = frame.X_canon.cpu().numpy()
        confidence = frame.get_average_conf().cpu().numpy().reshape(-1)
        pose = frame.T_WC.matrix().cpu().numpy()[0]
        validate_sim3(pose)
        if points.shape != (h * w, 3) or confidence.shape != (h * w,):
            raise ValueError("Image/pointmap pixel correspondence mismatch")
        valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
        if valid.sum() < 100:
            raise ValueError("Insufficient finite confident points")
        if not cv2.imwrite(str(folder / "image.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Image write failed")
        np.save(folder / "pose_4x4.npy", pose)
        np.savez_compressed(
            folder / "pointcloud_local.npz",
            points=points,
            colors=rgb.reshape(-1, 3),
            valid=valid,
        )
        rigid = as_SE3(frame.T_WC).data.numpy().reshape(-1)
        (folder / "pose_tum.txt").write_text(" ".join(map(str, [timestamp, *rigid])) + "\n")
        entries.append({"dir": folder.name, "frame_id": source_id, "timestamp": timestamp})
    if not entries or [e["frame_id"] for e in entries] != sorted(
        set(e["frame_id"] for e in entries)
    ):
        raise RuntimeError("Empty or duplicate/out-of-order keyframe export")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Run the unchanged upstream frontend with full stored-video input and dense export."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--match-frac-thresh",
        type=validate_keyframe_threshold,
        default=None,
        help="Override tracking.match_frac_thresh; larger values select keyframes more readily",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Outputs must not be written inside raw input")
    output.mkdir(parents=True, exist_ok=False)
    shm = os.statvfs("/dev/shm")
    if shm.f_bavail * shm.f_frsize < 1400 * 1024**2:
        raise RuntimeError("Original shared stores need more /dev/shm; recheck host remount")
    load_config(str(ROOT / "config/base.yaml"))
    dataset = StoredVideoDataset(source)
    cfg = yaml.safe_load((ROOT / "config/base.yaml").read_text())
    cfg["single_thread"] = True
    if args.match_frac_thresh is not None:
        cfg["tracking"]["match_frac_thresh"] = args.match_frac_thresh
    config_path = output / "frontend-config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    dataloader.load_dataset = lambda path: dataset
    os.chdir(ROOT)
    sys.argv = [
        str(ROOT / "main.py"),
        "--dataset",
        str(source),
        "--config",
        str(config_path),
        "--no-viz",
    ]
    namespace = None
    try:
        namespace = runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
        if namespace["backend"].exitcode != 0:
            raise RuntimeError("Original frontend backend process failed")
        if namespace["i"] != dataset.count:
            raise RuntimeError("Original frontend did not process every requested frame")
        states = namespace["states"]
        if list(states.global_optimizer_tasks) or states.reloc_sem.value:
            raise RuntimeError("Unresolved frontend optimization/relocalization queue")
        dataset.verify_eof()
        actual_threshold = namespace["config"]["tracking"]["match_frac_thresh"]
        if actual_threshold != cfg["tracking"]["match_frac_thresh"]:
            raise RuntimeError("Frontend did not use the requested keyframe threshold")
        entries = export_keyframes(namespace["keyframes"], dataset, output)
        write_json(
            output / "manifest.json",
            {
                "status": "complete",
                "frames_total": dataset.count,
                "frames_processed": dataset.count,
                "full_session_processed": True,
                "keyframes": entries,
            },
        )
        print(
            f"COMPLETE {source.name}: {dataset.count} frames, {len(entries)} keyframes", flush=True
        )
    finally:
        dataset.container.close()
        if namespace is not None:
            namespace["manager"].shutdown()


if __name__ == "__main__":
    main()
