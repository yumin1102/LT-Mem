#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from ltmem.instance_counting import MIN_VOXEL_POINTS, VOXEL_SIZE_M, count_session_instances
from ltmem.memory_system import MemorySystem
from ltmem.memory_qa import save_qa_memory
from ltmem.runtime.artifacts import record_run


def run_parking(config: Path, output_dir: Path) -> MemorySystem:
    from ltmem.perception import MAST3RDataLoader, SAM3Segmenter

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    with record_run(output_dir, config, mode="parking"):
        data_root = (config.resolve().parent / raw["data_root"]).resolve()
        sessions = {session_id: folder for folder, session_id in raw["sessions"]}
        loader = MAST3RDataLoader(str(data_root), sessions)
        segmenter = SAM3Segmenter()
        memory = MemorySystem()
        memory.meta.session_times = raw["session_times"]
        rows = []
        instance_records = []
        for object_index, obj in enumerate(raw["objects"], 1):
            print(f"[Object {object_index}/{len(raw['objects'])}] {obj['name']}", flush=True)
            counts = memory.meta.session_counts.setdefault(obj["name"], {})
            for session_index, session_id in enumerate(sessions, 1):
                print(f"  [{session_index}/{len(sessions)}] {session_id}", flush=True)
                instances = count_session_instances(
                    loader, segmenter, session_id, obj["object_classes"]
                )
                minimum = obj.get("min_voxel_points", MIN_VOXEL_POINTS)
                count = sum(instance.voxel_points >= minimum for instance in instances)
                instance_records.append(
                    {
                        "object": obj["name"],
                        "session": session_id,
                        "min_voxel_points": minimum,
                        "instances": [
                            {
                                "id": index,
                                "voxel_points": instance.voxel_points,
                                "accepted": instance.voxel_points >= minimum,
                                "observations": [
                                    {
                                        "frame": observation.frame,
                                        "prompt": observation.prompt,
                                        "mask_index": observation.mask_index,
                                    }
                                    for observation in instance.observations
                                ],
                            }
                            for index, instance in enumerate(instances, 1)
                        ],
                    }
                )
                counts[session_id] = count
                rows.append({"object": obj["name"], "session": session_id, "count": count})
                print(f"{session_id} {obj['name']}: {count}")
        with (output_dir / "pred_count_ours_parking.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("object", "session", "count"))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "parking_instances.json").write_text(
            json.dumps({"voxel_size_m": VOXEL_SIZE_M, "sessions": instance_records}, indent=2),
            encoding="utf-8",
        )
        save_qa_memory(memory, output_dir / "qa_memory.json")
        return memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count parking objects in aligned sessions; save memory for ltmem.memory_qa."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/parking.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/parking"))
    args = parser.parse_args()
    run_parking(args.config, args.output_dir)


if __name__ == "__main__":
    main()
