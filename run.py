#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from ltmem.runtime.config import load_runtime_config, seed_everything
from ltmem.runtime.runner import DEFAULT_RUNTIME_ARTIFACT_DIR, run_pipeline
from ltmem.runtime.results import write_predictions
from ltmem.memory_qa import save_qa_memory
from ltmem.runtime.artifacts import record_run

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def registration_environment():
    env = os.environ.copy()
    mr = Path(env.get("MRSCALEMASTER_ROOT", ROOT / "third_party/MR.ScaleMaster"))
    slam = mr / "MASt3R-SLAM"
    library_path = env.get("MRSCALEMASTER_LIBRARY_PATH", str(mr / "cpp/install/lib"))
    env.update(
        MRSCALEMASTER_ROOT=str(mr),
        MAST3R_SLAM_DIR=str(slam),
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, mr, slam, slam / "thirdparty/mast3r"))),
        LD_LIBRARY_PATH=os.pathsep.join(filter(None, (library_path, env.get("LD_LIBRARY_PATH")))),
    )
    return env


def run_raw(source, config, output, *, match_frac_thresh=None, seed=42):
    source, config, output = source.resolve(), config.resolve(), output.resolve()
    runtime = load_runtime_config(config, require_data=False)
    raw = yaml.safe_load(config.read_text())
    sessions = []
    for index, folder in enumerate(runtime.sessions.values(), 1):
        if folder != f"robot_{index}":
            raise ValueError("Raw pipeline config must use robot_1, robot_2, ... in order")
        choices = [
            p for p in source.glob("robot_*") if p.name[6:].isdigit() and int(p.name[6:]) == index
        ]
        if len(choices) != 1:
            raise ValueError(
                f"Expected one raw capture directory for {folder}, found {len(choices)}"
            )
        session = choices[0]
        if not (session / "rgb.mp4").is_file():
            raise FileNotFoundError(session / "rgb.mp4")
        sessions.append(session)
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the raw capture directory")
    env = registration_environment()
    python = env.get(
        "MRSCALEMASTER_PYTHON",
        str(Path(env["MRSCALEMASTER_ROOT"]) / ".venv/mrscalemaster/bin/python"),
    )
    with record_run(
        output, config, mode="raw", seed=seed, input=source, match_frac_thresh=match_frac_thresh
    ):
        keyframes = output / "keyframes"
        aligned = output / "aligned"
        for index, session in enumerate(sessions, 1):
            command = [
                python,
                "-m",
                "ltmem.registration.frontend",
                "--source",
                str(session),
                "--output",
                str(keyframes / f"robot_{index}"),
            ]
            if match_frac_thresh is not None:
                command += ["--match-frac-thresh", str(match_frac_thresh)]
            print(f"[Frontend {index}/{len(sessions)}] {session.name}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        print("[MR.ScaleMaster] Register and export aligned data", flush=True)
        subprocess.run(
            [
                python,
                "-m",
                "ltmem.registration.register",
                "--input",
                str(keyframes),
                "--output",
                str(output / "registration"),
                "--aligned-output",
                str(aligned),
                "--expected-sessions",
                str(len(sessions)),
                "--page-inactive-frames",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        raw["data_root"] = str(aligned)
        resolved = output / "runtime.yaml"
        resolved.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        print("[LT-Mem] Process aligned data and export events", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "run",
                "--config",
                str(resolved),
                "--output_dir",
                str(output / "events"),
                "--runtime_artifact_dir",
                str(output / "memory"),
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"COMPLETE: {output / 'events'}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LT-Mem on aligned sessions or raw RGB videos", allow_abbrev=False
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, help="Raw robot_*/rgb.mp4 root; omit for aligned data"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Aligned result directory or new raw run directory",
    )
    parser.add_argument(
        "--runtime_artifact_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rag_persist_dir",
        type=Path,
        default=None,
        help="Object search store path; its collection is reset at experiment startup",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-frac-thresh", type=float, help="Raw registration option")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.input is not None:
        if args.output_dir is None:
            parser.error("--output_dir is required with --input")
        if args.runtime_artifact_dir or args.rag_persist_dir:
            parser.error("Memory options are only supported for aligned input")
        run_raw(
            args.input,
            args.config,
            args.output_dir,
            match_frac_thresh=args.match_frac_thresh,
            seed=args.seed,
        )
        return
    if args.match_frac_thresh is not None:
        parser.error("Registration options require --input")
    seed_everything(args.seed)
    runtime = load_runtime_config(args.config, require_data=True)
    output_dir = args.output_dir or args.runtime_artifact_dir or DEFAULT_RUNTIME_ARTIFACT_DIR
    with record_run(
        output_dir,
        args.config,
        mode="aligned",
        seed=args.seed,
        runtime_artifact_dir=args.runtime_artifact_dir or output_dir,
        rag_persist_dir=args.rag_persist_dir,
    ):
        pipeline, positions = run_pipeline(
            runtime,
            runtime_artifact_dir=args.runtime_artifact_dir or output_dir,
            rag_persist_dir=args.rag_persist_dir,
        )
        write_predictions(runtime, pipeline, positions, output_dir)
        save_qa_memory(pipeline.memory, output_dir / "qa_memory.json")


if __name__ == "__main__":
    main()
