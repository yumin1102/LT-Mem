from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ltmem.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from ltmem.pipeline import IntegratedPipeline

LOCATION_MERGE_DISTANCE_M = 0.3


def track_matches_object(track_object_id: str, obj_name: str) -> bool:
    name = re.sub(r"_Session[1-9][0-9]*_[0-9]+$", "", track_object_id)
    return name == obj_name


def _session_number(session_label: str) -> int:
    return int(session_label.removeprefix("Session").removeprefix("s"))


def _canonical_session_id(session_label: str) -> str:
    return f"s{_session_number(session_label)}"


def export_object_predictions(
    obj_name: str,
    sessions: Collection[str],
    positions_by_label: Dict[str, np.ndarray],
    regions: Dict[str, List[Dict[str, Any]]],
    pipeline: IntegratedPipeline,
    env: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    canonical_sessions = [_canonical_session_id(label) for label in sessions]
    positions = {
        _canonical_session_id(session): center for session, center in positions_by_label.items()
    }
    held_sessions = {
        _canonical_session_id(session) for session in sessions if pipeline.is_session_held(session)
    }
    state: Dict[str, Dict[str, Any]] = {}
    for session_id in canonical_sessions:
        if session_id not in positions:
            state[session_id] = {"present": 0, "loc_token": "ND", "centroid": None}
            continue
        centroid = np.asarray(positions[session_id], dtype=float)
        state[session_id] = {
            "present": 1,
            "loc_token": _location_token(centroid, regions[obj_name]),
            "centroid": centroid,
        }

    event_by_session: Dict[str, str] = {}
    for event in pipeline.get_events():
        if not track_matches_object(event.object_id, obj_name):
            continue
        session_id = _canonical_session_id(event.timestamp)
        event_type = event.event_type
        if session_id in event_by_session:
            raise RuntimeError(
                f"Duplicate canonical events for {env}/{session_id}/{obj_name}: "
                f"{event_by_session[session_id]} and {event_type}"
            )
        event_by_session[session_id] = event_type

    first_observed = min((_session_number(session) for session in positions), default=None)
    for session_id in canonical_sessions:
        if session_id in event_by_session:
            continue
        session_num = _session_number(session_id)
        if session_id in held_sessions:
            event_by_session[session_id] = "HOLD"
        elif first_observed is None or session_num < first_observed:
            event_by_session[session_id] = "NONE"
        else:
            raise RuntimeError(
                f"Missing canonical Delta event for {env}/{session_id}/{obj_name}; "
                "event inference in the exporter is disabled"
            )
    events = [
        {"env": env, "session": session, "object": obj_name, "event": event}
        for session, event in event_by_session.items()
    ]
    return state, events


def generate_regions(
    all_object_centroids: Dict[str, List[np.ndarray]],
) -> Dict[str, List[Dict[str, Any]]]:
    regions: Dict[str, List[Dict[str, Any]]] = {}
    for obj_name, centroids in all_object_centroids.items():
        if not centroids:
            continue
        location_history: List[np.ndarray] = []
        assigned_regions: Dict[str, List[np.ndarray]] = defaultdict(list)
        for centroid in centroids:
            centroid = np.asarray(centroid, dtype=float)
            loc_index = next(
                (
                    index
                    for index, previous in enumerate(location_history)
                    if float(np.linalg.norm(centroid - previous)) < LOCATION_MERGE_DISTANCE_M
                ),
                None,
            )
            if loc_index is None:
                loc_index = len(location_history)
                location_history.append(centroid)
            assigned_regions[f"L{loc_index}"].append(centroid)
        regions[obj_name] = [
            {
                "loc_token": token,
                "centroid": np.mean(values, axis=0).tolist(),
            }
            for token, values in assigned_regions.items()
        ]
    return regions


def _location_token(
    centroid: np.ndarray,
    object_regions: Sequence[Dict[str, Any]],
) -> str:
    centroid = np.asarray(centroid, dtype=float)
    return min(
        object_regions,
        key=lambda region: float(
            np.linalg.norm(centroid - np.asarray(region["centroid"], dtype=float))
        ),
    )["loc_token"]


def _sorted_prediction_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["session_num"] = frame["session"].str.replace("s", "").astype(int)
    return frame.sort_values(["object", "session_num"]).drop(columns=["session_num"])


def write_predictions(
    runtime: RuntimeConfig,
    pipeline: IntegratedPipeline,
    object_positions: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    env = runtime.env

    state_records: List[Dict[str, Any]] = []
    event_records: List[Dict[str, Any]] = []
    all_centroids = {
        obj_name: list(positions.values()) for obj_name, positions in object_positions.items()
    }
    regions = generate_regions(all_centroids)
    for obj_name, positions in object_positions.items():
        state, events = export_object_predictions(
            obj_name,
            runtime.sessions,
            positions,
            regions,
            pipeline,
            env,
        )
        state_records.extend(
            {
                "env": env,
                "session": session,
                "object": obj_name,
                **values,
            }
            for session, values in state.items()
        )
        event_records.extend(events)

    locations_output_path = output_dir / f"locations_{env}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(locations_output_path, "w", encoding="utf-8") as location_file:
        json.dump(regions, location_file, indent=2)

    state_frame = _sorted_prediction_frame(state_records).drop(columns=["centroid"])
    event_frame = _sorted_prediction_frame(event_records)
    suffix = "_lab_l" if env == "Lab-L" else ""
    state_path = output_dir / f"pred_state_ours{suffix}.csv"
    event_path = output_dir / f"pred_event_ours{suffix}.csv"
    state_frame.to_csv(state_path, index=False)
    event_frame.to_csv(event_path, index=False)
    print(f"✓ Saved state predictions: {state_path.resolve()}")
    print(f"✓ Saved event predictions: {event_path.resolve()}")
    print(f"✓ Saved locations: {locations_output_path.resolve()}")
