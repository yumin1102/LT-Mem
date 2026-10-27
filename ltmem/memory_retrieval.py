"""Object-centric retrieval from a saved Tri-Memory snapshot."""

import os
from typing import Any, Dict
import re
from uuid import uuid4

import numpy as np

from ltmem.local_model_integrity import (
    assert_artifact_unchanged,
    capture_artifact_fingerprint,
    enforce_read_only_model_policy,
    verify_local_model,
)


COMPONENT_CUES = {
    "Live": r"\b(where|location|position|current|currently|now|present)\b",
    "Delta": r"\b(when|history|happened|appear\w*|disappear\w*|reappear\w*|mov\w*|displacement|distance|travel\w*|seen|sessions?|s\d+)\b",
    "Meta": r"\b(how many|how often|counts?|times|frequen\w*|most|least|volatil\w*|park\w*|occupancy|busiest|captur\w*|clock|time|date)\b",
}
GLOBAL_CUES = (
    r"\b(which|all|every|each|compare|most|least|objects?|landmarks?|stable|stability|busiest)\b"
)
LANDMARK_CUES = r"\b(landmarks?|stable|stability|navigation)\b"
METRIC_CUES = r"\b(how many|how often|counts?|times|frequen\w*|most|least|distance|displacement|how far|busiest)\b"
MIN_OBJECT_SIMILARITY = 0.7
SESSION = r"(?:sessions?|s)\s*(\d+)"
OBJECT_META_FIELDS = ("volatility_score", "event_count", "session_counts")


def _load_embedding():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    path, _ = verify_local_model(
        os.getenv("BGE_MODEL_PATH", ""),
        os.getenv("BGE_MODEL_SHA256", ""),
        os.getenv("BGE_MODEL_MANIFEST") or None,
        model_label="BGE model",
        path_env="BGE_MODEL_PATH",
        sha_env="BGE_MODEL_SHA256",
    )
    enforce_read_only_model_policy(path, specific_env="BGE_REQUIRE_READ_ONLY_MODEL")
    fingerprint = capture_artifact_fingerprint(path)
    embedding = HuggingFaceEmbedding(model_name=str(path), device="cpu")
    assert_artifact_unchanged(path, fingerprint, artifact_label="BGE model")
    return embedding


def object_name(record):
    return record.get("object_id", "").rsplit("_Session", 1)[0]


def mentioned_objects(question, names):
    remaining = question.casefold()
    selected = []
    for name in sorted(names, key=lambda name: (-len(name), name)):
        pattern = r"(?<!\w)" + re.escape(name.casefold()) + r"(?![a-z0-9_])"
        if re.search(pattern, remaining):
            selected.append(name)
            remaining = re.sub(pattern, " ", remaining)
    return selected, remaining


def query_components(question):
    if re.search(LANDMARK_CUES, question, re.I):
        return {"Live", "Delta", "Meta"}
    components = {key for key, cue in COMPONENT_CUES.items() if re.search(cue, question, re.I)}
    if "Live" in components or "Meta" in components:
        components.add("Delta")
    return components or {"Live", "Delta", "Meta"}


def session_interval(question, recorded_sessions=()):
    numbers = [int(value) for value in re.findall(SESSION, question, re.I)]
    ranges = re.findall(
        SESSION + r"\s*(?:-|–|~|to|through|and)\s*(?:(?:sessions?|s)\s*)?(\d+)",
        question,
        re.I,
    )
    for start, end in ranges:
        numbers.extend((int(start), int(end)))
    if len(set(numbers)) == 1 and recorded_sessions:
        number = numbers[0]
        if re.search(r"\b(after|since)\s+" + SESSION + r"\b", question, re.I):
            exclusive = bool(re.search(r"\bafter\s+" + SESSION, question, re.I))
            return number + exclusive, max(recorded_sessions)
        if re.search(r"\b(before|until|through)\s+" + SESSION + r"\b", question, re.I):
            exclusive = bool(re.search(r"\bbefore\s+" + SESSION, question, re.I))
            return min(recorded_sessions), number - exclusive
    return (min(numbers), max(numbers)) if numbers else None


class MemoryRetriever:
    def __init__(self, context):
        self.context = context
        names = {object_name(row) for key in ("Live", "Delta") for row in context.get(key, [])}
        for key in OBJECT_META_FIELDS:
            names.update(context.get("Meta", {}).get(key, {}))
        self.names = sorted(names - {""})
        self.collection = None
        self.embedding = None

    def _search(self, question, names, limit, *, require_similarity=False):
        import chromadb
        from chromadb.config import Settings

        if self.collection is None:
            self.embedding = _load_embedding()
            client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
            self.collection = client.create_collection(
                "qa_" + uuid4().hex,
                metadata={"hnsw:space": "cosine"},
            )
            documents = [f"Object: {name}" for name in self.names]
            self.collection.add(
                ids=self.names,
                documents=documents,
                metadatas=[{"object": name} for name in self.names],
                embeddings=self.embedding.get_text_embedding_batch(documents),
            )
        result = self.collection.query(
            query_embeddings=[self.embedding.get_query_embedding(question)],
            n_results=min(limit, len(names)),
            where={"object": {"$in": names}},
            include=["distances"],
        )
        return [
            name
            for name, distance in zip(result["ids"][0], result["distances"][0])
            if not require_similarity or 1.0 - distance >= MIN_OBJECT_SIMILARITY
        ]

    def retrieve(self, question):
        mentioned, query = mentioned_objects(question, self.names)
        components = query_components(query)
        global_query = bool(re.search(GLOBAL_CUES, query, re.I)) and (
            not mentioned or bool(re.search(r"\b(objects?|others?)\b", query, re.I))
        )
        candidates = self.names if global_query else mentioned or self.names
        limit = len(candidates) if mentioned or global_query else 5
        names = (
            self._search(
                question,
                candidates,
                limit,
                require_similarity=not (mentioned or global_query),
            )
            if candidates
            else []
        )
        selected = set(names)
        memory = {}
        for component in ("Live", "Delta"):
            if component in components:
                memory[component] = [
                    row for row in self.context.get(component, []) if object_name(row) in selected
                ]
        if "Meta" in components:
            memory["Meta"] = {
                key: {name: value for name, value in values.items() if name in selected}
                if key in OBJECT_META_FIELDS
                else values
                for key, values in self.context.get("Meta", {}).items()
            }
        metrics = numeric_evidence(query, memory, names)
        facts = object_facts(memory)
        metric_query = bool(re.search(METRIC_CUES, query, re.I))
        if metric_query and "Live" not in components:
            memory = {}
            facts = {}
        elif re.search(LANDMARK_CUES, query, re.I):
            memory = {}
            facts = {
                name: {
                    key: fact[key] for key in ("current_states", "event_counts", "volatility_score")
                }
                for name, fact in facts.items()
            }
        elif (
            not metric_query
            and re.search(COMPONENT_CUES["Live"], query, re.I)
            and session_interval(query) is None
        ):
            memory = {}
            last_seen = bool(re.search(r"\b(seen|last|previous|before)\b", query, re.I))
            facts = {
                name: {"latest_detection": fact["latest_detection"]}
                if last_seen
                else {"current_states": fact["current_states"]}
                for name, fact in facts.items()
            }
        return {
            "memory": memory,
            "object_facts": facts,
            "retrieval": {
                "objects": names,
                "components": sorted(components),
                "scope": "all_objects" if global_query else "object_candidates",
                "available_objects": self.names,
            },
            "numeric_evidence": metrics if metric_query else {},
        }


def numeric_evidence(question, memory, names):
    records = memory.get("Delta", [])
    sessions = {int(row["session"].removeprefix("Session")) for row in records}
    for counts in memory.get("Meta", {}).get("session_counts", {}).values():
        sessions.update(int(session.removeprefix("Session")) for session in counts)
    interval = session_interval(question, sessions)
    evidence = {}
    for name in names:
        history = [row for row in records if object_name(row) == name]
        selected = [
            row
            for row in history
            if interval is None
            or interval[0] <= int(row["session"].removeprefix("Session")) <= interval[1]
        ]
        counts = {event: 0 for event in ("APPEAR", "DISAPPEAR", "REAPPEAR", "MOVE", "NONE", "HOLD")}
        for row in selected:
            if row["event"] in counts:
                counts[row["event"]] += 1
        moves = [row for row in selected if row["event"] == "MOVE"]
        distances = [
            {
                "session": row["session"],
                "distance_m": float(
                    np.linalg.norm(
                        np.asarray(row["to_location"]) - np.asarray(row["from_location"])
                    )
                ),
            }
            for row in moves
            if row.get("from_location") is not None and row.get("to_location") is not None
        ]
        unique = len({row["session"] for row in selected}) == len(selected)
        net = None
        if interval is not None:
            endpoints = [
                [
                    row
                    for row in history
                    if row["session"] == f"Session{session}" and _has_observed_location(row)
                ]
                for session in interval
            ]
            if all(len(rows) == 1 for rows in endpoints):
                net = float(
                    np.linalg.norm(
                        np.asarray(endpoints[1][0]["to_location"])
                        - np.asarray(endpoints[0][0]["to_location"])
                    )
                )
        occupancy = memory.get("Meta", {}).get("session_counts", {}).get(name, {})
        occupancy = {
            session: count
            for session, count in occupancy.items()
            if interval is None
            or interval[0] <= int(session.removeprefix("Session")) <= interval[1]
        }
        evidence[name] = {
            "recorded_event_counts": counts if selected and unique else None,
            "accepted_move_distances_m": distances,
            "total_accepted_move_distance_m": sum(row["distance_m"] for row in distances)
            if selected and unique and len(distances) == len(moves)
            else None,
            "net_observed_displacement_m": net,
            "unresolved_sessions": [row["session"] for row in selected if row["event"] == "HOLD"],
            "occupancy": occupancy,
            "busiest_sessions": [
                session for session, count in occupancy.items() if count == max(occupancy.values())
            ],
        }
    move_counts = {
        name: item["recorded_event_counts"]["MOVE"]
        for name, item in evidence.items()
        if item["recorded_event_counts"] is not None
    }
    result = {
        "session_interval": interval,
        "objects": evidence,
        "most_recorded_moves": [
            name for name, count in move_counts.items() if count == max(move_counts.values())
        ],
    }
    if "Live" in memory:
        states = memory["Live"]
        result["current_present_count"] = (
            sum(state["present"] for state in states)
            if states and all(type(state.get("present")) is bool for state in states)
            else None
        )
    return result


def _has_observed_location(event: Dict[str, Any]) -> bool:
    if event.get("to_location") is None or event.get("event") not in {
        "APPEAR",
        "MOVE",
        "REAPPEAR",
        "NONE",
    }:
        return False
    return event.get("detection_status") == "PD"


def object_facts(context: Dict[str, Any]) -> Dict[str, Any]:
    names = {
        item["object_id"].rsplit("_Session", 1)[0]
        for item in [*context.get("Live", []), *context.get("Delta", [])]
        if "object_id" in item
    }
    for field in ("volatility_score", "event_count", "session_counts"):
        names.update(context.get("Meta", {}).get(field, {}))
    facts = {
        name: {
            "current_states": [],
            "latest_detection": None,
            "event_counts": {},
            "volatility_score": context.get("Meta", {}).get("volatility_score", {}).get(name),
            "session_counts": context.get("Meta", {}).get("session_counts", {}).get(name, {}),
        }
        for name in sorted(names)
    }
    for track in context.get("Live", []):
        if "object_id" in track:
            name = track["object_id"].rsplit("_Session", 1)[0]
            facts[name]["current_states"].append(
                {
                    key: track[key]
                    for key in ("track_id", "present", "current_location", "location_is_estimate")
                    if key in track
                }
            )
    for event in sorted(
        context.get("Delta", []), key=lambda item: int(item["session"].removeprefix("Session"))
    ):
        if "object_id" not in event:
            continue
        entry = facts[event["object_id"].rsplit("_Session", 1)[0]]
        counts = entry["event_counts"]
        counts[event["event"]] = counts.get(event["event"], 0) + 1
        if _has_observed_location(event):
            entry["latest_detection"] = {
                "session": event["session"],
                "location": event["to_location"],
                "event": event["event"],
                "track_id": event.get("track_id"),
            }
    return facts
