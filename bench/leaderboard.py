from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Mapping

import json


@dataclass(frozen=True)
class LeaderboardEntry:
    """Single leaderboard entry."""

    policy_id: str
    score: float
    testsuite_dir: str
    timestamp: str
    metrics: Mapping[str, float]
    cases: list[Mapping[str, object]]


def update_leaderboard(path: str | Path, entry: Mapping[str, object]) -> dict[str, object]:
    """Append an entry and keep leaderboard sorted by score."""
    path = Path(path)
    leaderboard = _load_leaderboard(path)
    entries = leaderboard.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    entries.append(entry)
    ref_hash = _select_reference_hash(entries)
    _annotate_conditions(entries, ref_hash)
    entries.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    leaderboard["entries"] = entries
    leaderboard["updated"] = entry.get("timestamp")
    leaderboard["conditions_reference"] = {"suite_hash": ref_hash}
    _write_json(path, leaderboard)
    return leaderboard


def _select_reference_hash(entries: list[Mapping[str, object]]) -> str | None:
    hashes = []
    for item in entries:
        suite_hash = _suite_hash(item)
        if suite_hash:
            hashes.append(suite_hash)
    if not hashes:
        return None
    counts = Counter(hashes)
    max_count = max(counts.values())
    candidates = [h for h, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    best_hash = None
    best_score = float("-inf")
    for candidate in candidates:
        candidate_scores = [
            float(item.get("score", 0.0))
            for item in entries
            if _suite_hash(item) == candidate
        ]
        candidate_best = max(candidate_scores) if candidate_scores else float("-inf")
        if candidate_best > best_score:
            best_score = candidate_best
            best_hash = candidate
    return best_hash


def _suite_hash(entry: Mapping[str, object]) -> str | None:
    conditions = entry.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    suite_hash = conditions.get("suite_hash")
    return str(suite_hash) if suite_hash else None


def _annotate_conditions(entries: list[Mapping[str, object]], ref_hash: str | None) -> None:
    for entry in entries:
        suite_hash = _suite_hash(entry)
        if ref_hash is None or suite_hash is None:
            entry["conditions_match"] = None
            entry["conditions_warning"] = "conditions_missing"
            entry["score_comparable"] = False
            continue
        match = suite_hash == ref_hash
        entry["conditions_match"] = match
        entry["score_comparable"] = match
        entry["conditions_warning"] = None if match else f"conditions_mismatch (ref: {ref_hash})"


def _load_leaderboard(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": "0.1", "updated": None, "entries": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


__all__ = ["LeaderboardEntry", "update_leaderboard"]
