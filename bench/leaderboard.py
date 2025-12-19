from __future__ import annotations

from dataclasses import dataclass
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
    entries.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    leaderboard["entries"] = entries
    leaderboard["updated"] = entry.get("timestamp")
    _write_json(path, leaderboard)
    return leaderboard


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
