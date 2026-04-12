from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


STEP27_MOTOR_ACCEPTANCE_JSON = "step27_motor_acceptance.json"
STEP27_AIR56_ACCEPTANCE_JSON_LEGACY = "step27_air56_acceptance.json"


def acceptance_candidate_names() -> List[str]:
    return [
        STEP27_MOTOR_ACCEPTANCE_JSON,
        STEP27_AIR56_ACCEPTANCE_JSON_LEGACY,
    ]


def find_acceptance_json(mode_dir: Path) -> Path:
    for name in acceptance_candidate_names():
        path = mode_dir / name
        if path.exists():
            return path
    return mode_dir / STEP27_MOTOR_ACCEPTANCE_JSON


def existing_acceptance_jsons(mode_dir: Path) -> List[Path]:
    out: List[Path] = []
    for name in acceptance_candidate_names():
        path = mode_dir / name
        if path.exists():
            out.append(path)
    return out


def required_mode_artifacts_with_acceptance(base_names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for name in base_names:
        if name == STEP27_AIR56_ACCEPTANCE_JSON_LEGACY:
            out.append(STEP27_MOTOR_ACCEPTANCE_JSON)
        else:
            out.append(str(name))
    if STEP27_MOTOR_ACCEPTANCE_JSON not in out:
        out.append(STEP27_MOTOR_ACCEPTANCE_JSON)
    return out
