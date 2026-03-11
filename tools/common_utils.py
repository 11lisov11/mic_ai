from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in parse_csv_list(text)]


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def json_load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(row_list)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_rmtree(
    path: Path,
    *,
    repo_root: Path,
    allowed_leaf_names: Sequence[str] | None = None,
    min_relative_depth: int = 1,
) -> bool:
    """
    Remove directory tree with strict guardrails.

    Safety checks:
    - target must exist and be a directory;
    - target must be inside repo_root;
    - target must not be repo_root itself;
    - relative depth from repo_root must be at least min_relative_depth;
    - optional leaf-name allowlist.
    """
    target = Path(path).resolve()
    root = Path(repo_root).resolve()
    if not target.exists():
        return False
    if not target.is_dir():
        raise ValueError(f"safe_rmtree expects directory path, got: {target}")
    if target == root:
        raise ValueError(f"Refuse to remove repository root: {target}")
    try:
        rel = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refuse to remove path outside repository root: {target}") from exc
    if len(rel.parts) < int(max(min_relative_depth, 1)):
        raise ValueError(f"Refuse to remove shallow path: {target} (relative depth={len(rel.parts)})")
    if allowed_leaf_names:
        allow = {str(x) for x in allowed_leaf_names}
        if target.name not in allow:
            raise ValueError(f"Refuse to remove unexpected leaf name '{target.name}': {target}")
    shutil.rmtree(target)
    return True
