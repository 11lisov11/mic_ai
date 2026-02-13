from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _nearest(value: float, grid: List[float]) -> float:
    if not grid:
        return float(value)
    lo, hi = _bounds(value, grid)
    if abs(value - lo) <= abs(hi - value):
        return float(lo)
    return float(hi)


def _bounds(value: float, grid: List[float]) -> tuple[float, float]:
    if not grid:
        return float(value), float(value)
    idx = bisect_left(grid, value)
    if idx <= 0:
        return float(grid[0]), float(grid[0])
    if idx >= len(grid):
        return float(grid[-1]), float(grid[-1])
    return float(grid[idx - 1]), float(grid[idx])


def _parse_key(key: object) -> Tuple[float, float] | None:
    if isinstance(key, tuple) and len(key) == 2:
        return float(key[0]), float(key[1])
    if isinstance(key, str) and "|" in key:
        left, right = key.split("|", 1)
        return float(left), float(right)
    return None


class IdRefLut:
    def __init__(self, omega_grid: Iterable[float], load_grid: Iterable[float], lut: Dict[object, float]):
        self.omega_grid = sorted(float(x) for x in omega_grid)
        self.load_grid = sorted(float(x) for x in load_grid)
        parsed: Dict[Tuple[float, float], float] = {}
        for key, value in dict(lut).items():
            parsed_key = _parse_key(key)
            if parsed_key is None:
                continue
            parsed[parsed_key] = float(value)
        self.lut = parsed

    @classmethod
    def from_json(cls, path: str | Path) -> "IdRefLut":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            omega_grid=[float(x) for x in data.get("omega_ref_grid", [])],
            load_grid=[float(x) for x in data.get("load_grid", [])],
            lut={str(k): float(v) for k, v in data.get("lut", {}).items()},
        )

    def query(self, omega_ref: float, load_torque: float) -> float:
        omega_val = float(omega_ref)
        load_val = float(load_torque)
        omega_lo, omega_hi = _bounds(omega_val, self.omega_grid)
        load_lo, load_hi = _bounds(load_val, self.load_grid)

        def lookup(omega_key: float, load_key: float) -> float | None:
            return self.lut.get((float(omega_key), float(load_key)))

        v00 = lookup(omega_lo, load_lo)
        v10 = lookup(omega_hi, load_lo)
        v01 = lookup(omega_lo, load_hi)
        v11 = lookup(omega_hi, load_hi)

        if v00 is None or v10 is None or v01 is None or v11 is None:
            omega_sel = _nearest(omega_val, self.omega_grid)
            load_sel = _nearest(load_val, self.load_grid)
            return float(self.lut.get((float(omega_sel), float(load_sel)), 0.0))

        if omega_hi == omega_lo and load_hi == load_lo:
            return float(v00)
        if omega_hi == omega_lo:
            denom = (load_hi - load_lo) or 1.0
            u = (load_val - load_lo) / denom
            return float(v00 + u * (v01 - v00))
        if load_hi == load_lo:
            denom = (omega_hi - omega_lo) or 1.0
            t = (omega_val - omega_lo) / denom
            return float(v00 + t * (v10 - v00))

        t = (omega_val - omega_lo) / (omega_hi - omega_lo)
        u = (load_val - load_lo) / (load_hi - load_lo)
        return float(
            (1.0 - t) * (1.0 - u) * v00
            + t * (1.0 - u) * v10
            + (1.0 - t) * u * v01
            + t * u * v11
        )


__all__ = ["IdRefLut"]
