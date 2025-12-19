from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import json
import math

import numpy as np

from bench.leaderboard import update_leaderboard
from bench.validation import suite_conditions


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for normalized metric components."""

    iae: float = 1.0
    overshoot: float = 0.5
    settling_time: float = 0.2
    i_rms: float = 0.5
    smoothness: float = 0.01
    energy_proxy: float = 0.2
    ripple: float = 0.2


@dataclass(frozen=True)
class ScoreConfig:
    """Scoring configuration."""

    weights: ScoreWeights = ScoreWeights()
    safety_penalty: float = 0.0
    base_score: float = 100.0
    eps: float = 1e-6


def score_testsuite(
    run_dir: str | Path,
    policy_id: str,
    leaderboard_path: str | Path = "leaderboard.json",
    score_config: ScoreConfig | None = None,
) -> dict[str, object]:
    """Score a testsuite run directory and update the leaderboard."""
    entry = build_score_entry(run_dir, policy_id, score_config)
    leaderboard_path = Path(leaderboard_path)
    update_leaderboard(leaderboard_path, entry)
    return {
        "score": entry["score"],
        "leaderboard_path": leaderboard_path,
        "entry": entry,
    }


def build_score_entry(
    run_dir: str | Path,
    policy_id: str,
    score_config: ScoreConfig | None = None,
) -> dict[str, object]:
    """Build a leaderboard entry without modifying the leaderboard."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"testsuite run_dir not found: {run_dir}")
    score_config = score_config or ScoreConfig()

    case_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("E")])
    if not case_dirs:
        raise ValueError(f"No case directories found in {run_dir}")

    case_results = []
    for case_dir in case_dirs:
        metrics, score = score_run(case_dir, score_config)
        case_results.append({"case": case_dir.name, "metrics": metrics, "score": score})

    suite_disqualified = any(case["metrics"].get("safety_trip", 0.0) > 0.0 for case in case_results)
    suite_score = float(np.mean([case["score"] for case in case_results]))
    if suite_disqualified:
        suite_score = float(score_config.safety_penalty)

    suite_metrics = _aggregate_metrics([case["metrics"] for case in case_results])
    suite_metrics["suite_score"] = suite_score
    suite_metrics["disqualified"] = float(suite_disqualified)

    conditions = suite_conditions(run_dir)
    entry = {
        "policy_id": policy_id,
        "score": suite_score,
        "testsuite_dir": str(run_dir),
        "timestamp": _now_stamp(),
        "metrics": suite_metrics,
        "cases": [{"case": c["case"], "score": c["score"]} for c in case_results],
        "conditions": {
            "suite_hash": conditions.get("suite_hash"),
            "case_hashes": conditions.get("case_hashes"),
        },
        "disqualified": suite_disqualified,
    }
    return entry


def score_run(run_dir: str | Path, score_config: ScoreConfig | None = None) -> tuple[dict[str, float], float]:
    """Score a single experiment run directory."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found in {run_dir}")
    meta = _read_json(meta_path)
    npz_path = Path(meta.get("npz_path", run_dir / "timeseries.npz"))
    if not npz_path.exists():
        raise FileNotFoundError(f"timeseries.npz not found: {npz_path}")

    dt = None
    config = meta.get("config") if isinstance(meta, Mapping) else None
    if isinstance(config, Mapping) and "dt" in config:
        dt = float(config["dt"])

    timeseries = _load_timeseries(npz_path)
    metrics = compute_metrics(timeseries, dt=dt)
    score_config = score_config or ScoreConfig()
    score = compute_score(metrics, score_config)

    meta["metrics"] = metrics
    meta["score"] = score
    meta["disqualified"] = bool(metrics.get("safety_trip", 0.0) > 0.0)
    _write_json(meta_path, meta)
    return metrics, score


def compute_metrics(timeseries: Mapping[str, np.ndarray], dt: float | None = None) -> dict[str, float]:
    """Compute tracking metrics from time series arrays."""
    required = (
        "t",
        "omega_m",
        "omega_ref",
        "i_d",
        "i_q",
        "i_a",
        "i_b",
        "i_c",
        "v_d",
        "v_q",
        "flag_safety",
    )
    missing = [key for key in required if key not in timeseries]
    if missing:
        raise KeyError(f"Missing timeseries keys: {missing}")

    t = np.asarray(timeseries["t"], dtype=float)
    omega_m = np.asarray(timeseries["omega_m"], dtype=float)
    omega_ref = np.asarray(timeseries["omega_ref"], dtype=float)
    i_d = np.asarray(timeseries["i_d"], dtype=float)
    i_q = np.asarray(timeseries["i_q"], dtype=float)
    i_a = np.asarray(timeseries["i_a"], dtype=float)
    i_b = np.asarray(timeseries["i_b"], dtype=float)
    i_c = np.asarray(timeseries["i_c"], dtype=float)
    v_d = np.asarray(timeseries["v_d"], dtype=float)
    v_q = np.asarray(timeseries["v_q"], dtype=float)
    flag_safety = np.asarray(timeseries["flag_safety"], dtype=float)

    dt_val = _infer_dt(t, dt)
    duration = _duration(t, dt_val)
    error = omega_ref - omega_m

    iae = float(np.sum(np.abs(error)) * dt_val)
    overshoot = float(np.max(np.maximum(0.0, np.abs(omega_m) - np.abs(omega_ref))))
    settling_time = float(_settling_time(t, error, omega_ref, dt_val))
    i_rms = float(np.sqrt(np.mean((i_a * i_a + i_b * i_b + i_c * i_c) / 3.0)))
    smoothness = float(_control_smoothness(v_d, v_q, dt_val))
    safety_trip = float(np.any(flag_safety > 0.5))
    energy_proxy = float(_mean_abs(1.5 * (v_d * i_d + v_q * i_q)))
    ripple = float(_current_ripple(i_d, i_q))

    ref_scale = float(max(np.max(np.abs(omega_ref)), 1.0))
    current_scale = float(max(np.max(np.abs(np.concatenate([i_a, i_b, i_c]))), 1.0))
    voltage_scale = float(max(np.max(np.hypot(v_d, v_q)), 1.0))
    power_scale = float(max(1.5 * voltage_scale * current_scale, 1.0))

    return {
        "iae": iae,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "i_rms": i_rms,
        "smoothness": smoothness,
        "safety_trip": safety_trip,
        "disqualified": float(safety_trip > 0.0),
        "energy_proxy": energy_proxy,
        "ripple": ripple,
        "dt": float(dt_val),
        "duration": float(duration),
        "ref_scale": ref_scale,
        "current_scale": current_scale,
        "voltage_scale": voltage_scale,
        "power_scale": power_scale,
    }


def compute_score(metrics: Mapping[str, float], config: ScoreConfig) -> float:
    """Compute scalar score from metrics."""
    if metrics.get("safety_trip", 0.0) > 0.0:
        return float(config.safety_penalty)

    w = config.weights
    eps = config.eps
    duration = max(metrics.get("duration", 0.0), metrics.get("dt", 0.0), eps)
    dt = max(metrics.get("dt", 0.0), eps)

    iae_norm = metrics["iae"] / (metrics["ref_scale"] * duration + eps)
    overshoot_norm = metrics["overshoot"] / (metrics["ref_scale"] + eps)
    settling_norm = metrics["settling_time"] / (duration + eps)
    i_rms_norm = metrics["i_rms"] / (metrics["current_scale"] + eps)
    smooth_norm = metrics["smoothness"] / (metrics["voltage_scale"] / dt + eps)
    energy_norm = metrics["energy_proxy"] / (metrics.get("power_scale", 1.0) + eps)
    ripple_norm = metrics["ripple"] / (metrics["current_scale"] + eps)

    cost = (
        w.iae * iae_norm
        + w.overshoot * overshoot_norm
        + w.settling_time * settling_norm
        + w.i_rms * i_rms_norm
        + w.smoothness * smooth_norm
        + w.energy_proxy * energy_norm
        + w.ripple * ripple_norm
    )
    score = config.base_score / (1.0 + cost)
    return float(score)


def _load_timeseries(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def _infer_dt(t: np.ndarray, fallback: float | None) -> float:
    if fallback is not None and fallback > 0.0:
        return float(fallback)
    if t.size > 1:
        diffs = np.diff(t)
        return float(np.median(diffs))
    return 0.0


def _duration(t: np.ndarray, dt: float) -> float:
    if t.size > 1:
        return float(t[-1] - t[0])
    return float(max((t.size - 1) * dt, 0.0))


def _settling_time(t: np.ndarray, error: np.ndarray, omega_ref: np.ndarray, dt: float) -> float:
    if t.size < 1:
        return 0.0
    ref_end = float(omega_ref[-1]) if omega_ref.size else 0.0
    band = 0.05 * max(abs(ref_end), 1.0)
    if error.size < 1:
        return 0.0
    violating = np.where(np.abs(error) > band)[0]
    if violating.size == 0:
        return 0.0
    last_idx = int(violating[-1])
    if t.size > last_idx:
        return float(t[-1] - t[last_idx])
    return float((t.size - 1 - last_idx) * dt)


def _control_smoothness(v_d: np.ndarray, v_q: np.ndarray, dt: float) -> float:
    if dt <= 0.0 or v_d.size < 2 or v_q.size < 2:
        return 0.0
    dv_d = np.diff(v_d) / dt
    dv_q = np.diff(v_q) / dt
    return float(np.sqrt(np.mean(dv_d * dv_d + dv_q * dv_q)))


def _mean_abs(values: np.ndarray) -> float:
    if values.size < 1:
        return 0.0
    return float(np.mean(np.abs(values)))


def _current_ripple(i_d: np.ndarray, i_q: np.ndarray) -> float:
    if i_d.size < 1 or i_q.size < 1:
        return 0.0
    i_mag = np.sqrt(i_d * i_d + i_q * i_q)
    i_mean = float(np.mean(i_mag))
    ripple = i_mag - i_mean
    return float(np.sqrt(np.mean(ripple * ripple)))


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = (
        "iae",
        "overshoot",
        "settling_time",
        "i_rms",
        "smoothness",
        "energy_proxy",
        "ripple",
        "safety_trip",
        "disqualified",
    )
    return {key: float(np.mean([m[key] for m in metrics_list])) for key in keys}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _now_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


__all__ = [
    "ScoreWeights",
    "ScoreConfig",
    "compute_metrics",
    "compute_score",
    "score_run",
    "score_testsuite",
    "build_score_entry",
]
