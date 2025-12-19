from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def extract_conditions(meta: Mapping[str, object]) -> dict[str, object]:
    cfg = meta.get("config")
    if not isinstance(cfg, Mapping):
        cfg = {}
    return {
        "dt": cfg.get("dt"),
        "duration": cfg.get("duration"),
        "controller": cfg.get("controller"),
        "speed_profile": cfg.get("speed_profile"),
        "load_profile": cfg.get("load_profile"),
        "disturbance_profile": cfg.get("disturbance_profile"),
        "param_drift": cfg.get("param_drift"),
        "load_config": cfg.get("load_config"),
        "sensor": cfg.get("sensor"),
        "safety": cfg.get("safety"),
        "motor": cfg.get("motor"),
        "inverter": cfg.get("inverter"),
        "scalar_vf": cfg.get("scalar_vf"),
        "foc": cfg.get("foc"),
        "identification_enabled": bool(meta.get("identification")),
    }


def suite_conditions(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    cases: dict[str, dict[str, object]] = {}
    for case_dir in _iter_case_dirs(run_dir):
        meta_path = case_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        cases[case_dir.name] = extract_conditions(meta)
    suite_hash = _hash_payload(cases)
    case_hashes = {name: _hash_payload(payload) for name, payload in cases.items()}
    return {
        "suite_hash": suite_hash,
        "case_hashes": case_hashes,
        "cases": cases,
    }


def diff_suites(run_a: str | Path, run_b: str | Path) -> dict[str, object]:
    cond_a = suite_conditions(run_a)
    cond_b = suite_conditions(run_b)
    cases_a = cond_a.get("cases", {})
    cases_b = cond_b.get("cases", {})
    diffs: dict[str, object] = {}
    for case in sorted(set(cases_a) | set(cases_b)):
        if case not in cases_a:
            diffs[case] = {"missing_in_a": True}
            continue
        if case not in cases_b:
            diffs[case] = {"missing_in_b": True}
            continue
        case_diff = _diff_dict(cases_a[case], cases_b[case])
        if case_diff:
            diffs[case] = case_diff
    return {
        "suite_hash_a": cond_a.get("suite_hash"),
        "suite_hash_b": cond_b.get("suite_hash"),
        "case_differences": diffs,
    }


def compare_action_traces(
    run_a: str | Path,
    run_b: str | Path,
    case: str | None = None,
    max_steps: int = 200,
) -> dict[str, object]:
    trace_a = _load_action_trace(run_a, case)
    trace_b = _load_action_trace(run_b, case)
    limit = min(len(trace_a), len(trace_b), max_steps)
    max_vd = 0.0
    max_vq = 0.0
    for idx in range(limit):
        max_vd = max(max_vd, abs(float(trace_a[idx]["v_d"]) - float(trace_b[idx]["v_d"])))
        max_vq = max(max_vq, abs(float(trace_a[idx]["v_q"]) - float(trace_b[idx]["v_q"])))
    match = limit > 0 and max_vd == 0.0 and max_vq == 0.0
    return {
        "steps_compared": limit,
        "max_abs_diff_v_d": max_vd,
        "max_abs_diff_v_q": max_vq,
        "match": match,
    }


def _iter_case_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    candidates = [p for p in run_dir.iterdir() if p.is_dir()]
    case_dirs = []
    for p in sorted(candidates):
        name = p.name.lower()
        if name.startswith("identify"):
            continue
        if (p / "run_meta.json").exists():
            case_dirs.append(p)
    return case_dirs


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _diff_dict(a: Mapping[str, object], b: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    diffs: dict[str, object] = {}
    keys = set(a) | set(b)
    for key in sorted(keys):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in a:
            diffs[path] = {"a": None, "b": b.get(key)}
            continue
        if key not in b:
            diffs[path] = {"a": a.get(key), "b": None}
            continue
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, Mapping) and isinstance(vb, Mapping):
            nested = _diff_dict(va, vb, path)
            diffs.update(nested)
        elif va != vb:
            diffs[path] = {"a": va, "b": vb}
    return diffs


def _load_action_trace(run_dir: str | Path, case: str | None) -> list[Mapping[str, object]]:
    base = Path(run_dir)
    if case:
        base = base / case
    path = base / "action_trace.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Invalid action_trace.json format: {path}")
        return data

    npz_path = base / "timeseries.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"action_trace.json or timeseries.npz not found: {base}")
    import numpy as np

    data = np.load(npz_path)
    t = data.get("t", [])
    v_d = data.get("v_d", [])
    v_q = data.get("v_q", [])
    steps = min(len(t), len(v_d), len(v_q))
    return [
        {"t": float(t[idx]), "v_d": float(v_d[idx]), "v_q": float(v_q[idx])}
        for idx in range(steps)
    ]


def _main() -> None:
    parser = argparse.ArgumentParser(description="MIC_AI validation helpers")
    parser.add_argument("--run-a", required=True, help="testsuite run directory A")
    parser.add_argument("--run-b", required=True, help="testsuite run directory B")
    parser.add_argument("--case", default=None, help="case name for action trace comparison")
    parser.add_argument("--compare-actions", action="store_true", help="compare action traces instead of configs")
    parser.add_argument("--max-steps", type=int, default=200, help="max steps for action comparison")
    args = parser.parse_args()

    if args.compare_actions:
        result = compare_action_traces(args.run_a, args.run_b, case=args.case, max_steps=args.max_steps)
    else:
        result = diff_suites(args.run_a, args.run_b)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    _main()


__all__ = ["extract_conditions", "suite_conditions", "diff_suites", "compare_action_traces"]
