from __future__ import annotations

"""
Compare benchmark summaries against a baseline with tolerances.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _index_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = str(row.get("file_tag") or row.get("scenario") or "")
        if not key:
            continue
        out[key] = row
    return out


def compare_summaries(
    baseline: List[Dict[str, object]],
    current: List[Dict[str, object]],
    err_key: str = "mic_mean_err",
    power_key: str = "mic_p_el_pos",
    max_err_rel: float = 0.1,
    max_err_abs: float = 0.0,
    max_power_rel: float = 0.1,
    max_power_abs: float = 0.0,
    require_err_ok: bool = True,
) -> Tuple[bool, Dict[str, object]]:
    base_map = _index_rows(baseline)
    cur_map = _index_rows(current)
    failures: List[Dict[str, object]] = []

    for key, base_row in base_map.items():
        cur_row = cur_map.get(key)
        if cur_row is None:
            failures.append({"scenario": key, "reason": "missing_current"})
            continue

        if require_err_ok and not bool(cur_row.get("err_ok", True)):
            failures.append({"scenario": key, "reason": "err_ok_false"})

        base_err = float(base_row.get(err_key, 0.0))
        cur_err = float(cur_row.get(err_key, 0.0))
        err_limit = base_err * (1.0 + max_err_rel) + max_err_abs
        if cur_err > err_limit:
            failures.append(
                {
                    "scenario": key,
                    "reason": "err_increase",
                    "baseline": base_err,
                    "current": cur_err,
                    "limit": err_limit,
                }
            )

        base_power = float(base_row.get(power_key, 0.0))
        cur_power = float(cur_row.get(power_key, 0.0))
        power_limit = base_power * (1.0 + max_power_rel) + max_power_abs
        if cur_power > power_limit:
            failures.append(
                {
                    "scenario": key,
                    "reason": "power_increase",
                    "baseline": base_power,
                    "current": cur_power,
                    "limit": power_limit,
                }
            )

    ok = len(failures) == 0
    report = {
        "scenarios": len(base_map),
        "failures": failures,
        "err_key": err_key,
        "power_key": power_key,
        "max_err_rel": max_err_rel,
        "max_err_abs": max_err_abs,
        "max_power_rel": max_power_rel,
        "max_power_abs": max_power_abs,
        "require_err_ok": require_err_ok,
        "passed": ok,
    }
    return ok, report


def _load(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Summary file must contain a list.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare summary.json to baseline with tolerances.")
    parser.add_argument("--baseline", required=True, help="Baseline summary.json")
    parser.add_argument("--current", required=True, help="Current summary.json")
    parser.add_argument("--err-key", default="mic_mean_err")
    parser.add_argument("--power-key", default="mic_p_el_pos")
    parser.add_argument("--max-err-rel", type=float, default=0.1)
    parser.add_argument("--max-err-abs", type=float, default=0.0)
    parser.add_argument("--max-power-rel", type=float, default=0.1)
    parser.add_argument("--max-power-abs", type=float, default=0.0)
    parser.add_argument("--no-require-err-ok", dest="require_err_ok", action="store_false")
    parser.add_argument("--report", default=None, help="Write report JSON to path")
    parser.set_defaults(require_err_ok=True)
    args = parser.parse_args()

    baseline = _load(Path(args.baseline).expanduser().resolve())
    current = _load(Path(args.current).expanduser().resolve())
    ok, report = compare_summaries(
        baseline,
        current,
        err_key=str(args.err_key),
        power_key=str(args.power_key),
        max_err_rel=float(args.max_err_rel),
        max_err_abs=float(args.max_err_abs),
        max_power_rel=float(args.max_power_rel),
        max_power_abs=float(args.max_power_abs),
        require_err_ok=bool(args.require_err_ok),
    )

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
