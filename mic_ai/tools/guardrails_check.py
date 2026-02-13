from __future__ import annotations

"""
Guardrails checker for scenario_compare summaries.

Reads summary.json (or summary_v3.json) and enforces:
- speed error not worse than FOC (err_ok)
- minimum power saving percentage
- optional minimum eta gain
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_summary(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("summary file must contain a list")
    return data


def evaluate_summary(
    rows: List[Dict[str, object]],
    min_power_saving_pct: float = 0.0,
    min_eta_gain_pct: float | None = None,
    max_err_failures: int = 0,
    require_err_ok: bool = True,
) -> Tuple[bool, Dict[str, object]]:
    failures = 0
    worst_power = 1e9
    worst_eta = 1e9
    for row in rows:
        err_ok = bool(row.get("err_ok", False))
        if require_err_ok and not err_ok:
            failures += 1
        power_saving = float(row.get("power_saving_pct", 0.0))
        if power_saving < min_power_saving_pct:
            failures += 1
        worst_power = min(worst_power, power_saving)

        if min_eta_gain_pct is not None:
            eta_gain = float(row.get("eta_gain_pct", 0.0))
            if eta_gain < float(min_eta_gain_pct):
                failures += 1
            worst_eta = min(worst_eta, eta_gain)

    ok = failures <= int(max_err_failures)
    report = {
        "scenarios": len(rows),
        "failures": failures,
        "max_err_failures": int(max_err_failures),
        "min_power_saving_pct": float(min_power_saving_pct),
        "worst_power_saving_pct": float(worst_power if worst_power < 1e9 else 0.0),
        "min_eta_gain_pct": float(min_eta_gain_pct) if min_eta_gain_pct is not None else None,
        "worst_eta_gain_pct": float(worst_eta if worst_eta < 1e9 else 0.0),
        "require_err_ok": bool(require_err_ok),
        "passed": bool(ok),
    }
    return ok, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrails check for scenario_compare summary.json")
    parser.add_argument("--summary", default="outputs/scenario_compare/summary.json", help="Path to summary.json")
    parser.add_argument("--min-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--min-eta-gain-pct", type=float, default=None)
    parser.add_argument("--max-err-failures", type=int, default=0)
    parser.add_argument("--no-require-err-ok", dest="require_err_ok", action="store_false")
    parser.add_argument("--write-report", default=None, help="Write report JSON to path")
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(require_err_ok=True)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    rows = load_summary(summary_path)
    ok, report = evaluate_summary(
        rows,
        min_power_saving_pct=float(args.min_power_saving_pct),
        min_eta_gain_pct=args.min_eta_gain_pct,
        max_err_failures=int(args.max_err_failures),
        require_err_ok=bool(args.require_err_ok),
    )

    if args.write_report:
        Path(args.write_report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        status = "PASS" if ok else "FAIL"
        print(f"[guardrails] {status} | scenarios={report['scenarios']} failures={report['failures']}")
        print(json.dumps(report, indent=2))

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
