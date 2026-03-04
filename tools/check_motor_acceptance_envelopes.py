from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _merge_rules(common: Dict[str, object], motor_specific: Dict[str, object], scenario: str) -> Dict[str, object]:
    base = dict(common.get(scenario, {}))
    override = dict(motor_specific.get(scenario, {}))
    base.update(override)
    return base


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(v: object) -> float:
    return float(pd.to_numeric(v, errors="coerce"))


def _eval_row(motor: str, seed: int, row: Dict[str, object], rules: Dict[str, object]) -> Tuple[List[Dict[str, object]], bool]:
    scenario = str(row.get("scenario", ""))
    checks: List[Dict[str, object]] = []

    def add_min(metric: str, limit_key: str) -> None:
        if limit_key not in rules:
            return
        val = _to_float(row.get(metric))
        lim = _to_float(rules.get(limit_key))
        ok = bool(val >= lim)
        checks.append(
            {
                "motor": motor,
                "seed": int(seed),
                "scenario": scenario,
                "metric": metric,
                "rule": f"{metric}>={limit_key}",
                "value": val,
                "limit": lim,
                "passed": ok,
            }
        )

    def add_max(metric: str, limit_key: str) -> None:
        if limit_key not in rules:
            return
        val = _to_float(row.get(metric))
        lim = _to_float(rules.get(limit_key))
        ok = bool(val <= lim)
        checks.append(
            {
                "motor": motor,
                "seed": int(seed),
                "scenario": scenario,
                "metric": metric,
                "rule": f"{metric}<={limit_key}",
                "value": val,
                "limit": lim,
                "passed": ok,
            }
        )

    add_min("power_saving_pct", "power_saving_pct_min")
    add_min("eta_gain_pct", "eta_gain_pct_min")
    add_max("current_peak_ratio", "current_peak_ratio_max")
    add_max("current_mean_ratio", "current_mean_ratio_max")
    if "mic_mean_err_max" in rules:
        add_max("mic_mean_err", "mic_mean_err_max")
    if "err_ok_required" in rules:
        req = _bool(rules.get("err_ok_required"))
        ok = bool(_bool(row.get("err_ok", False)) or (not req))
        checks.append(
            {
                "motor": motor,
                "seed": int(seed),
                "scenario": scenario,
                "metric": "err_ok",
                "rule": "err_ok_required",
                "value": bool(_bool(row.get("err_ok", False))),
                "limit": bool(req),
                "passed": ok,
            }
        )

    row_pass = all(bool(c["passed"]) for c in checks) if checks else True
    return checks, row_pass


def _render_md(summary_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Acceptance Envelope Check")
    lines.append("")
    lines.append("| motor | scenario | samples | pass_count | pass_rate | power_min | eta_min | peak_max | mean_max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            "| {motor} | {scenario} | {samples} | {pass_count} | {pass_rate:.3f} | {power_saving_pct_min:+.3f} | {eta_gain_pct_min:+.3f} | {current_peak_ratio_max:.3f} | {current_mean_ratio_max:.3f} |".format(
                **r
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check per-motor per-scenario MIC acceptance envelopes from step27 run traces.")
    parser.add_argument("--run-dir", required=True, help="Step27 run directory containing runs/<motor>/seed_*/mic_summary_rows.json.")
    parser.add_argument("--envelopes", default="config/acceptance_envelopes_3motors.json")
    parser.add_argument("--motors", default="air56,al31,ao2")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    env_path = Path(args.envelopes).expanduser().resolve()
    if not env_path.exists():
        raise FileNotFoundError(env_path)
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (run_dir / "acceptance_envelopes")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(env_path)
    common = dict(payload.get("common", {}))
    motors_cfg = dict(payload.get("motors", {}))
    motors = [m.strip().lower() for m in str(args.motors).split(",") if m.strip()]

    check_rows: List[Dict[str, object]] = []
    scenario_rows: List[Dict[str, object]] = []

    for motor in motors:
        seed_dirs = sorted((run_dir / "runs" / motor).glob("seed_*"))
        motor_cfg = dict(motors_cfg.get(motor, {}))
        for seed_dir in seed_dirs:
            seed = int(str(seed_dir.name).split("_")[-1])
            mic_json = seed_dir / "mic_summary_rows.json"
            if not mic_json.exists():
                continue
            rows = json.loads(mic_json.read_text(encoding="utf-8"))
            for row in rows:
                scenario = str(row.get("scenario", ""))
                rules = _merge_rules(common, motor_cfg, scenario)
                checks, row_pass = _eval_row(motor, seed, dict(row), rules)
                check_rows.extend(checks)
                scenario_rows.append(
                    {
                        "motor": motor,
                        "seed": seed,
                        "scenario": scenario,
                        "row_pass": bool(row_pass),
                        "power_saving_pct": _to_float(row.get("power_saving_pct")),
                        "eta_gain_pct": _to_float(row.get("eta_gain_pct")),
                        "current_peak_ratio": _to_float(row.get("current_peak_ratio")),
                        "current_mean_ratio": _to_float(row.get("current_mean_ratio")),
                    }
                )

    checks_df = pd.DataFrame(check_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    summary_rows: List[Dict[str, object]] = []
    if not scenario_df.empty:
        for (motor, scenario), part in scenario_df.groupby(["motor", "scenario"], dropna=False):
            samples = int(len(part))
            pass_count = int(pd.to_numeric(part["row_pass"], errors="coerce").sum())
            summary_rows.append(
                {
                    "motor": str(motor),
                    "scenario": str(scenario),
                    "samples": samples,
                    "pass_count": pass_count,
                    "pass_rate": float(pass_count / max(samples, 1)),
                    "power_saving_pct_min": float(pd.to_numeric(part["power_saving_pct"], errors="coerce").min()),
                    "eta_gain_pct_min": float(pd.to_numeric(part["eta_gain_pct"], errors="coerce").min()),
                    "current_peak_ratio_max": float(pd.to_numeric(part["current_peak_ratio"], errors="coerce").max()),
                    "current_mean_ratio_max": float(pd.to_numeric(part["current_mean_ratio"], errors="coerce").max()),
                }
            )

    checks_csv = out_dir / "acceptance_envelope_checks.csv"
    scenarios_csv = out_dir / "acceptance_envelope_scenarios.csv"
    summary_csv = out_dir / "acceptance_envelope_summary.csv"
    summary_json = out_dir / "acceptance_envelope_summary.json"
    summary_md = out_dir / "acceptance_envelope_summary.md"

    checks_df.to_csv(checks_csv, index=False)
    scenario_df.to_csv(scenarios_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    summary_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "envelopes": str(env_path),
        "rows_checks": int(len(checks_df)),
        "rows_scenarios": int(len(scenario_df)),
        "rows_summary": int(len(summary_rows)),
        "all_rows_pass": bool(scenario_df["row_pass"].all()) if not scenario_df.empty else False,
        "summary_rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(_render_md(summary_rows), encoding="utf-8")

    print(f"saved: {checks_csv}")
    print(f"saved: {scenarios_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
