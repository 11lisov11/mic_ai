from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step27_artifacts import STEP27_MOTOR_ACCEPTANCE_JSON, find_acceptance_json


MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)

ROOT_FILES = (
    "step28_ieee_summary.csv",
    "step28_ieee_summary.md",
    "package_manifest.json",
)

MODE_FILES = (
    "step27_per_seed_metrics.csv",
    "step27_stats_motor_controller.csv",
    "step27_final_pi_vs_foc_vs_mic.csv",
    STEP27_MOTOR_ACCEPTANCE_JSON,
    "step27_reproducibility.json",
    "step27_report.md",
)

DEFAULT_GUARDRAILS_POLICY = "paper/ieee_2026/guardrails_policy.json"


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _check_exists(path: Path) -> bool:
    return path.exists()


def _mark(ok: bool) -> str:
    return "[x]" if ok else "[ ]"


def _parse_motor_thresholds(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        token = str(part).strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        motor = str(key).strip().lower()
        if not motor:
            continue
        try:
            out[motor] = float(value)
        except Exception:
            continue
    return out


def _to_float(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except Exception:
        return float("nan")
    return value


def _load_guardrails_policy(path: Path) -> Dict[str, float]:
    payload = _read_json(path)
    raw = payload.get("motor_saving_thresholds_pct", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        motor = str(k).strip().lower()
        if not motor:
            continue
        try:
            out[motor] = float(v)
        except Exception:
            continue
    return out


def build_checklist(
    step28_dir: Path,
    *,
    ieee_root: Path | None = None,
    require_lock: bool = False,
    motor_saving_thresholds: Mapping[str, float] | None = None,
    guardrails_policy_path: str = "",
) -> str:
    thresholds = {str(k).strip().lower(): float(v) for k, v in dict(motor_saving_thresholds or {}).items()}
    lines: List[str] = []
    lines.append("# IEEE Final Checklist (Auto)")
    lines.append("")
    lines.append(f"- step28_dir: `{step28_dir}`")
    lines.append("")

    lines.append("## Core artifacts")
    core_ok = True
    for rel in ROOT_FILES:
        ok = _check_exists(step28_dir / rel)
        core_ok = core_ok and ok
        lines.append(f"- {_mark(ok)} `{rel}`")
    lines.append("")

    lines.append("## Mode artifacts")
    mode_ok = True
    acceptance_all_ok = True
    reproducibility_all_ok = True
    for mode in MODE_DIRS:
        mode_dir = step28_dir / mode
        lines.append(f"### {mode}")
        local_ok = True
        for rel in MODE_FILES:
            path = mode_dir / rel
            if rel == STEP27_MOTOR_ACCEPTANCE_JSON:
                path = find_acceptance_json(mode_dir)
                ok = path.exists()
                rendered_rel = f"{mode}/{path.name}"
            else:
                ok = _check_exists(path)
                rendered_rel = f"{mode}/{rel}"
            local_ok = local_ok and ok
            lines.append(f"- {_mark(ok)} `{rendered_rel}`")
        mode_ok = mode_ok and local_ok

        acc_path = find_acceptance_json(mode_dir)
        if acc_path.exists():
            acc = _read_json(acc_path)
            mean_pass = bool(acc.get("mean_pass", False))
            worst_pass = bool(acc.get("worst_case_pass", False))
            acc_ok = mean_pass and worst_pass
            acceptance_all_ok = acceptance_all_ok and acc_ok
            lines.append(f"- {_mark(acc_ok)} AIR56 acceptance: mean_pass={mean_pass}, worst_case_pass={worst_pass}")
        else:
            acceptance_all_ok = False
            lines.append("- [ ] AIR56 acceptance: missing")

        rep_path = mode_dir / "step27_reproducibility.json"
        if rep_path.exists():
            rep = _read_json(rep_path)
            stable = rep.get("stable_vs_previous", None)
            # `None` is acceptable for first frozen publication.
            rep_ok = (stable is None) or bool(stable)
            reproducibility_all_ok = reproducibility_all_ok and rep_ok
            lines.append(f"- {_mark(rep_ok)} reproducibility: stable_vs_previous={stable}, sha={rep.get('table_sha256', '')}")
        else:
            reproducibility_all_ok = False
            lines.append("- [ ] reproducibility: missing")
        lines.append("")

    lines.append("## Derived IEEE figures/tables")
    derived = step28_dir / "derived_ieee"
    derived_items = (
        "ieee_pi_foc_mic_stats.csv",
        "ieee_pi_foc_mic_stats.md",
        "fig_ieee_pi_foc_mic_power.png",
        "fig_ieee_pi_foc_mic_power.pdf",
        "fig_ieee_pi_foc_mic_power.svg",
    )
    derived_ok = True
    for rel in derived_items:
        ok = _check_exists(derived / rel)
        derived_ok = derived_ok and ok
        lines.append(f"- {_mark(ok)} `derived_ieee/{rel}`")
    lines.append("")

    lines.append("## Motor acceptance guardrails")
    motor_guardrails_ok = True
    if str(guardrails_policy_path).strip():
        lines.append(f"- policy: `{guardrails_policy_path}`")
    if thresholds:
        kv = ", ".join([f"{k}>={v:+.3f}%" for k, v in sorted(thresholds.items())])
        lines.append(f"- thresholds: `{kv}`")
    motor_summary_json = derived / "motor_tuning_acceptance_summary.json"
    if motor_summary_json.exists():
        lines.append(f"- [x] `derived_ieee/{motor_summary_json.name}`")
        try:
            payload = _read_json(motor_summary_json)
            rows = payload.get("rows", [])
            if not isinstance(rows, list) or not rows:
                motor_guardrails_ok = False
                lines.append("- [ ] motor summary rows are missing")
            else:
                for row in rows:
                    if not isinstance(row, dict):
                        motor_guardrails_ok = False
                        continue
                    motor = str(row.get("motor", "")).strip().lower()
                    acceptance_pass = bool(row.get("acceptance_pass", False))
                    threshold = float(thresholds.get(motor, 0.0))
                    mean_saving = _to_float(row, "avg_power_saving_pct_mean")
                    min_saving = _to_float(row, "avg_power_saving_pct_min")
                    threshold_ok = bool(
                        math.isfinite(mean_saving)
                        and math.isfinite(min_saving)
                        and mean_saving >= threshold
                        and min_saving >= threshold
                    )
                    row_ok = bool(acceptance_pass and threshold_ok)
                    motor_guardrails_ok = motor_guardrails_ok and row_ok
                    lines.append(
                        "- {} motor={} acceptance_pass={} saving_mean={:+.3f}% saving_min={:+.3f}% threshold={:+.3f}%".format(
                            _mark(row_ok),
                            motor if motor else "<unknown>",
                            acceptance_pass,
                            mean_saving if math.isfinite(mean_saving) else float("nan"),
                            min_saving if math.isfinite(min_saving) else float("nan"),
                            threshold,
                        )
                    )
        except Exception:
            motor_guardrails_ok = False
            lines.append("- [ ] motor acceptance summary parse error")
    else:
        motor_guardrails_ok = False
        lines.append(f"- [ ] `derived_ieee/{motor_summary_json.name}` missing")
    lines.append("")

    lines.append("## Passport")
    passport_dir = step28_dir / "passport"
    passport_json = passport_dir / "passport_compare_3motors.json"
    passport_csv = passport_dir / "passport_compare_3motors.csv"
    passport_md = passport_dir / "passport_compare_3motors.md"
    passport_present = passport_json.exists() and passport_csv.exists() and passport_md.exists()
    lines.append(f"- {_mark(passport_present)} `passport/passport_compare_3motors.(csv|md|json)`")
    passport_ok = True
    if passport_present:
        payload = _read_json(passport_json)
        failures = payload.get("failures", [])
        warnings = payload.get("warnings", [])
        fail_count = len(failures) if isinstance(failures, list) else 0
        warn_count = len(warnings) if isinstance(warnings, list) else 0
        passport_ok = fail_count == 0
        lines.append(f"- {_mark(passport_ok)} passport failures: {fail_count}")
        lines.append(f"- {_mark(True)} passport warnings: {warn_count}")
    else:
        # Optional in some smoke setups; do not block readiness.
        lines.append("- [x] passport checks skipped (artifacts missing)")
    lines.append("")

    lines.append("## Publication assets")
    publication_ok = True
    if ieee_root is not None and ieee_root.exists():
        manuscript = ieee_root / "manuscript.md"
        fig_dir = ieee_root / "fig"
        fig_items = (
            "fig1_mic_methodology.png",
            "fig2_pi_foc_mic_power.pdf",
            "fig3_air56_working_characteristics.pdf",
            "fig4_cross_motor_robustness.pdf",
            "fig5_training_to_foc.pdf",
        )
        manuscript_ok = manuscript.exists()
        publication_ok = publication_ok and manuscript_ok
        lines.append(f"- {_mark(manuscript_ok)} `manuscript.md`")
        for rel in fig_items:
            ok = (fig_dir / rel).exists()
            publication_ok = publication_ok and ok
            lines.append(f"- {_mark(ok)} `fig/{rel}`")
    else:
        lines.append("- [x] publication checks skipped (ieee_root missing)")
    lines.append("")

    lines.append("## Submission lock")
    lock_path = step28_dir / "submission_candidate_lock.json"
    lock_ok = True
    lock_required = bool(require_lock)
    if lock_path.exists():
        lines.append(f"- [x] `{lock_path.name}`")
        try:
            lock_payload = _read_json(lock_path)
            raw_lock_ok = bool(lock_payload.get("lock_ok", False))
            missing_required = lock_payload.get("required_files_missing", [])
            missing_required_count = len(missing_required) if isinstance(missing_required, list) else 0
            lines.append(
                f"- {_mark(raw_lock_ok)} lock_ok={raw_lock_ok}, required_files_missing={missing_required_count}"
            )
            lock_ok = raw_lock_ok
        except Exception:
            lines.append("- [ ] lock parse error")
            lock_ok = False
    else:
        lines.append(f"- {_mark(not lock_required)} `{lock_path.name}` missing")
        lock_ok = not lock_required
    lines.append("")

    lines.append("## Submission readiness")
    ready = (
        core_ok
        and mode_ok
        and reproducibility_all_ok
        and acceptance_all_ok
        and derived_ok
        and motor_guardrails_ok
        and passport_ok
        and publication_ok
        and lock_ok
    )
    lines.append(f"- {'[x]' if ready else '[ ]'} ready_for_submission: `{ready}`")
    lines.append("")
    if not ready:
        lines.append("### Blocking items")
        if not core_ok:
            lines.append("- missing core artifacts")
        if not mode_ok:
            lines.append("- missing mode artifacts")
        if not reproducibility_all_ok:
            lines.append("- reproducibility check failed")
        if not acceptance_all_ok:
            lines.append("- AIR56 acceptance gate is not fully satisfied")
        if not derived_ok:
            lines.append("- missing derived IEEE figures/tables")
        if not motor_guardrails_ok:
            lines.append("- motor acceptance guardrails failed")
        if not passport_ok:
            lines.append("- passport failures are present")
        if not publication_ok:
            lines.append("- manuscript/figure publication assets are incomplete")
        if not lock_ok:
            lines.append("- submission lock is missing or invalid")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auto checklist for IEEE step28 package.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument(
        "--guardrails-policy",
        default=DEFAULT_GUARDRAILS_POLICY,
        help="JSON policy with motor_saving_thresholds_pct.",
    )
    parser.add_argument(
        "--motor-saving-thresholds",
        default="",
        help="Comma-separated per-motor MIC saving thresholds in percent, e.g. 'air56:0.5,al31:0.0,ao2:0.05'.",
    )
    parser.add_argument("--require-lock", action="store_true", help="Require submission_candidate_lock.json and lock_ok=true.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when ready_for_submission=false.")
    args = parser.parse_args()

    step28_dir = Path(args.step28_dir).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md).strip() else (step28_dir / "FINAL_CHECKLIST_AUTO.md")

    ieee_root = Path(args.ieee_root).expanduser().resolve() if str(args.ieee_root).strip() else None
    policy_path = Path(str(args.guardrails_policy)).expanduser().resolve()
    policy_thresholds: Dict[str, float] = {}
    if policy_path.exists():
        policy_thresholds = _load_guardrails_policy(policy_path)
    cli_thresholds = _parse_motor_thresholds(str(args.motor_saving_thresholds))
    thresholds = dict(policy_thresholds)
    thresholds.update(cli_thresholds)
    if not thresholds:
        thresholds = {"air56": 0.5, "al31": 0.0, "ao2": 0.05}
    payload = build_checklist(
        step28_dir,
        ieee_root=ieee_root,
        require_lock=bool(args.require_lock),
        motor_saving_thresholds=thresholds,
        guardrails_policy_path=str(policy_path),
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(payload, encoding="utf-8")
    print(f"saved: {out_md}")

    if args.strict:
        text = payload.lower()
        if "ready_for_submission: `true`" not in text:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
