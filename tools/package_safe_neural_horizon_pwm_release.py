from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_safe_neural_horizon_pwm_report import build_report
from tools.build_safe_neural_horizon_pwm_figures import build_figures
from tools.check_safe_neural_horizon_pwm_release import analyze_release


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _article_draft(payload: Dict[str, Any]) -> str:
    scenarios = list(payload.get("scenarios", []))
    lines: List[str] = []
    lines.append("# Safe Neural Horizon PWM with Event-Triggered Twin Feedback")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "This draft describes a host-simulated induction-motor control variant that combines a neural cost shaper, "
        "short-horizon inverter-vector search, an event-triggered neural twin, and a protected AI-PWM Safety Gateway. "
        "The method is evaluated only in software simulation in this release. No MCU, HIL, or bench claim is made."
    )
    lines.append("")
    lines.append("## Contribution")
    lines.append("")
    lines.append("- Alpha-beta induction-motor model with parameter randomization hooks.")
    lines.append("- Two-level inverter model with legal vector set, dead-time proxy, loss proxy, and common-mode proxy.")
    lines.append("- Safety Gateway that prevents direct AI access to raw high/low gate commands.")
    lines.append("- Host-tested no-shoot-through timing waveform invariant for vector transitions.")
    lines.append("- Horizon AI-PWM controller with neural cost shaping and event-triggered feedback policy.")
    lines.append("- Scenario matrix, ablation smoke, Pareto extraction, and fault-injection summary.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("The AI layer requests only `vector_id in {0..7}`. The gateway maps accepted vectors to gate states and inserts BOTH_OFF dead-time states on changing legs. Unsafe requests are rejected, held, or latched depending on fault severity.")
    lines.append("")
    lines.append("The optimization cost includes speed error, torque error, current stress, flux building, torque-ripple proxy, switching events, loss proxy, thermal proxy, feedback usage, confidence/risk, and common-mode proxy.")
    lines.append("")
    lines.append("## Evaluation")
    lines.append("")
    lines.append(f"- Status: `{payload.get('status', 'unknown')}`")
    lines.append(f"- Hardware claim: `{bool(payload.get('hardware_claim', False))}`")
    lines.append(f"- MC trials: `{payload.get('mc_trials', 0)}`")
    lines.append(f"- Steps per trial: `{payload.get('steps_per_trial', 0)}`")
    lines.append(f"- Scenarios: `{len(scenarios)}`")
    lines.append("")
    if scenarios:
        lines.append("Scenario list:")
        for scenario in scenarios:
            lines.append(f"- `{scenario}`")
        lines.append("")
    fault = dict(payload.get("fault_injection", {}))
    if fault:
        lines.append("Fault-injection result:")
        lines.append(f"- all_gateway_cases_no_shoot_through: `{bool(fault.get('all_gateway_cases_no_shoot_through', False))}`")
        lines.append(f"- raw_shoot_through_detector_triggered: `{bool(fault.get('raw_shoot_through_detector_triggered', False))}`")
        lines.append("")
    lines.append("## Preliminary Findings")
    lines.append("")
    lines.append("- H2 is the safer current research candidate than the sparse H4 variant in the short host matrix.")
    lines.append("- Sparse H4 can reduce feedback and switching, but current stress and fallback events increase in several scenarios.")
    lines.append("- One-step FCS proxy tends to keep current lower but uses dense feedback and more switching.")
    lines.append("- Current proxy baselines are useful for smoke testing but are not publication-grade strong baselines.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Host simulation only.")
    lines.append("- Proxy baselines, not final tuned FOC-SVM/FCS-MPC/DTC-SVM implementations.")
    lines.append("- No trained domain-randomized neural twin yet.")
    lines.append("- No fixed-point/WCET analysis.")
    lines.append("- No MCU, HIL, oscilloscope, inverter, or motor-bench validation.")
    lines.append("")
    lines.append("## Required Next Work")
    lines.append("")
    lines.append("- Replace proxy baselines with tuned strong baselines.")
    lines.append("- Run publication-scale MC after baseline replacement.")
    lines.append("- Add publication-grade plots and FFT/THD metrics.")
    lines.append("- Port the safety gateway and timing checks to the target MCU/HIL path.")
    lines.append("- Validate gate timing and current trips on real hardware before any hardware-ready claim.")
    lines.append("")
    return "\n".join(lines)


def _open_items() -> str:
    return "\n".join(
        [
            "# Safe Neural Horizon PWM Open Items",
            "",
            "- Replace host proxy baselines with tuned key-level FOC-SVM, FCS-MPC, DTC, DTC-SVM, deadbeat, and sensorless/adaptive FOC.",
            "- Add publication-grade long-run metrics: THD, FFT torque, switching loss, conduction loss, thermal imbalance, EMI/common-mode proxy.",
            "- Run MC=500..1000 after strong baselines are ready.",
            "- Train or identify the neural twin with domain randomization and multi-step losses.",
            "- Add fixed-point or bounded floating-point MCU implementation plus WCET.",
            "- Add HIL, oscilloscope gate timing, current trip, watchdog, and bench validation.",
            "- Do not claim hardware-ready status until real MCU/HIL/bench evidence exists.",
            "",
        ]
    )


def package_release(input_json: Path, out_dir: Path, tag: str) -> Dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_json = out_dir / "safe_neural_horizon_pwm_results.json"
    shutil.copyfile(input_json, copied_json)
    report_md = out_dir / "safe_neural_horizon_pwm_report.md"
    article_md = out_dir / "safe_neural_horizon_pwm_article_draft.md"
    open_items_md = out_dir / "WHAT_IS_NOT_DONE.md"
    acceptance_json = out_dir / "HOST_ACCEPTANCE_SUMMARY.json"

    _write(report_md, build_report(payload))
    _write(article_md, _article_draft(payload))
    _write(open_items_md, _open_items())
    _write(acceptance_json, json.dumps(analyze_release(copied_json), ensure_ascii=False, indent=2) + "\n")
    figure_files = build_figures(copied_json, out_dir / "figures")

    files = [copied_json, report_md, article_md, open_items_md, acceptance_json, *figure_files]
    acceptance = json.loads(acceptance_json.read_text(encoding="utf-8"))
    manifest = {
        "tag": tag,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_SIMULATION_ONLY",
        "hardware_claim": False,
        "input_json": str(input_json),
        "reproduce_commands": [
            "python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 3 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json",
            "python tools/package_safe_neural_horizon_pwm_release.py --input-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json --out-dir paper/safe_neural_horizon_pwm_2026/20260522_host_release --tag 20260522_safe_neural_horizon_pwm_host_release",
        ],
        "files": [
            {
                "path": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "acceptance": {
            "report_written": report_md.exists(),
            "article_draft_written": article_md.exists(),
            "open_items_written": open_items_md.exists(),
            "host_release_ready": bool(acceptance.get("host_release_ready", False)),
            "hardware_ready": False,
        },
    }
    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    _write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Safe Neural Horizon PWM host-simulation release evidence.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="safe_neural_horizon_pwm_host_release")
    args = parser.parse_args()

    manifest = package_release(
        input_json=Path(args.input_json).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        tag=str(args.tag),
    )
    print(f"saved: {Path(args.out_dir).expanduser().resolve()}")
    print(f"files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
