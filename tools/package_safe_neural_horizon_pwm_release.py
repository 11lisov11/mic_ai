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
from tools.check_safe_neural_horizon_pwm_algorithm_identity import analyze_algorithm_identity
from tools.check_safe_neural_horizon_pwm_baselines import analyze_baselines
from tools.check_safe_neural_horizon_pwm_release import analyze_release
from tools.check_safe_neural_horizon_pwm_novelty import analyze_novelty
from tools.check_safe_neural_horizon_pwm_theory import analyze_theory


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _article_draft(
    payload: Dict[str, Any],
    trace_payload: Dict[str, Any] | None = None,
    twin_payload: Dict[str, Any] | None = None,
    mc500_payload: Dict[str, Any] | None = None,
    baseline_tuning_payload: Dict[str, Any] | None = None,
) -> str:
    scenarios = list(payload.get("scenarios", []))
    trace_ready = bool(trace_payload and trace_payload.get("trace_evidence_ready", False))
    twin_ready = bool(twin_payload and twin_payload.get("trained_domain_randomized_twin_ready", False))
    mc500_ready = bool(mc500_payload and int(mc500_payload.get("mc_trials", 0)) >= 500)
    baseline_tuning_ready = bool(
        baseline_tuning_payload and baseline_tuning_payload.get("baseline_tuning_ready", False)
    )
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
    lines.append("- Host-tested no-shoot-through and no-direct-HIGH-to-LOW timing-path invariants for vector transitions.")
    lines.append("- Horizon AI-PWM controller with neural cost shaping and event-triggered feedback policy.")
    lines.append("- Domain-randomized theta-conditioned twin identification evidence with multi-step rollout losses.")
    if baseline_tuning_ready:
        lines.append("- Bounded parameter-sweep tuning evidence for all named host comparison baselines.")
    lines.append("- Scenario matrix, ablation smoke, Pareto extraction, fault-injection summary, and host trace/FFT evidence package.")
    lines.append("- Machine-checkable release, algorithm-identity, novelty, and theory-completion audits.")
    lines.append("")
    lines.append("## Novelty Claim Scope")
    lines.append("")
    lines.append(
        "The host-level novelty claim is architectural, not a hardware or universal-superiority claim: SNH-PWM combines "
        "event-triggered twin feedback, neural cost shaping, finite-horizon inverter-vector search, and a protected "
        "AI-PWM Safety Gateway into one control law."
    )
    lines.append("")
    lines.append(
        "Compared with classical FOC-SVM, the controller does not synthesize continuous voltage references and then apply "
        "SVM; it searches legal inverter vectors directly under feedback/switching/risk costs. Compared with one-step "
        "FCS-MPC, it adds neural cost shaping, event-triggered feedback economy, and a mandatory gate-safety layer. "
        "Compared with the prior protected AI-PWM H1 model, it adds horizon search, twin uncertainty, and explicit "
        "feedback-usage optimization."
    )
    lines.append("")
    lines.append("The tracked release therefore supports only this claim: a distinct host-simulated control architecture exists and is machine-checked against the current host evidence.")
    lines.append("")
    lines.append(
        "The companion theory-completion audit separates host/software evidence from hardware evidence. "
        "When `publication_theory_complete = true`, it means the current host-theory evidence package passes "
        "the configured software gates; it still does not claim MCU, HIL, bench, or universal-superiority readiness."
    )
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
        no_deadtime = dict(dict(fault.get("cases", {})).get("no_deadtime_transition_emulation", {}))
        lines.append(f"- deadtime_transition_detector_triggered: `{bool(no_deadtime.get('blocked_by_gateway_deadtime_path', False))}`")
        lines.append("")
    lines.append("## Preliminary Findings")
    lines.append("")
    lines.append("- H2 is the safer current research candidate than the sparse H4 variant in the short host matrix.")
    lines.append("- Sparse H4 can reduce feedback and switching, but current stress and fallback events increase in several scenarios.")
    lines.append("- The FCS-MPC comparison is now a separate one-step current/torque/flux predictive baseline.")
    lines.append("- The prior protected AI-PWM H1, FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current, and sensorless/adaptive FOC comparisons are now separate host baselines.")
    lines.append("- The new FOC-SVM/FCS-MPC/DTC/DTC-SVM/deadbeat/sensorless baselines are competitive, so SNH-PWM cannot claim classical-control superiority yet.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Host simulation only.")
    if baseline_tuning_ready:
        lines.append("- FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC have bounded host tuning evidence, but are still not hardware or vendor-grade certified controllers.")
    else:
        lines.append("- FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC are host baselines, but not final tuned publication-grade.")
    if twin_ready:
        lines.append("- Domain-randomized theta-conditioned twin evidence exists, but it is host-only and assumes theta/passport or identification context.")
    else:
        lines.append("- No trained/identified domain-randomized neural twin yet.")
    if mc500_ready:
        lines.append("- Host MC=500 publication-scale smoke exists, but final MC must be repeated after strong-baseline tuning.")
    else:
        lines.append("- First MC=100 smoke exists, but no MC=500..1000 publication-scale run yet.")
    if trace_ready:
        lines.append("- Host trace package with time-series CSV plus FFT/THD-like torque-current evidence exists, but it is still simulation-only and not hardware THD.")
    else:
        lines.append("- No long-run trace package with FFT/THD torque-current evidence yet.")
    lines.append("- No fixed-point/WCET analysis.")
    lines.append("- No MCU, HIL, oscilloscope, inverter, or motor-bench validation.")
    lines.append("")
    lines.append("## Required Next Work")
    lines.append("")
    if baseline_tuning_ready:
        lines.append("- Expand the bounded tuning sweep into larger publication sweeps after any model/controller change.")
    else:
        lines.append("- Tune the FOC-SVM/FCS-MPC/DTC/DTC-SVM/deadbeat/sensorless baselines into strong publication baselines.")
    if mc500_ready:
        lines.append("- Re-run publication-scale MC after baseline replacement/tuning.")
    else:
        lines.append("- Run publication-scale MC after baseline replacement.")
    if trace_ready:
        lines.append("- Expand the host trace/FFT package after baseline tuning and validate it against HIL/bench traces.")
    else:
        lines.append("- Add publication-grade plots and FFT/THD metrics.")
    if twin_ready:
        lines.append("- Replace the theta-conditioned host twin with a production online identifier before MCU/HIL/bench claims.")
    else:
        lines.append("- Train or identify the neural twin with domain randomization and multi-step losses.")
    lines.append("- Port the safety gateway and timing checks to the target MCU/HIL path.")
    lines.append("- Validate gate timing and current trips on real hardware before any hardware-ready claim.")
    lines.append("")
    return "\n".join(lines)


def _open_items(
    trace_payload: Dict[str, Any] | None = None,
    twin_payload: Dict[str, Any] | None = None,
    mc500_payload: Dict[str, Any] | None = None,
    baseline_tuning_payload: Dict[str, Any] | None = None,
) -> str:
    trace_ready = bool(trace_payload and trace_payload.get("trace_evidence_ready", False))
    twin_ready = bool(twin_payload and twin_payload.get("trained_domain_randomized_twin_ready", False))
    mc500_ready = bool(mc500_payload and int(mc500_payload.get("mc_trials", 0)) >= 500)
    baseline_tuning_ready = bool(
        baseline_tuning_payload and baseline_tuning_payload.get("baseline_tuning_ready", False)
    )
    if trace_ready and baseline_tuning_ready:
        trace_item = "- Expand the host trace/FFT/THD-like package after future model/controller changes; current evidence is simulation-only and not hardware power-analyzer THD."
    elif trace_ready:
        trace_item = "- Expand the host trace/FFT/THD-like package after baseline tuning; current evidence is simulation-only and not hardware power-analyzer THD."
    else:
        trace_item = "- Add publication-grade long-run metrics: THD, FFT torque, switching loss, conduction loss, thermal imbalance, EMI/common-mode proxy."
    twin_item = (
        "- Replace the theta-conditioned host twin evidence with a production online parameter identifier before MCU/HIL/bench claims."
        if twin_ready
        else "- Train or identify the neural twin with domain randomization and multi-step losses."
    )
    if baseline_tuning_ready and mc500_ready:
        mc_item = "- Re-run/scale MC=500..1000 after any controller, model, or tuning-grid change; current MC500 is valid for this host release."
    elif mc500_ready:
        mc_item = "- Re-run MC=500..1000 after strong baselines are tuned; current MC500 is host evidence before final tuning."
    else:
        mc_item = "- Run MC=500..1000 after strong baselines are ready."
    if baseline_tuning_ready and mc500_ready:
        baseline_item = "- Expand bounded baseline tuning to wider MC/scenario sweeps before any journal superiority claim."
        complete_guard = "- Keep hardware readiness false: host theory completion does not replace MCU/HIL/bench validation."
    elif baseline_tuning_ready:
        baseline_item = "- Bounded baseline tuning evidence exists; MC=500..1000 remains open after any final controller/model edits."
        complete_guard = "- Keep `publication_theory_complete=false` until MC=500..1000 and plot gates are present."
    else:
        baseline_item = "- Tune the host key-level FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat, and sensorless/adaptive FOC baselines to publication-grade strength."
        complete_guard = (
            "- Keep `publication_theory_complete=false` until strong baselines are present; host trace/twin/MC500 evidence alone is not enough."
            if mc500_ready
            else "- Keep `publication_theory_complete=false` until strong baselines and MC=500..1000 are present; host trace/twin evidence alone is not enough."
        )
    return "\n".join(
        [
            "# Safe Neural Horizon PWM Open Items",
            "",
            baseline_item,
            trace_item,
            mc_item,
            twin_item,
            complete_guard,
            "- Add fixed-point or bounded floating-point MCU implementation plus WCET.",
            "- Add HIL, oscilloscope gate timing, current trip, watchdog, and bench validation.",
            "- Do not claim hardware-ready status until real MCU/HIL/bench evidence exists.",
            "",
        ]
    )


def _copy_trace_evidence(trace_dir: Path | None, out_dir: Path) -> tuple[list[Path], Dict[str, Any] | None]:
    if trace_dir is None:
        return [], None
    if not trace_dir.exists():
        raise FileNotFoundError(trace_dir)
    summary_json = trace_dir / "trace_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)
    target = out_dir / "trace_evidence"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(trace_dir, target)
    trace_payload = json.loads((target / "trace_summary.json").read_text(encoding="utf-8"))
    files = [path for path in target.rglob("*") if path.is_file()]
    return files, trace_payload


def _copy_twin_evidence(twin_dir: Path | None, out_dir: Path) -> tuple[list[Path], Dict[str, Any] | None]:
    if twin_dir is None:
        return [], None
    if not twin_dir.exists():
        raise FileNotFoundError(twin_dir)
    summary_json = twin_dir / "twin_training_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)
    target = out_dir / "twin_evidence"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(twin_dir, target)
    twin_payload = json.loads((target / "twin_training_summary.json").read_text(encoding="utf-8"))
    files = [path for path in target.rglob("*") if path.is_file()]
    return files, twin_payload


def package_release(
    input_json: Path,
    out_dir: Path,
    tag: str,
    mc100_json: Path | None = None,
    mc500_json: Path | None = None,
    baseline_stress_json: Path | None = None,
    baseline_tuning_json: Path | None = None,
    trace_dir: Path | None = None,
    twin_dir: Path | None = None,
) -> Dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_json = out_dir / "safe_neural_horizon_pwm_results.json"
    shutil.copyfile(input_json, copied_json)
    mc100_source = mc100_json if mc100_json is not None else ROOT / ".tmp_pytest" / "safe_neural_horizon_pwm_study_mc100.json"
    mc100_json = out_dir / "safe_neural_horizon_pwm_mc100_smoke.json"
    if not mc100_source.exists():
        raise FileNotFoundError(
            f"tracked release requires MC100 smoke evidence; run "
            f"`python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 100 --steps 120 "
            f"--out-json {mc100_source}` or pass --mc100-json"
        )
    shutil.copyfile(mc100_source, mc100_json)
    mc500_source = mc500_json if mc500_json is not None else ROOT / ".tmp_pytest" / "safe_neural_horizon_pwm_study_mc500.json"
    mc500_json = out_dir / "safe_neural_horizon_pwm_mc500_publication_smoke.json"
    if not mc500_source.exists():
        raise FileNotFoundError(
            f"tracked release requires MC500 host evidence; run "
            f"`python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 500 --steps 120 "
            f"--out-json {mc500_source}` or pass --mc500-json"
        )
    shutil.copyfile(mc500_source, mc500_json)
    mc500_payload = json.loads(mc500_json.read_text(encoding="utf-8"))
    stress_source = (
        baseline_stress_json
        if baseline_stress_json is not None
        else ROOT / ".tmp_pytest" / "safe_neural_horizon_pwm_baseline_stress.json"
    )
    baseline_stress_json = out_dir / "safe_neural_horizon_pwm_baseline_stress_evidence.json"
    if not stress_source.exists():
        raise FileNotFoundError(
            f"tracked release requires baseline stress evidence; run "
            f"`python tools/build_safe_neural_horizon_pwm_baseline_stress.py "
            f"--out-json {stress_source}` or pass --baseline-stress-json"
        )
    shutil.copyfile(stress_source, baseline_stress_json)
    tuning_source = (
        baseline_tuning_json
        if baseline_tuning_json is not None
        else ROOT / ".tmp_pytest" / "safe_neural_horizon_pwm_baseline_tuning.json"
    )
    baseline_tuning_json = out_dir / "safe_neural_horizon_pwm_baseline_tuning_evidence.json"
    if not tuning_source.exists():
        raise FileNotFoundError(
            f"tracked release requires baseline tuning evidence; run "
            f"`python tools/build_safe_neural_horizon_pwm_baseline_tuning.py "
            f"--out-json {tuning_source}` or pass --baseline-tuning-json"
        )
    shutil.copyfile(tuning_source, baseline_tuning_json)
    baseline_tuning_payload = json.loads(baseline_tuning_json.read_text(encoding="utf-8"))
    report_md = out_dir / "safe_neural_horizon_pwm_report.md"
    article_md = out_dir / "safe_neural_horizon_pwm_article_draft.md"
    baseline_json = out_dir / "safe_neural_horizon_pwm_baseline_strength_audit.json"
    identity_json = out_dir / "safe_neural_horizon_pwm_algorithm_identity_audit.json"
    novelty_json = out_dir / "safe_neural_horizon_pwm_novelty_audit.json"
    theory_json = out_dir / "safe_neural_horizon_pwm_theory_completion_audit.json"
    open_items_md = out_dir / "WHAT_IS_NOT_DONE.md"
    acceptance_json = out_dir / "HOST_ACCEPTANCE_SUMMARY.json"

    trace_files, trace_payload = _copy_trace_evidence(trace_dir, out_dir)
    twin_files, twin_payload = _copy_twin_evidence(twin_dir, out_dir)
    _write(report_md, build_report(payload))
    _write(article_md, _article_draft(payload, trace_payload, twin_payload, mc500_payload, baseline_tuning_payload))
    _write(baseline_json, json.dumps(analyze_baselines(out_dir), ensure_ascii=False, indent=2) + "\n")
    _write(identity_json, json.dumps(analyze_algorithm_identity(out_dir), ensure_ascii=False, indent=2) + "\n")
    _write(novelty_json, json.dumps(analyze_novelty(out_dir), ensure_ascii=False, indent=2) + "\n")
    _write(open_items_md, _open_items(trace_payload, twin_payload, mc500_payload, baseline_tuning_payload))
    figure_files = build_figures(copied_json, out_dir / "figures")
    _write(theory_json, json.dumps(analyze_theory(out_dir), ensure_ascii=False, indent=2) + "\n")

    # Do not include HOST_ACCEPTANCE_SUMMARY.json in the manifest hash list: it is
    # generated after the manifest so it can validate the manifest itself.
    files = [
        copied_json,
        mc100_json,
        mc500_json,
        baseline_stress_json,
        baseline_tuning_json,
        report_md,
        article_md,
        baseline_json,
        identity_json,
        novelty_json,
        theory_json,
        open_items_md,
        *figure_files,
        *trace_files,
        *twin_files,
    ]
    manifest = {
        "tag": tag,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_SIMULATION_ONLY",
        "hardware_claim": False,
        "input_json": str(input_json),
        "reproduce_commands": [
            "python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 3 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json",
            "python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 100 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc100.json",
            "python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 500 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc500.json",
            "python tools/build_safe_neural_horizon_pwm_baseline_stress.py --mc 3 --steps 80 --out-json .tmp_pytest/safe_neural_horizon_pwm_baseline_stress.json",
            "python tools/build_safe_neural_horizon_pwm_baseline_tuning.py --mc 2 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_baseline_tuning.json",
            "python tools/build_safe_neural_horizon_pwm_trace_evidence.py --steps 512 --out-dir .tmp_pytest/safe_neural_horizon_pwm_trace_evidence",
            "python tools/build_safe_neural_horizon_pwm_twin_evidence.py --out-dir .tmp_pytest/safe_neural_horizon_pwm_twin_evidence",
            "python tools/package_safe_neural_horizon_pwm_release.py --input-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json --out-dir paper/safe_neural_horizon_pwm_2026/20260522_host_release --tag 20260522_safe_neural_horizon_pwm_host_release --trace-dir .tmp_pytest/safe_neural_horizon_pwm_trace_evidence --twin-dir .tmp_pytest/safe_neural_horizon_pwm_twin_evidence --mc500-json .tmp_pytest/safe_neural_horizon_pwm_study_mc500.json --baseline-stress-json .tmp_pytest/safe_neural_horizon_pwm_baseline_stress.json --baseline-tuning-json .tmp_pytest/safe_neural_horizon_pwm_baseline_tuning.json",
        ],
        "files": [
            {
                "path": path.relative_to(out_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "acceptance": {
            "report_written": report_md.exists(),
            "article_draft_written": article_md.exists(),
            "baseline_strength_audit_written": baseline_json.exists(),
            "algorithm_identity_audit_written": identity_json.exists(),
            "novelty_audit_written": novelty_json.exists(),
            "theory_completion_audit_written": theory_json.exists(),
            "mc100_smoke_written": mc100_json.exists(),
            "mc500_publication_smoke_written": mc500_json.exists(),
            "baseline_stress_evidence_written": baseline_stress_json.exists(),
            "baseline_tuning_evidence_written": baseline_tuning_json.exists(),
            "open_items_written": open_items_md.exists(),
            "trace_evidence_written": bool(trace_files),
            "twin_evidence_written": bool(twin_files),
            "acceptance_summary_written": True,
            "host_release_ready": False,
            "hardware_ready": False,
        },
    }
    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    _write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write(acceptance_json, json.dumps({"status": "pending_acceptance_summary"}, ensure_ascii=False, indent=2) + "\n")
    acceptance = analyze_release(out_dir)
    _write(acceptance_json, json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n")
    manifest["acceptance"]["host_release_ready"] = bool(acceptance.get("host_release_ready", False))
    _write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Safe Neural Horizon PWM host-simulation release evidence.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="safe_neural_horizon_pwm_host_release")
    parser.add_argument("--mc100-json", default="")
    parser.add_argument("--mc500-json", default="")
    parser.add_argument("--baseline-stress-json", default="")
    parser.add_argument("--baseline-tuning-json", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--twin-dir", default="")
    args = parser.parse_args()

    manifest = package_release(
        input_json=Path(args.input_json).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        tag=str(args.tag),
        mc100_json=Path(args.mc100_json).expanduser().resolve() if str(args.mc100_json).strip() else None,
        mc500_json=Path(args.mc500_json).expanduser().resolve() if str(args.mc500_json).strip() else None,
        baseline_stress_json=Path(args.baseline_stress_json).expanduser().resolve()
        if str(args.baseline_stress_json).strip()
        else None,
        baseline_tuning_json=Path(args.baseline_tuning_json).expanduser().resolve()
        if str(args.baseline_tuning_json).strip()
        else None,
        trace_dir=Path(args.trace_dir).expanduser().resolve() if str(args.trace_dir).strip() else None,
        twin_dir=Path(args.twin_dir).expanduser().resolve() if str(args.twin_dir).strip() else None,
    )
    print(f"saved: {Path(args.out_dir).expanduser().resolve()}")
    print(f"files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
