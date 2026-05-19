from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "train3_fullprog_20260519" / "final_selected_strict_recheck_20260519" / "selected_recheck_summary.json"
DEFAULT_JOINT_MANIFEST = ROOT / "outputs" / "train3_fullprog_20260519" / "20260519_205216_joint_domain_randomized" / "training_manifest_3motors.json"
DEFAULT_FINETUNE_MANIFEST = ROOT / "outputs" / "train3_fullprog_20260519" / "20260519_205338_fine_tune_per_motor" / "training_manifest_3motors.json"
DEFAULT_OUT_DIR = ROOT / "paper" / "ieee_2026" / "data" / "release" / "20260519_train3_refresh"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _runs_by_motor(finetune_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = finetune_manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("fine-tune manifest 'runs' must be a list")
    out: dict[str, dict[str, Any]] = {}
    for row in runs:
        if not isinstance(row, dict):
            continue
        motor = str(row.get("motor", "")).strip().lower()
        if motor:
            out[motor] = dict(row)
    return out


def build_refresh_manifest(
    *,
    summary_path: Path,
    joint_manifest_path: Path,
    finetune_manifest_path: Path,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    joint_manifest = _load_json(joint_manifest_path)
    finetune_manifest = _load_json(finetune_manifest_path)
    if not isinstance(summary, list):
        raise ValueError("selected recheck summary must be a list")
    if not isinstance(joint_manifest, dict) or not isinstance(finetune_manifest, dict):
        raise ValueError("training manifests must be JSON objects")

    run_index = _runs_by_motor(finetune_manifest)
    motors: list[dict[str, Any]] = []
    for row in summary:
        if not isinstance(row, dict):
            continue
        motor = str(row.get("motor", "")).strip().lower()
        if not motor:
            continue
        run = run_index.get(motor, {})
        selected_path = _resolve(str(row.get("step27_selected_checkpoint", "")))
        promoted_path = _resolve(str(run.get("step27_promoted_checkpoint", run.get("best_checkpoint", ""))))
        canonical_source = str(run.get("step27_included_canonical_checkpoint", "")).strip()
        selected_is_canonical = bool(row.get("selected_is_canonical_baseline", False))
        decision = "keep_canonical_baseline" if selected_is_canonical else "promote_training_checkpoint"
        release_checkpoint = _resolve(canonical_source) if selected_is_canonical and canonical_source else promoted_path
        motors.append(
            {
                "motor": motor,
                "decision": decision,
                "acceptance_pass": bool(row.get("acceptance_pass", False)),
                "envelope_fail_count": int(row.get("envelope_fail_count", 0)),
                "err_failures": float(row.get("err_failures", 0.0)),
                "avg_power_saving_pct": float(row.get("avg_power_saving_pct", 0.0)),
                "avg_eta_gain_pct": float(row.get("avg_eta_gain_pct", 0.0)),
                "start_stop_power_saving_pct": float(row.get("start_stop_power_saving_pct", 0.0)),
                "candidate_tag": str(row.get("candidate_tag", "")),
                "selected_is_canonical_baseline": selected_is_canonical,
                "selected_checkpoint": _rel(selected_path),
                "selected_checkpoint_exists": selected_path.is_file(),
                "selected_checkpoint_sha256": _sha256(selected_path),
                "release_checkpoint": _rel(release_checkpoint),
                "release_checkpoint_exists": release_checkpoint.is_file(),
                "release_checkpoint_sha256": _sha256(release_checkpoint),
                "canonical_source_checkpoint": "" if not canonical_source else _rel(_resolve(canonical_source)),
            }
        )

    all_green = bool(motors) and all(
        bool(row["acceptance_pass"])
        and int(row["envelope_fail_count"]) == 0
        and float(row["err_failures"]) == 0.0
        and bool(row["release_checkpoint_exists"])
        for row in motors
    )
    return {
        "schema": "mic_theory.train3_refresh_release.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "refresh_tag": "20260519_train3_refresh",
        "research_refresh_complete": all_green,
        "hardware_deploy_complete": False,
        "summary_source": _rel(summary_path),
        "joint_manifest": _rel(joint_manifest_path),
        "fine_tune_manifest": _rel(finetune_manifest_path),
        "joint_protocol_hash": str(joint_manifest.get("protocol_hash", "")),
        "fine_tune_protocol_hash": str(finetune_manifest.get("protocol_hash", "")),
        "motors": motors,
        "promotion_policy": {
            "air56": "keep previous canonical baseline",
            "al31": "promote 2026-05-19 fine-tuned checkpoint because strict recheck improved AL31",
            "ao2": "keep previous canonical nameplate-first baseline",
        },
        "reproduce_commands": {
            "joint_domain_randomized": (
                "python tools/train_3motors_pipeline.py --mode joint-domain-randomized "
                "--motors air56,al31,ao2 --seeds 101 --joint-cycles 1 "
                "--joint-cycle-episodes 24 --episode-steps 2400 "
                "--scenarios speed_step,ramp,load_step,start_stop --scenario-sample cycle "
                "--relative --out-dir outputs/train3_fullprog_20260519 "
                "--ai-output-dir outputs/train3_fullprog_20260519/ai_outputs "
                "--results-root outputs/train3_fullprog_20260519/results_run"
            ),
            "fine_tune_per_motor": (
                "python tools/train_3motors_pipeline.py --mode fine_tune_per_motor "
                "--motors air56,al31,ao2 --seeds 101 --episodes 40 --episode-steps 2400 "
                "--scenarios speed_step,ramp,load_step,start_stop --scenario-sample cycle --relative "
                "--base-manifest outputs/train3_fullprog_20260519/20260519_205216_joint_domain_randomized/training_manifest_3motors.json "
                "--step27-select --step27-seeds 101 --step27-scenarios speed_step,ramp,load_step,start_stop "
                "--out-dir outputs/train3_fullprog_20260519 "
                "--ai-output-dir outputs/train3_fullprog_20260519/ai_outputs "
                "--results-root outputs/train3_fullprog_20260519/results_run"
            ),
        },
        "artifact_storage_note": (
            "Large checkpoints stay under ignored outputs/. This tracked manifest records paths and hashes; "
            "rerun the reproduce commands if a clean clone needs to regenerate them."
        ),
    }


def _markdown_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# 20260519 Train3 Refresh Release",
        "",
        f"- research_refresh_complete: `{str(manifest['research_refresh_complete']).lower()}`",
        f"- hardware_deploy_complete: `{str(manifest['hardware_deploy_complete']).lower()}`",
        f"- summary_source: `{manifest['summary_source']}`",
        "",
        "| Motor | Decision | Power saving | Eta gain | Start-stop | Err | Envelope fails |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["motors"]:
        lines.append(
            "| {motor} | {decision} | {power:.3f}% | {eta:.3f}% | {start:.3f}% | {err:.0f} | {env} |".format(
                motor=row["motor"],
                decision=row["decision"],
                power=float(row["avg_power_saving_pct"]),
                eta=float(row["avg_eta_gain_pct"]),
                start=float(row["start_stop_power_saving_pct"]),
                err=float(row["err_failures"]),
                env=int(row["envelope_fail_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Checkpoint Policy",
            "",
            "AIR56 and AO2 remain on the accepted canonical baselines. AL31 is promoted from the 2026-05-19 fine-tune run.",
            "",
            "Large checkpoint binaries are intentionally not committed; use the manifest hashes plus reproduce commands to regenerate them.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tracked manifest for the 2026-05-19 3-motor training refresh.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--joint-manifest", default=str(DEFAULT_JOINT_MANIFEST))
    parser.add_argument("--fine-tune-manifest", default=str(DEFAULT_FINETUNE_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    manifest = build_refresh_manifest(
        summary_path=_resolve(str(args.summary)),
        joint_manifest_path=_resolve(str(args.joint_manifest)),
        finetune_manifest_path=_resolve(str(args.fine_tune_manifest)),
    )
    out_dir = _resolve(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "research_refresh_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_markdown_report(manifest), encoding="utf-8")
    print(json.dumps({"out_dir": _rel(out_dir), "research_refresh_complete": manifest["research_refresh_complete"]}, indent=2))
    return 0 if bool(manifest["research_refresh_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
