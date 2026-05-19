from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.train_ai_id_ref import train as train_id_ref
from tools.common_utils import json_load as _json_load_shared
from tools.common_utils import parse_csv_list as _parse_csv_list_shared
from tools.common_utils import parse_int_list as _parse_int_list_shared
from tools.common_utils import write_csv as _write_csv_shared


@dataclass(frozen=True)
class MotorSpec:
    key: str
    config_path: str


MOTOR_REGISTRY: Dict[str, MotorSpec] = {
    "air56": MotorSpec("air56", "config/env_research_air56_025kw.py"),
    "al31": MotorSpec("al31", "config/env_research_al31_4_06kw.py"),
    "ao2": MotorSpec("ao2", "config/env_research_ao2_32_4_3kw.py"),
}

CANONICAL_STEP27_SELECTION: Dict[str, Dict[str, str]] = {
    "air56": {
        "ai_control_mode": "ai_id_ref_hybrid",
        "candidate_json": "outputs/tmp_air56_rand007_soft_track_single_20260326.json",
        "candidate_tag": "rand007_soft_track",
        "checkpoint_path": (
            "outputs/air56_ep002_loadheavy_wspeed2_20260408h/results_run/"
            "20260408_203735_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep001.pth"
        ),
    },
    "al31": {
        "ai_control_mode": "ai_id_ref",
        "candidate_json": "outputs/tmp_al31_mid04_ultrafine2_20260328.json",
        "candidate_tag": "mid04_speed_dn_04",
        "checkpoint_path": (
            "outputs/train3_fullprog_20260519/results_run/fine_tune_seed101_al31/"
            "20260519_205752_env_research_al31_4_06kw_ai_id_ref/best_actor_step27_train3.pth"
        ),
    },
    "ao2": {
        "ai_control_mode": "ai_id_ref",
        "candidate_json": "config/step27_ao2_current_repro_candidate_20260519.json",
        "candidate_tag": "ao2_current_repro_rand017",
        "checkpoint_path": "outputs/ao2_tuned_rampfocus_pilot_20260412m/shared/checkpoints/env_backlog_ao2_nameplate_foc_tuned/best_actor.pth",
    },
}


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    value = value.strip("_")
    return value or "run"


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_csv(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _resolve_motors(text: str) -> List[MotorSpec]:
    keys = _parse_csv_list(text)
    out: List[MotorSpec] = []
    for key in keys:
        k = str(key).lower()
        if k not in MOTOR_REGISTRY:
            raise KeyError(f"Unknown motor '{key}'. Known: {','.join(sorted(MOTOR_REGISTRY.keys()))}")
        out.append(MOTOR_REGISTRY[k])
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _load_json(path: Path) -> Dict[str, object]:
    return _json_load_shared(path)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_episode_entries(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        out: List[Dict[str, object]] = []
        for row in data:
            if isinstance(row, dict):
                out.append(dict(row))
        return out
    return []


def _run_key(*, motor: str, seed: int, stage: str) -> str:
    return f"{str(motor).lower()}::{int(seed)}::{str(stage)}"


def _load_resume_index(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    rows = manifest.get("runs", [])
    out: Dict[str, Dict[str, object]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        motor = str(row.get("motor", "")).strip().lower()
        stage = str(row.get("stage", "")).strip()
        try:
            seed = int(row.get("seed", 0))
        except Exception:
            continue
        if not motor or not stage:
            continue
        out[_run_key(motor=motor, seed=seed, stage=stage)] = dict(row)
    return out


def _load_acceptance_index(manifest: Dict[str, object]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return out
    csv_path_text = str(artifacts.get("training_acceptance_csv", "")).strip()
    if not csv_path_text:
        return out
    csv_path = Path(csv_path_text).resolve()
    if not csv_path.exists():
        return out
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                motor = str(row.get("motor", "")).strip().lower()
                stage = str(row.get("stage", "")).strip()
                seed_text = str(row.get("seed", "")).strip()
                passed_text = str(row.get("acceptance_pass", "")).strip().lower()
                if not motor or not stage:
                    continue
                try:
                    seed = int(seed_text)
                except Exception:
                    continue
                passed = passed_text in {"1", "true", "yes", "y"}
                out[_run_key(motor=motor, seed=seed, stage=stage)] = passed
    except Exception:
        return {}
    return out


def _checkpoint_paths_exist(row: Dict[str, object]) -> bool:
    best = _resolve_path(str(row.get("best_checkpoint", "")))
    episodes = _resolve_path(str(row.get("episodes_log", "")))
    if not best.exists():
        return False
    if not episodes.exists():
        return False
    return True


def _resolve_step27_selection_spec(motor_key: str, args: argparse.Namespace) -> Dict[str, str]:
    profile = str(getattr(args, "step27_profile", "canonical")).strip().lower()
    if profile != "canonical":
        raise ValueError(f"Unsupported Step27 selection profile: {profile}")
    key = str(motor_key).strip().lower()
    if key not in CANONICAL_STEP27_SELECTION:
        raise KeyError(f"No canonical Step27 selection spec for motor={motor_key}")
    spec = dict(CANONICAL_STEP27_SELECTION[key])
    candidate_path = _resolve_path(spec["candidate_json"])
    if not candidate_path.exists():
        raise FileNotFoundError(f"Step27 candidate json for {key} does not exist: {candidate_path}")
    spec["candidate_json"] = str(candidate_path)
    checkpoint_text = str(spec.get("checkpoint_path", "")).strip()
    if checkpoint_text:
        checkpoint_path = _resolve_path(checkpoint_text)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Step27 canonical checkpoint for {key} does not exist: {checkpoint_path}")
        spec["checkpoint_path"] = str(checkpoint_path)
    return spec


def _run_step27_selection(
    *,
    row: Dict[str, object],
    motor: MotorSpec,
    args: argparse.Namespace,
    run_root: Path,
) -> Dict[str, object]:
    if not bool(getattr(args, "step27_select", False)):
        return row

    from tools.scan_step27_checkpoints import scan_checkpoints

    run_dir = _resolve_path(str(row.get("run_dir", "")))
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Step27 selection eval dir does not exist: {eval_dir}")

    spec = _resolve_step27_selection_spec(motor.key, args)
    included_canonical_checkpoint = ""
    canonical_eval_checkpoint = eval_dir / "actor_ep_canonical_baseline.pth"
    if bool(getattr(args, "step27_include_canonical_baseline", True)):
        checkpoint_text = str(spec.get("checkpoint_path", "")).strip()
        if not checkpoint_text:
            raise ValueError(f"Step27 canonical baseline is enabled, but no checkpoint_path is configured for {motor.key}")
        source_checkpoint = _resolve_path(checkpoint_text)
        included_canonical_checkpoint = str(source_checkpoint)
        if source_checkpoint.resolve() != canonical_eval_checkpoint.resolve():
            shutil.copyfile(source_checkpoint, canonical_eval_checkpoint)
    elif canonical_eval_checkpoint.exists():
        canonical_eval_checkpoint.unlink()

    seed_text = str(getattr(args, "step27_seeds", "") or "").strip()
    selection_seeds = _parse_int_csv(seed_text) if seed_text else _parse_int_csv(str(args.seeds))
    selection_scenarios = _parse_csv_list(str(getattr(args, "step27_scenarios", "") or args.eval_scenarios))
    out_dir = (
        run_root
        / "step27_selection"
        / _slug(f"{row.get('stage', '')}_seed{int(row.get('seed', 0))}_{motor.key}")
    )
    summary = scan_checkpoints(
        motor=motor.key,
        config_path=motor.config_path,
        checkpoint_glob=str(eval_dir),
        ai_control_mode=str(spec["ai_control_mode"]),
        candidate_json=str(spec["candidate_json"]),
        candidate_tag=str(spec["candidate_tag"]),
        seeds=selection_seeds,
        scenarios=selection_scenarios,
        out_dir=out_dir,
        use_envelope_acceptance=bool(getattr(args, "step27_use_envelope_acceptance", True)),
        acceptance_envelopes=Path(str(getattr(args, "step27_acceptance_envelopes", "config/acceptance_envelopes_3motors.json"))),
        min_avg_power_saving_pct=float(getattr(args, "step27_min_avg_power_saving_pct", 0.0)),
        min_avg_eta_gain_pct=float(getattr(args, "step27_min_avg_eta_gain_pct", 0.0)),
        max_avg_eta_gain_pct=float(getattr(args, "step27_max_avg_eta_gain_pct", 25.0)),
        max_err_failures=float(getattr(args, "step27_max_err_failures", 2.0)),
        min_start_stop_saving_pct=float(getattr(args, "step27_min_start_stop_saving_pct", -0.5)),
        max_start_stop_saving_pct=float(getattr(args, "step27_max_start_stop_saving_pct", 20.0)),
        max_worst_current_peak_ratio=float(getattr(args, "step27_max_worst_current_peak_ratio", 1.30)),
        max_worst_current_mean_ratio=float(getattr(args, "step27_max_worst_current_mean_ratio", 1.20)),
        top_k=int(getattr(args, "step27_top_k", 10)),
        resume=bool(getattr(args, "step27_resume", False)),
    )
    best = dict(summary.get("best") or {})
    selected_checkpoint = _resolve_path(str(best.get("checkpoint", "")))
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Step27 selected checkpoint does not exist: {selected_checkpoint}")

    promoted = (run_dir / "best_actor_step27_train3.pth").resolve()
    shutil.copyfile(selected_checkpoint, promoted)

    selected = dict(row)
    selected["best_checkpoint_train_internal"] = str(row.get("best_checkpoint", ""))
    selected["best_checkpoint"] = str(promoted)
    selected["step27_select_enabled"] = True
    selected["step27_profile"] = str(getattr(args, "step27_profile", "canonical"))
    selected["step27_ai_control_mode"] = str(spec["ai_control_mode"])
    selected["step27_candidate_json"] = str(spec["candidate_json"])
    selected["step27_candidate_tag"] = str(spec["candidate_tag"])
    selected["step27_scan_summary_json"] = str((out_dir / f"{motor.key}_checkpoint_scan_summary.json").resolve())
    selected["step27_scan_rows_json"] = str((out_dir / f"{motor.key}_checkpoint_scan.json").resolve())
    selected["step27_selected_checkpoint"] = str(selected_checkpoint)
    selected["step27_promoted_checkpoint"] = str(promoted)
    selected["step27_included_canonical_checkpoint"] = str(included_canonical_checkpoint)
    selected["step27_selected_is_canonical_baseline"] = (
        bool(included_canonical_checkpoint) and selected_checkpoint.resolve() == canonical_eval_checkpoint.resolve()
    )
    selected["step27_acceptance_pass"] = bool(best.get("acceptance_pass", False))
    selected["step27_envelope_all_rows_pass"] = bool(best.get("envelope_all_rows_pass", False))
    selected["step27_avg_power_saving_pct"] = _safe_float(best.get("avg_power_saving_pct"))
    selected["step27_avg_eta_gain_pct"] = _safe_float(best.get("avg_eta_gain_pct"))
    selected["step27_err_failures"] = _safe_float(best.get("err_failures"))
    selected["step27_worst_current_peak_ratio"] = _safe_float(best.get("worst_current_peak_ratio"))
    selected["step27_worst_current_mean_ratio"] = _safe_float(best.get("worst_current_mean_ratio"))
    return selected


def _build_protocol_snapshot(
    *,
    args: argparse.Namespace,
    motors: List[MotorSpec],
    seeds: List[int],
    run_root: Path,
    resume_manifest: str | None,
) -> Dict[str, object]:
    motor_rows: List[Dict[str, object]] = []
    for m in motors:
        cfg = _resolve_path(m.config_path)
        motor_rows.append(
            {
                "motor": m.key,
                "config_path": str(cfg),
                "config_sha256": _file_sha256(cfg),
            }
        )
    protocol = {
        "mode": str(args.mode),
        "motors": [m.key for m in motors],
        "seeds": [int(s) for s in seeds],
        "scenarios": _parse_csv_list(args.scenarios),
        "eval_scenarios": _parse_csv_list(args.eval_scenarios),
        "joint_cycles": int(args.joint_cycles),
        "joint_cycle_episodes": int(args.joint_cycle_episodes),
        "episodes": int(args.episodes),
        "episode_steps": int(args.episode_steps),
        "control_mode": str(args.control_mode),
        "resume_manifest": "" if resume_manifest is None else str(Path(resume_manifest).resolve()),
        "eval_first": bool(args.eval_first),
        "step27_select": bool(getattr(args, "step27_select", False)),
        "step27_profile": str(getattr(args, "step27_profile", "canonical")),
        "step27_include_canonical_baseline": bool(getattr(args, "step27_include_canonical_baseline", True)),
        "step27_seeds": str(getattr(args, "step27_seeds", "")),
        "step27_scenarios": str(getattr(args, "step27_scenarios", "")),
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "motor_configs": motor_rows,
        "cli_args": {k: v for k, v in vars(args).items()},
    }
    return payload


def _build_repro_package_manifest(
    *,
    run_root: Path,
    run_rows: List[Dict[str, object]],
    protocol_hash: str,
) -> Dict[str, object]:
    checkpoints: List[Dict[str, object]] = []
    seen: set[str] = set()
    for row in run_rows:
        for field in ("best_checkpoint", "last_checkpoint"):
            p = _resolve_path(str(row.get(field, "")))
            key = str(p)
            if not key or key in seen:
                continue
            seen.add(key)
            checkpoints.append(
                {
                    "path": key,
                    "sha256": _file_sha256(p),
                    "exists": bool(p.exists()),
                }
            )
    artifacts = {
        "training_runs_csv": str((run_root / "training_runs_3motors.csv").resolve()),
        "training_summaries_csv": str((run_root / "training_run_summaries_3motors.csv").resolve()),
        "training_acceptance_csv": str((run_root / "training_acceptance_matrix_3motors.csv").resolve()),
        "training_eval_snapshots_csv": str((run_root / "training_eval_snapshots_3motors.csv").resolve()),
        "checkpoint_registry_json": str((run_root / "checkpoints_registry_3motors.json").resolve()),
    }
    artifact_hashes = {
        name: _file_sha256(Path(path))
        for name, path in artifacts.items()
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "protocol_hash": str(protocol_hash),
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "checkpoints": checkpoints,
    }


def _summarize_run(
    row: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    episodes_log_path = _resolve_path(str(row.get("episodes_log", "")))
    run_dir_path = _resolve_path(str(row.get("run_dir", "")))
    best_ckpt_path = _resolve_path(str(row.get("best_checkpoint", "")))
    last_ckpt_path = _resolve_path(str(row.get("last_checkpoint", "")))

    entries = _load_episode_entries(episodes_log_path)
    n_ep = int(len(entries))
    final = entries[-1] if entries else {}

    speed_vals = [_safe_float(r.get("mean_speed_error")) for r in entries]
    eta_vals = [_safe_float(r.get("eta_energy")) for r in entries]
    p_in_vals = [_safe_float(r.get("mean_p_in_pos")) for r in entries]
    current_vals = [_safe_float(r.get("mean_current_rms")) for r in entries]

    final_speed = _safe_float(final.get("mean_speed_error"), default=float("inf"))
    final_eta = _safe_float(final.get("eta_energy"), default=float("-inf"))
    final_p_in = _safe_float(final.get("mean_p_in_pos"), default=float("inf"))
    final_current = _safe_float(final.get("mean_current_rms"), default=float("inf"))

    min_speed = min(speed_vals) if speed_vals else float("nan")
    max_eta = max(eta_vals) if eta_vals else float("nan")
    min_p_in = min(p_in_vals) if p_in_vals else float("nan")

    w = max(1, int(args.degradation_window))
    speed_degradation = False
    eta_degradation = False
    speed_delta = 0.0
    eta_delta = 0.0
    if len(speed_vals) >= 2 * w:
        prev_speed = float(sum(speed_vals[-2 * w : -w]) / w)
        last_speed = float(sum(speed_vals[-w:]) / w)
        speed_delta = last_speed - prev_speed
        speed_degradation = bool(speed_delta > float(args.degradation_speed_delta))
    if len(eta_vals) >= 2 * w:
        prev_eta = float(sum(eta_vals[-2 * w : -w]) / w)
        last_eta = float(sum(eta_vals[-w:]) / w)
        eta_delta = prev_eta - last_eta
        eta_degradation = bool(eta_delta > float(args.degradation_eta_delta))

    speed_ok = bool(final_speed <= float(args.accept_max_speed_error))
    eta_min_ok = bool(math.isfinite(final_eta) and final_eta >= float(args.accept_min_eta_energy))
    eta_max_ok = bool(math.isfinite(final_eta) and final_eta <= float(args.accept_max_eta_energy))
    eta_ok = bool(eta_min_ok and eta_max_ok)
    current_ok = bool(final_current <= float(args.accept_max_current_rms))
    p_in_ok = True if args.accept_max_p_in_pos is None else bool(final_p_in <= float(args.accept_max_p_in_pos))
    degradation_ok = bool(not speed_degradation and not eta_degradation)
    training_episode_acceptance_pass = bool(speed_ok and eta_ok and current_ok and p_in_ok and degradation_ok)
    step27_enabled = bool(getattr(args, "step27_select", False))
    step27_acceptance_pass = bool(row.get("step27_acceptance_pass", False))
    acceptance_pass = bool(step27_acceptance_pass) if step27_enabled else training_episode_acceptance_pass

    actor_snapshots: List[Path] = []
    eval_dir = run_dir_path / "eval"
    if eval_dir.exists():
        actor_snapshots = sorted(eval_dir.glob("actor_ep*.pth"))
    snapshot_rows: List[Dict[str, object]] = [
        {
            "motor": row.get("motor", ""),
            "seed": int(row.get("seed", 0)),
            "stage": row.get("stage", ""),
            "snapshot_file": str(p.resolve()),
        }
        for p in actor_snapshots
    ]

    summary_row: Dict[str, object] = {
        "motor": row.get("motor", ""),
        "seed": int(row.get("seed", 0)),
        "stage": row.get("stage", ""),
        "episodes_total": n_ep,
        "final_mean_speed_error": final_speed,
        "final_eta_energy": final_eta,
        "final_mean_p_in_pos": final_p_in,
        "final_mean_current_rms": final_current,
        "min_mean_speed_error": min_speed,
        "max_eta_energy": max_eta,
        "min_mean_p_in_pos": min_p_in,
        "speed_degradation": speed_degradation,
        "eta_degradation": eta_degradation,
        "speed_degradation_delta": speed_delta,
        "eta_degradation_delta": eta_delta,
        "eval_actor_snapshots": int(len(actor_snapshots)),
        "episodes_log": str(episodes_log_path),
        "run_dir": str(run_dir_path),
        "step27_select_enabled": bool(row.get("step27_select_enabled", False)),
        "step27_acceptance_pass": bool(row.get("step27_acceptance_pass", False)),
        "step27_avg_power_saving_pct": row.get("step27_avg_power_saving_pct", ""),
        "step27_avg_eta_gain_pct": row.get("step27_avg_eta_gain_pct", ""),
        "step27_err_failures": row.get("step27_err_failures", ""),
    }
    acceptance_row: Dict[str, object] = {
        "motor": row.get("motor", ""),
        "seed": int(row.get("seed", 0)),
        "stage": row.get("stage", ""),
        "acceptance_source": "step27" if step27_enabled else "training_episode",
        "speed_ok": speed_ok,
        "eta_ok": eta_ok,
        "eta_min_ok": eta_min_ok,
        "eta_max_ok": eta_max_ok,
        "current_ok": current_ok,
        "p_in_ok": p_in_ok,
        "degradation_ok": degradation_ok,
        "training_episode_acceptance_pass": training_episode_acceptance_pass,
        "step27_acceptance_pass": step27_acceptance_pass if step27_enabled else "",
        "step27_envelope_all_rows_pass": row.get("step27_envelope_all_rows_pass", ""),
        "acceptance_pass": acceptance_pass,
        "accept_max_speed_error": float(args.accept_max_speed_error),
        "accept_min_eta_energy": float(args.accept_min_eta_energy),
        "accept_max_eta_energy": float(args.accept_max_eta_energy),
        "accept_max_current_rms": float(args.accept_max_current_rms),
        "accept_max_p_in_pos": "" if args.accept_max_p_in_pos is None else float(args.accept_max_p_in_pos),
        "degradation_window": int(args.degradation_window),
        "degradation_speed_delta": float(args.degradation_speed_delta),
        "degradation_eta_delta": float(args.degradation_eta_delta),
    }
    registry_row: Dict[str, object] = {
        "motor": row.get("motor", ""),
        "seed": int(row.get("seed", 0)),
        "stage": row.get("stage", ""),
        "best_checkpoint": str(best_ckpt_path),
        "best_checkpoint_sha256": _file_sha256(best_ckpt_path),
        "best_checkpoint_train_internal": str(row.get("best_checkpoint_train_internal", "")),
        "last_checkpoint": str(last_ckpt_path),
        "last_checkpoint_sha256": _file_sha256(last_ckpt_path),
        "episodes_log": str(episodes_log_path),
        "episodes_log_sha256": _file_sha256(episodes_log_path),
        "step27_scan_summary_json": str(row.get("step27_scan_summary_json", "")),
    }
    return summary_row, acceptance_row, snapshot_rows, registry_row


def _base_train_kwargs(args: argparse.Namespace, *, seed: int) -> Dict[str, object]:
    return {
        "episodes": int(args.episodes),
        "episode_steps": int(args.episode_steps),
        "control_mode": str(args.control_mode),
        "w_speed": float(args.w_speed),
        "w_power": float(args.w_power),
        "w_current": None if args.w_current is None else float(args.w_current),
        "w_smooth": float(args.w_smooth),
        "w_mag": float(args.w_mag),
        "w_shaft": float(args.w_shaft),
        "w_eta": float(args.w_eta),
        "w_eta_episode": float(args.w_eta_episode),
        "eta_clip": float(args.eta_clip),
        "id_ref_alpha": float(args.id_ref_alpha),
        "id_ref_rate_limit": None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
        "ai_id_speed_tol": float(args.ai_id_speed_tol),
        "ai_id_speed_tol_rel": None if args.ai_id_speed_tol_rel is None else float(args.ai_id_speed_tol_rel),
        "id_ref_gate_speed_tol": None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
        "id_ref_gate_speed_tol_rel": (
            None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel)
        ),
        "id_ref_gate_min_scale": float(args.id_ref_gate_min_scale),
        "id_ref_gate_exponent": float(args.id_ref_gate_exponent),
        "fast": bool(args.fast),
        "time_budget_min": None if args.time_budget_min is None else float(args.time_budget_min),
        "override_load_torque": False,
        "override_omega_ref": False,
        "ai_id_ref_relative": bool(args.relative),
        "delta_id_max": float(args.delta_id_max),
        "load_torque": None,
        "omega_ref_override": None,
        "scenarios": _parse_csv_list(args.scenarios),
        "scenario_sample": str(args.scenario_sample),
        "omega_ref_range": None,
        "load_torque_range": None,
        "seed": int(seed),
        "sigma_start": float(args.sigma_start),
        "sigma_end": float(args.sigma_end),
        "sigma_decay_episodes": int(args.sigma_decay_episodes),
        "power_warmup_episodes": int(args.power_warmup_episodes),
        "power_ramp_episodes": int(args.power_ramp_episodes),
        "energy_warmup_episodes": int(args.energy_warmup_episodes),
        "energy_ramp_episodes": int(args.energy_ramp_episodes),
        "eval_interval": int(args.eval_interval),
        "eval_scenarios": str(args.eval_scenarios),
        "eval_dt": None if args.eval_dt is None else float(args.eval_dt),
        "eval_t_end": None if args.eval_t_end is None else float(args.eval_t_end),
        "eval_window_frac": float(args.eval_window_frac),
        "eval_error_tol_rel": float(args.eval_error_tol_rel),
        "eval_error_tol_abs": float(args.eval_error_tol_abs),
        "eval_use_total_power": bool(args.eval_use_total_power),
        "include_energy_obs": bool(args.include_energy_obs),
        "include_episode_eta_obs": bool(args.include_episode_eta_obs),
        "update_every_episodes": int(args.update_every_episodes),
        "output_dir": None if not str(args.ai_output_dir).strip() else str(args.ai_output_dir),
        "results_root": None if not str(args.results_root).strip() else str(args.results_root),
    }


def _run_train(
    *,
    motor: MotorSpec,
    seed: int,
    stage: str,
    init_checkpoint: str | None,
    kwargs: Dict[str, object],
    run_root: Path,
) -> Dict[str, object]:
    run_slug = _slug(f"{stage}_seed{int(seed)}_{motor.key}")
    run_kwargs = dict(kwargs)

    # train_ai_id_ref writes best_actor.pth and episode logs under env-name paths.
    # Keep every motor/seed/stage isolated so manifests remain reproducible.
    output_base = run_kwargs.get("output_dir")
    if output_base is None or not str(output_base).strip():
        output_base_path = (run_root / "ai_outputs").resolve()
    else:
        output_base_path = Path(str(output_base)).expanduser().resolve()
    results_base = run_kwargs.get("results_root")
    if results_base is None or not str(results_base).strip():
        results_base_path = (run_root / "results_run").resolve()
    else:
        results_base_path = Path(str(results_base)).expanduser().resolve()

    run_kwargs["output_dir"] = str((output_base_path / run_slug).resolve())
    run_kwargs["results_root"] = str((results_base_path / run_slug).resolve())

    res = train_id_ref(env_config=motor.config_path, init_checkpoint=init_checkpoint, **run_kwargs)
    return {
        "motor": motor.key,
        "seed": int(seed),
        "stage": stage,
        "init_checkpoint": "" if init_checkpoint is None else str(init_checkpoint),
        "best_checkpoint": str(res.get("best", "")),
        "last_checkpoint": str(res.get("last", "")),
        "episodes_log": str(res.get("episodes", "")),
        "run_dir": str(res.get("run_dir", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified 3-motor training pipeline (separate, joint-domain-randomized emulation, fine-tune)."
    )
    parser.add_argument("--mode", choices=["separate-per-motor", "joint-domain-randomized", "fine_tune_per_motor"], required=True)
    parser.add_argument("--motors", default="air56,al31,ao2")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--out-dir", default="outputs/train_3motors_pipeline")
    parser.add_argument("--ai-output-dir", default="")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--base-manifest", default=None, help="Manifest from joint run for fine_tune_per_motor mode.")
    parser.add_argument(
        "--resume-manifest",
        default=None,
        help="Optional previous training_manifest_3motors.json to reuse accepted runs (eval-first policy).",
    )
    parser.add_argument(
        "--eval-first",
        action="store_true",
        help="If set with --resume-manifest, reuse accepted runs when artifacts exist; train only missing/failed runs.",
    )
    parser.add_argument("--joint-cycles", type=int, default=2)
    parser.add_argument("--joint-cycle-episodes", type=int, default=40)
    parser.add_argument("--control-mode", default="ai_id_ref", choices=["ai_id_ref", "ai_current"])
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=2400)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--scenario-sample", default="random", choices=["random", "cycle"])
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--delta-id-max", type=float, default=0.3)
    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--w-power", type=float, default=6.0)
    parser.add_argument("--w-current", type=float, default=None)
    parser.add_argument("--w-smooth", type=float, default=0.05)
    parser.add_argument("--w-mag", type=float, default=0.0)
    parser.add_argument("--w-shaft", type=float, default=2.0)
    parser.add_argument("--w-eta", type=float, default=1.0)
    parser.add_argument("--w-eta-episode", type=float, default=0.0)
    parser.add_argument("--eta-clip", type=float, default=1.2)
    parser.add_argument("--id-ref-alpha", type=float, default=1.0)
    parser.add_argument("--id-ref-rate-limit", type=float, default=None)
    parser.add_argument("--ai-id-speed-tol", type=float, default=0.5)
    parser.add_argument("--ai-id-speed-tol-rel", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=None)
    parser.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    parser.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    parser.add_argument("--sigma-start", type=float, default=0.2)
    parser.add_argument("--sigma-end", type=float, default=0.05)
    parser.add_argument("--sigma-decay-episodes", type=int, default=100)
    parser.add_argument("--power-warmup-episodes", type=int, default=0)
    parser.add_argument("--power-ramp-episodes", type=int, default=50)
    parser.add_argument("--energy-warmup-episodes", type=int, default=0)
    parser.add_argument("--energy-ramp-episodes", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-scenarios", default="speed_step,ramp,load_step")
    parser.add_argument("--eval-dt", type=float, default=None)
    parser.add_argument("--eval-t-end", type=float, default=None)
    parser.add_argument("--eval-window-frac", type=float, default=0.25)
    parser.add_argument("--eval-error-tol-rel", type=float, default=0.05)
    parser.add_argument("--eval-error-tol-abs", type=float, default=0.0)
    parser.add_argument("--eval-use-total-power", action="store_true")
    parser.add_argument("--include-energy-obs", dest="include_energy_obs", action="store_true", default=True)
    parser.add_argument("--no-include-energy-obs", dest="include_energy_obs", action="store_false")
    parser.add_argument("--include-episode-eta-obs", dest="include_episode_eta_obs", action="store_true", default=False)
    parser.add_argument("--update-every-episodes", type=int, default=1)
    parser.add_argument("--accept-max-speed-error", type=float, default=30.0)
    parser.add_argument("--accept-min-eta-energy", type=float, default=0.0)
    parser.add_argument("--accept-max-eta-energy", type=float, default=1.2)
    parser.add_argument("--accept-max-current-rms", type=float, default=10.0)
    parser.add_argument("--accept-max-p-in-pos", type=float, default=None)
    parser.add_argument("--degradation-window", type=int, default=5)
    parser.add_argument("--degradation-speed-delta", type=float, default=2.0)
    parser.add_argument("--degradation-eta-delta", type=float, default=0.05)
    parser.add_argument("--step27-select", action="store_true", help="Select/promote the final checkpoint by Step27 scan.")
    parser.add_argument("--step27-profile", default="canonical", choices=["canonical"])
    parser.add_argument(
        "--step27-seeds",
        default="",
        help="Comma-separated Step27 scan seeds. Empty means reuse --seeds.",
    )
    parser.add_argument("--step27-scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--step27-acceptance-envelopes", default="config/acceptance_envelopes_3motors.json")
    parser.add_argument("--step27-use-envelope-acceptance", dest="step27_use_envelope_acceptance", action="store_true")
    parser.add_argument("--step27-no-envelope-acceptance", dest="step27_use_envelope_acceptance", action="store_false")
    parser.add_argument(
        "--step27-include-canonical-baseline",
        dest="step27_include_canonical_baseline",
        action="store_true",
        help="Include the already accepted canonical release checkpoint in Step27 selection.",
    )
    parser.add_argument(
        "--step27-no-include-canonical-baseline",
        dest="step27_include_canonical_baseline",
        action="store_false",
        help="Evaluate only checkpoints produced by this training run.",
    )
    parser.add_argument("--step27-min-avg-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--step27-min-avg-eta-gain-pct", type=float, default=0.0)
    parser.add_argument("--step27-max-avg-eta-gain-pct", type=float, default=25.0)
    parser.add_argument("--step27-max-err-failures", type=float, default=2.0)
    parser.add_argument("--step27-min-start-stop-saving-pct", type=float, default=-0.5)
    parser.add_argument("--step27-max-start-stop-saving-pct", type=float, default=20.0)
    parser.add_argument("--step27-max-worst-current-peak-ratio", type=float, default=1.30)
    parser.add_argument("--step27-max-worst-current-mean-ratio", type=float, default=1.20)
    parser.add_argument("--step27-top-k", type=int, default=10)
    parser.add_argument("--step27-resume", action="store_true")
    parser.set_defaults(step27_use_envelope_acceptance=True, step27_include_canonical_baseline=True)
    args = parser.parse_args()

    motors = _resolve_motors(args.motors)
    seeds = _parse_int_csv(args.seeds)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_dir).resolve() / f"{timestamp}_{str(args.mode).replace('-', '_')}"
    run_root.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    per_seed_shared: Dict[str, str] = {}
    per_seed_motor_shared: Dict[str, Dict[str, str]] = {}

    base_manifest_data: Dict[str, object] | None = None
    if args.base_manifest:
        base_manifest_path = Path(str(args.base_manifest)).resolve()
        if not base_manifest_path.exists():
            raise FileNotFoundError(base_manifest_path)
        base_manifest_data = _load_json(base_manifest_path)

    resume_manifest_data: Dict[str, object] | None = None
    resume_index: Dict[str, Dict[str, object]] = {}
    resume_acceptance: Dict[str, bool] = {}
    if args.resume_manifest:
        resume_path = Path(str(args.resume_manifest)).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        resume_manifest_data = _load_json(resume_path)
        resume_index = _load_resume_index(resume_manifest_data)
        resume_acceptance = _load_acceptance_index(resume_manifest_data)

    protocol_payload = _build_protocol_snapshot(
        args=args,
        motors=motors,
        seeds=seeds,
        run_root=run_root,
        resume_manifest=args.resume_manifest,
    )
    protocol_path = run_root / "training_protocol_3motors.json"
    protocol_path.write_text(json.dumps(protocol_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _maybe_reuse_or_train(
        *,
        motor: MotorSpec,
        seed: int,
        stage: str,
        init_checkpoint: str | None,
        kwargs: Dict[str, object],
    ) -> Dict[str, object]:
        key = _run_key(motor=motor.key, seed=int(seed), stage=stage)
        can_reuse = bool(args.eval_first and resume_manifest_data is not None)
        if can_reuse:
            prev = resume_index.get(key)
            prev_ok = bool(resume_acceptance.get(key, False))
            if prev is not None and prev_ok and _checkpoint_paths_exist(prev):
                reused = dict(prev)
                reused["reused_from_manifest"] = str(Path(str(args.resume_manifest)).resolve()) if args.resume_manifest else ""
                reused["reused"] = True
                print(f"[train3] mode={args.mode} seed={seed} motor={motor.key} stage={stage} reused=true")
                return reused

        row = _run_train(
            motor=motor,
            seed=seed,
            stage=stage,
            init_checkpoint=init_checkpoint,
            kwargs=kwargs,
            run_root=run_root,
        )
        row = _run_step27_selection(row=row, motor=motor, args=args, run_root=run_root)
        row["reused"] = False
        row["reused_from_manifest"] = ""
        return row

    for seed in seeds:
        kwargs = _base_train_kwargs(args, seed=seed)
        if str(args.mode) == "separate-per-motor":
            for motor in motors:
                row = _maybe_reuse_or_train(
                    motor=motor,
                    seed=seed,
                    stage="separate",
                    init_checkpoint=None,
                    kwargs=kwargs,
                )
                run_rows.append(row)
                print(f"[train3] mode=separate seed={seed} motor={motor.key} best={row['best_checkpoint']}")
            continue

        if str(args.mode) == "joint-domain-randomized":
            carry_ckpt: str | None = None
            motor_ckpts: Dict[str, str] = {}
            for cycle in range(int(args.joint_cycles)):
                kwargs_joint = dict(kwargs)
                kwargs_joint["episodes"] = int(args.joint_cycle_episodes)
                for motor in motors:
                    stage = f"joint_cycle_{cycle + 1}"
                    row = _maybe_reuse_or_train(
                        motor=motor,
                        seed=seed,
                        stage=stage,
                        init_checkpoint=carry_ckpt,
                        kwargs=kwargs_joint,
                    )
                    run_rows.append(row)
                    carry_ckpt = str(row["best_checkpoint"]) if str(row["best_checkpoint"]) else carry_ckpt
                    if str(row["best_checkpoint"]):
                        motor_ckpts[motor.key] = str(row["best_checkpoint"])
                    print(
                        f"[train3] mode=joint seed={seed} cycle={cycle + 1} motor={motor.key} best={row['best_checkpoint']}"
                    )
            per_seed_shared[str(seed)] = "" if carry_ckpt is None else str(carry_ckpt)
            per_seed_motor_shared[str(seed)] = dict(motor_ckpts)
            continue

        # fine_tune_per_motor
        if base_manifest_data is None:
            raise ValueError("Mode fine_tune_per_motor requires --base-manifest from a joint run.")
        shared = dict(base_manifest_data.get("per_seed_shared_checkpoints", {}))
        shared_by_seed_motor = dict(base_manifest_data.get("per_seed_motor_checkpoints", {}))
        motor_shared = shared_by_seed_motor.get(str(seed), {})
        if not isinstance(motor_shared, dict):
            motor_shared = {}
        fallback_seed_ckpt = str(shared.get(str(seed), ""))
        if not fallback_seed_ckpt and not motor_shared:
            raise ValueError(f"Base manifest does not contain shared checkpoint for seed={seed}.")
        for motor in motors:
            init_ckpt_seed = str(motor_shared.get(motor.key, "") or fallback_seed_ckpt)
            if not init_ckpt_seed:
                raise ValueError(f"Base manifest does not contain checkpoint for seed={seed}, motor={motor.key}.")
            row = _maybe_reuse_or_train(
                motor=motor,
                seed=seed,
                stage="fine_tune",
                init_checkpoint=init_ckpt_seed,
                kwargs=kwargs,
            )
            run_rows.append(row)
            print(f"[train3] mode=finetune seed={seed} motor={motor.key} best={row['best_checkpoint']}")

    _write_csv(run_root / "training_runs_3motors.csv", run_rows)
    summary_rows: List[Dict[str, object]] = []
    acceptance_rows: List[Dict[str, object]] = []
    snapshot_rows: List[Dict[str, object]] = []
    registry_rows: List[Dict[str, object]] = []
    for row in run_rows:
        summary_row, acceptance_row, snapshots, registry_row = _summarize_run(row, args)
        summary_rows.append(summary_row)
        acceptance_rows.append(acceptance_row)
        snapshot_rows.extend(snapshots)
        registry_rows.append(registry_row)

    _write_csv(run_root / "training_run_summaries_3motors.csv", summary_rows)
    _write_csv(run_root / "training_acceptance_matrix_3motors.csv", acceptance_rows)
    _write_csv(run_root / "training_eval_snapshots_3motors.csv", snapshot_rows)

    checkpoint_registry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(args.mode),
        "rows": registry_rows,
    }
    checkpoint_registry_path = run_root / "checkpoints_registry_3motors.json"
    checkpoint_registry_path.write_text(json.dumps(checkpoint_registry, ensure_ascii=False, indent=2), encoding="utf-8")

    passed_runs = int(sum(1 for r in acceptance_rows if bool(r.get("acceptance_pass", False))))
    failed_runs = int(len(acceptance_rows) - passed_runs)
    manifest = {
        "timestamp": timestamp,
        "mode": str(args.mode),
        "motors": [m.key for m in motors],
        "seeds": seeds,
        "scenarios": _parse_csv_list(args.scenarios),
        "run_root": str(run_root),
        "base_manifest": None if args.base_manifest is None else str(Path(args.base_manifest).resolve()),
        "resume_manifest": None if args.resume_manifest is None else str(Path(args.resume_manifest).resolve()),
        "eval_first": bool(args.eval_first),
        "step27_select": bool(args.step27_select),
        "step27_profile": str(args.step27_profile),
        "step27_seeds": _parse_int_csv(str(args.step27_seeds)) if str(args.step27_seeds).strip() else seeds,
        "step27_scenarios": _parse_csv_list(str(args.step27_scenarios)),
        "protocol_hash": str(protocol_payload.get("protocol_hash", "")),
        "per_seed_shared_checkpoints": per_seed_shared,
        "per_seed_motor_checkpoints": per_seed_motor_shared,
        "artifacts": {
            "training_runs_csv": str(run_root / "training_runs_3motors.csv"),
            "training_summaries_csv": str(run_root / "training_run_summaries_3motors.csv"),
            "training_acceptance_csv": str(run_root / "training_acceptance_matrix_3motors.csv"),
            "training_eval_snapshots_csv": str(run_root / "training_eval_snapshots_3motors.csv"),
            "checkpoint_registry_json": str(checkpoint_registry_path),
            "training_protocol_json": str(protocol_path),
        },
        "acceptance": {
            "total_runs": int(len(acceptance_rows)),
            "passed_runs": passed_runs,
            "failed_runs": failed_runs,
        },
        "runs": run_rows,
    }
    (run_root / "training_manifest_3motors.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    repro_payload = _build_repro_package_manifest(
        run_root=run_root,
        run_rows=run_rows,
        protocol_hash=str(protocol_payload.get("protocol_hash", "")),
    )
    repro_path = run_root / "training_repro_package_3motors.json"
    repro_path.write_text(json.dumps(repro_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train3] manifest: {run_root / 'training_manifest_3motors.json'}")


if __name__ == "__main__":
    main()
