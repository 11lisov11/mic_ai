from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisorConfig  # noqa: E402
from tools.common_utils import json_dump as _json_dump_shared  # noqa: E402
from tools.common_utils import parse_csv_list as _parse_csv_list_shared  # noqa: E402
from tools.common_utils import parse_int_list as _parse_int_list_shared  # noqa: E402
from tools.common_utils import write_csv as _write_csv_shared  # noqa: E402
from tools.step27_pipeline import (  # noqa: E402
    MOTOR_REGISTRY,
    SeedPerturbationSettings,
    _build_handcrafted_candidates,
    _id_ref_eval_params,
    _load_env_and_agent,
    _sample_supervisor_candidate,
    _supervisor_from_env,
    _supervisor_to_candidate,
)
from tools.tune_motor_step27 import _eval_candidate  # noqa: E402


@dataclass(frozen=True)
class RobustThresholds:
    baseline_min_power: float
    baseline_min_eta: float
    baseline_max_err: float
    baseline_min_start_stop: float
    perturb_min_power: float
    perturb_min_eta: float
    perturb_max_err: float
    perturb_min_start_stop: float


DEFAULT_THRESHOLDS: Dict[str, RobustThresholds] = {
    "ao2": RobustThresholds(
        baseline_min_power=0.20,
        baseline_min_eta=0.0,
        baseline_max_err=2.0,
        baseline_min_start_stop=-0.5,
        perturb_min_power=0.00,
        perturb_min_eta=-0.50,
        perturb_max_err=3.0,
        perturb_min_start_stop=-1.0,
    ),
    "al31": RobustThresholds(
        baseline_min_power=0.00,
        baseline_min_eta=0.0,
        baseline_max_err=2.0,
        baseline_min_start_stop=-0.5,
        perturb_min_power=0.00,
        perturb_min_eta=-0.05,
        perturb_max_err=2.0,
        perturb_min_start_stop=-1.0,
    ),
}


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_list(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in _parse_csv_list_shared(text)]


def _json_dump(path: Path, payload: object) -> None:
    _json_dump_shared(path, payload)


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _candidate_fields() -> Tuple[str, ...]:
    return (
        "tag",
        "source",
        "objective",
        "speed_tol_rel",
        "speed_tol_abs",
        "omega_min_pu",
        "update_steps",
        "dither_amp",
        "bias_step",
        "bias_max",
        "shaft_eps",
        "reset_decay",
        "objective_clip",
        "idle_enable",
        "idle_omega_pu",
        "idle_action",
        "idle_exit_boost_steps",
        "idle_exit_action",
        "idle_bias_decay",
        "id_ref_alpha",
        "delta_id_max",
        "id_ref_gate_speed_tol_rel",
        "id_ref_gate_min_scale",
        "id_ref_gate_exponent",
    )


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _candidate_from_row(row: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in _candidate_fields():
        value = row.get(key)
        if key in {"tag", "source", "objective"}:
            out[key] = str(value)
        elif key in {"update_steps", "idle_exit_boost_steps"}:
            out[key] = int(float(value))
        elif key in {"idle_enable"}:
            out[key] = _to_bool(value)
        elif key == "objective_clip":
            out[key] = None if value in {"", None, "None", "none"} else float(value)
        else:
            out[key] = float(value)
    return out


def _thresholds_for_motor(motor: str) -> RobustThresholds:
    return DEFAULT_THRESHOLDS.get(str(motor), DEFAULT_THRESHOLDS["al31"])


def _score_summary(summary: Dict[str, float], thr: RobustThresholds) -> float:
    penalty = 0.0
    # Baseline constraints (hard).
    penalty += max(0.0, thr.baseline_min_power - float(summary["baseline_power"])) * 40.0
    penalty += max(0.0, thr.baseline_min_eta - float(summary["baseline_eta"])) * 22.0
    penalty += max(0.0, float(summary["baseline_err"]) - thr.baseline_max_err) * 12.0
    penalty += max(0.0, thr.baseline_min_start_stop - float(summary["baseline_start_stop"])) * 16.0
    # Perturbed constraints (robustness objective).
    penalty += max(0.0, thr.perturb_min_power - float(summary["perturb_power_min"])) * 48.0
    penalty += max(0.0, thr.perturb_min_eta - float(summary["perturb_eta_min"])) * 30.0
    penalty += max(0.0, float(summary["perturb_err_max"]) - thr.perturb_max_err) * 10.0
    penalty += max(0.0, thr.perturb_min_start_stop - float(summary["perturb_start_stop_min"])) * 10.0
    penalty += max(0.0, float(summary["perturb_peak_ratio_max"]) - 1.20) * 3.0
    penalty += max(0.0, float(summary["perturb_mean_ratio_max"]) - 1.05) * 2.0
    # Reward terms.
    reward = (
        0.55 * float(summary["baseline_power"])
        + 0.30 * float(summary["perturb_power_mean"])
        + 0.10 * float(summary["baseline_eta"])
    )
    return float(penalty - reward)


def _pass_summary(summary: Dict[str, float], thr: RobustThresholds) -> bool:
    return bool(
        float(summary["baseline_power"]) >= thr.baseline_min_power
        and float(summary["baseline_eta"]) >= thr.baseline_min_eta
        and float(summary["baseline_err"]) <= thr.baseline_max_err
        and float(summary["baseline_start_stop"]) >= thr.baseline_min_start_stop
        and float(summary["perturb_power_min"]) >= thr.perturb_min_power
        and float(summary["perturb_eta_min"]) >= thr.perturb_min_eta
        and float(summary["perturb_err_max"]) <= thr.perturb_max_err
        and float(summary["perturb_start_stop_min"]) >= thr.perturb_min_start_stop
    )


def _baseline_guard_pass(summary: Dict[str, float], thr: RobustThresholds) -> bool:
    return bool(
        float(summary["baseline_power"]) >= thr.baseline_min_power
        and float(summary["baseline_eta"]) >= thr.baseline_min_eta
        and float(summary["baseline_err"]) <= thr.baseline_max_err
        and float(summary["baseline_start_stop"]) >= thr.baseline_min_start_stop
    )


def _aggregate_levels(level_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not level_rows:
        raise ValueError("Empty level rows")
    baseline = [r for r in level_rows if float(r["perturb_level"]) <= 0.0]
    if not baseline:
        raise ValueError("Missing baseline row")
    b = baseline[0]
    pert = [r for r in level_rows if float(r["perturb_level"]) > 0.0]
    if not pert:
        pert = [b]

    def _f(rows: List[Dict[str, object]], key: str) -> List[float]:
        return [float(r[key]) for r in rows]

    return {
        "baseline_power": float(b["avg_power_saving_pct"]),
        "baseline_eta": float(b["avg_eta_gain_pct"]),
        "baseline_err": float(b["err_failures"]),
        "baseline_start_stop": float(b["start_stop_power_saving_pct"]),
        "baseline_peak_ratio": float(b["worst_current_peak_ratio"]),
        "baseline_mean_ratio": float(b["worst_current_mean_ratio"]),
        "perturb_power_min": min(_f(pert, "avg_power_saving_pct")),
        "perturb_power_mean": sum(_f(pert, "avg_power_saving_pct")) / max(len(pert), 1),
        "perturb_eta_min": min(_f(pert, "avg_eta_gain_pct")),
        "perturb_eta_mean": sum(_f(pert, "avg_eta_gain_pct")) / max(len(pert), 1),
        "perturb_err_max": max(_f(pert, "err_failures")),
        "perturb_start_stop_min": min(_f(pert, "start_stop_power_saving_pct")),
        "perturb_peak_ratio_max": max(_f(pert, "worst_current_peak_ratio")),
        "perturb_mean_ratio_max": max(_f(pert, "worst_current_mean_ratio")),
    }


def _level_eval_rows(
    *,
    env_cfg: object,
    motor_key: str,
    agent: object,
    candidate: Dict[str, object],
    scenarios: List[str],
    seeds: List[int],
    perturb_levels: List[float],
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    all_levels = [0.0, *[float(x) for x in perturb_levels if float(x) > 0.0]]
    for level in all_levels:
        settings = SeedPerturbationSettings(enabled=bool(level > 0.0), level=float(level))
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=motor_key,
            agent=agent,
            candidate=candidate,
            scenarios=scenarios,
            seeds=seeds,
            window_frac=float(window_frac),
            error_tol_rel=float(error_tol_rel),
            error_tol_abs=float(error_tol_abs),
            use_total_power=bool(use_total_power),
            foc_feedback_mode=str(foc_feedback_mode),
            mic_feedback_mode=str(mic_feedback_mode),
            seed_perturbation=settings,
        )
        rows.append(
            {
                "perturb_level": float(level),
                "tag": str(candidate["tag"]),
                **metrics,
            }
        )
    return rows


def _assignment_repr(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    return repr(str(value))


def _set_or_add_assignment(text: str, key: str, value: object) -> str:
    repl = f"{key} = {_assignment_repr(value)}"
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    if pat.search(text):
        return pat.sub(repl, text)
    if not text.endswith("\n"):
        text += "\n"
    return text + repl + "\n"


def _apply_candidate_to_config(config_path: Path, candidate: Dict[str, object], source_note: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    updates: Dict[str, object] = {
        "ai_eval_id_ref_alpha": float(candidate["id_ref_alpha"]),
        "ai_eval_delta_id_max": float(candidate["delta_id_max"]),
        "ai_eval_id_ref_gate_speed_tol_rel": float(candidate["id_ref_gate_speed_tol_rel"]),
        "ai_eval_id_ref_gate_min_scale": float(candidate["id_ref_gate_min_scale"]),
        "ai_eval_id_ref_gate_exponent": float(candidate["id_ref_gate_exponent"]),
        "ai_eval_supervisor_enabled": True,
        "ai_eval_sup_objective": str(candidate["objective"]),
        "ai_eval_sup_speed_tol_rel": float(candidate["speed_tol_rel"]),
        "ai_eval_sup_speed_tol_abs": float(candidate["speed_tol_abs"]),
        "ai_eval_sup_omega_min": float(candidate["omega_min_pu"]),
        "ai_eval_sup_update": int(candidate["update_steps"]),
        "ai_eval_sup_dither": float(candidate["dither_amp"]),
        "ai_eval_sup_step": float(candidate["bias_step"]),
        "ai_eval_sup_bias_max": float(candidate["bias_max"]),
        "ai_eval_sup_shaft_eps": float(candidate["shaft_eps"]),
        "ai_eval_sup_reset_decay": float(candidate["reset_decay"]),
        "ai_eval_sup_objective_clip": None if candidate["objective_clip"] is None else float(candidate["objective_clip"]),
        "ai_eval_sup_idle_enable": bool(candidate["idle_enable"]),
        "ai_eval_sup_idle_omega_min": float(candidate["idle_omega_pu"]),
        "ai_eval_sup_idle_action": float(candidate["idle_action"]),
        "ai_eval_sup_idle_exit_boost": int(candidate["idle_exit_boost_steps"]),
        "ai_eval_sup_idle_exit_action": float(candidate["idle_exit_action"]),
        "ai_eval_sup_idle_bias_decay": float(candidate["idle_bias_decay"]),
    }
    for key, value in updates.items():
        text = _set_or_add_assignment(text, key, value)
    if source_note:
        marker = "# robust hardening source:"
        note_line = f"{marker} {source_note}"
        if marker in text:
            text = re.sub(rf"(?m)^{re.escape(marker)}.*$", note_line, text)
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += note_line + "\n"
    config_path.write_text(text, encoding="utf-8")


def _profile_for_motor(motor: str, profile_map: Dict[str, str], default_profile: str) -> str:
    m = str(motor).strip().lower()
    if m in profile_map:
        return str(profile_map[m]).strip().lower()
    if m == "ao2":
        return "local_safe"
    return str(default_profile).strip().lower()


def _parse_profile_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = str(text or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        left, right = token.split("=", 1)
        key = left.strip().lower()
        val = right.strip().lower()
        if key and val:
            out[key] = val
    return out


def _validate_out_dir(path: Path) -> Path:
    target = Path(path).resolve()
    root = ROOT.resolve()
    if target == root:
        raise ValueError("Refuse to use repository root as --out-dir. Use outputs/<run_name>.")
    try:
        rel = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"--out-dir must be inside repository: {target}") from exc
    if len(rel.parts) < 2:
        raise ValueError(f"--out-dir is too broad ({target}). Use outputs/<run_name>.")
    if rel.parts[0] != "outputs":
        raise ValueError(f"--out-dir must be under outputs/: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust hardening sweep for AL31/AO2 over perturbation levels without retraining.")
    parser.add_argument("--motors", default="al31,ao2")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--perturb-levels", default="0.2,0.4")
    parser.add_argument("--out-dir", default="outputs/robust_hardening_20260304")
    parser.add_argument("--stage1-trials", type=int, default=60)
    parser.add_argument("--stage2-topk", type=int, default=10)
    parser.add_argument("--stage1-seed", type=int, default=101)
    parser.add_argument("--search-seed", type=int, default=26117)
    parser.add_argument("--sample-profile", default="global", choices=["global", "local_safe"])
    parser.add_argument("--sample-profile-map", default="ao2=local_safe,al31=global")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--apply-config", action="store_true")
    parser.add_argument("--no-apply-config", dest="apply_config", action="store_false")
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--no-use-total-power", dest="use_total_power", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(foc_disable_lut=True)
    parser.set_defaults(apply_config=False)
    parser.set_defaults(use_total_power=True)
    args = parser.parse_args()

    motors = [m for m in _parse_csv_list(args.motors) if m in MOTOR_REGISTRY]
    if not motors:
        raise ValueError("No valid motors selected")
    seeds = _parse_int_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    perturb_levels = sorted({float(max(0.0, x)) for x in _parse_float_list(args.perturb_levels) if float(x) > 0.0})
    if not seeds:
        raise ValueError("Empty seeds list")
    if not scenarios:
        raise ValueError("Empty scenarios list")

    out_dir = _validate_out_dir(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_map = _parse_profile_map(str(args.sample_profile_map))
    global_rng = random.Random(int(args.search_seed))

    all_summary: List[Dict[str, object]] = []
    for motor in motors:
        motor_out = out_dir / str(motor)
        motor_out.mkdir(parents=True, exist_ok=True)
        print(f"[robust:{motor}] start", flush=True)

        if bool(args.dry_run):
            all_summary.append({"motor": motor, "dry_run": True})
            continue

        env_cfg, agent, ckpt = _load_env_and_agent(
            MOTOR_REGISTRY[motor].config_path,
            foc_disable_lut=bool(args.foc_disable_lut),
            require_agent=True,
            motor_key=str(motor),
            checkpoint_registry_path=str(args.checkpoint_registry),
        )
        if agent is None:
            raise RuntimeError(f"AI agent is not loaded for motor={motor}")

        base_sup = _supervisor_from_env(env_cfg)
        if base_sup is None:
            base_sup = AiIdRefSupervisorConfig(enabled=True)
        base_cand = _supervisor_to_candidate(base_sup, tag="baseline", source="config")
        base_id_ref = _id_ref_eval_params(env_cfg)
        base_cand.update(
            {
                "id_ref_alpha": float(base_id_ref["id_ref_alpha"]),
                "delta_id_max": float(base_id_ref["delta_id_max"]),
                "id_ref_gate_speed_tol_rel": float(base_id_ref["id_ref_gate_speed_tol_rel"] or 0.05),
                "id_ref_gate_min_scale": float(base_id_ref["id_ref_gate_min_scale"]),
                "id_ref_gate_exponent": float(base_id_ref["id_ref_gate_exponent"]),
            }
        )

        profile = _profile_for_motor(motor, profile_map, str(args.sample_profile))
        cand_pool: List[Dict[str, object]] = [dict(base_cand)]
        cand_pool.extend(_build_handcrafted_candidates(base_cand))
        for i in range(int(args.stage1_trials)):
            cand_pool.append(
                _sample_supervisor_candidate(
                    global_rng,
                    idx=i + 1,
                    base=base_cand,
                    profile=profile,
                )
            )

        # Stage1: one-seed robust screen.
        thr = _thresholds_for_motor(motor)
        stage1_rows: List[Dict[str, object]] = []
        for idx, cand in enumerate(cand_pool, start=1):
            level_rows = _level_eval_rows(
                env_cfg=env_cfg,
                motor_key=str(motor),
                agent=agent,
                candidate=cand,
                scenarios=scenarios,
                seeds=[int(args.stage1_seed)],
                perturb_levels=perturb_levels,
                window_frac=float(args.window_frac),
                error_tol_rel=float(args.error_tol_rel),
                error_tol_abs=float(args.error_tol_abs),
                use_total_power=bool(args.use_total_power),
                foc_feedback_mode=str(args.foc_feedback_mode),
                mic_feedback_mode=str(args.mic_feedback_mode),
            )
            agg = _aggregate_levels(level_rows)
            row = {**cand, **agg}
            row["robust_score"] = _score_summary(agg, thr)
            row["robust_pass"] = _pass_summary(agg, thr)
            stage1_rows.append(row)
            print(
                f"[robust:{motor}][stage1] {idx}/{len(cand_pool)} tag={cand['tag']} "
                f"score={float(row['robust_score']):+.3f} baseP={float(row['baseline_power']):+.3f}% "
                f"pertMinP={float(row['perturb_power_min']):+.3f}%",
                flush=True,
            )

        stage1_rows.sort(key=lambda r: float(r["robust_score"]))
        top_stage1 = stage1_rows[: max(1, int(args.stage2_topk))]

        # Stage2: full-seed robust check.
        stage2_rows: List[Dict[str, object]] = []
        all_level_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(top_stage1, start=1):
            cand = _candidate_from_row(row)
            level_rows = _level_eval_rows(
                env_cfg=env_cfg,
                motor_key=str(motor),
                agent=agent,
                candidate=cand,
                scenarios=scenarios,
                seeds=seeds,
                perturb_levels=perturb_levels,
                window_frac=float(args.window_frac),
                error_tol_rel=float(args.error_tol_rel),
                error_tol_abs=float(args.error_tol_abs),
                use_total_power=bool(args.use_total_power),
                foc_feedback_mode=str(args.foc_feedback_mode),
                mic_feedback_mode=str(args.mic_feedback_mode),
            )
            for lr in level_rows:
                all_level_rows.append({"motor": str(motor), **lr})
            agg = _aggregate_levels(level_rows)
            out_row = {**cand, **agg}
            out_row["robust_score"] = _score_summary(agg, thr)
            out_row["robust_pass"] = _pass_summary(agg, thr)
            stage2_rows.append(out_row)
            print(
                f"[robust:{motor}][stage2] {idx}/{len(top_stage1)} tag={cand['tag']} "
                f"score={float(out_row['robust_score']):+.3f} baseP={float(out_row['baseline_power']):+.3f}% "
                f"pertMinP={float(out_row['perturb_power_min']):+.3f}% pertMinEta={float(out_row['perturb_eta_min']):+.3f}%",
                flush=True,
            )

        stage2_rows.sort(key=lambda r: float(r["robust_score"]))
        safe_rows = [r for r in stage2_rows if _baseline_guard_pass(r, thr)]
        if safe_rows:
            safe_rows.sort(key=lambda r: float(r["robust_score"]))
            selected = dict(safe_rows[0])
            selection_policy = "safe_baseline_guard"
        else:
            selected = dict(stage2_rows[0])
            selection_policy = "best_penalty_fallback"
        baseline = next((r for r in stage2_rows if str(r.get("tag", "")) == "baseline"), None)
        improved_vs_baseline = bool(
            baseline is not None and float(selected["robust_score"]) < float(baseline["robust_score"]) - 1e-9
        )

        selected_for_config = _candidate_from_row(selected)
        config_path = Path(MOTOR_REGISTRY[motor].config_path).resolve()
        source_note = str((motor_out / f"{motor}_robust_selected.json").as_posix())
        if bool(args.apply_config):
            _apply_candidate_to_config(config_path, selected_for_config, source_note=source_note)

        _write_csv(motor_out / f"{motor}_robust_stage1_rank.csv", stage1_rows)
        _write_csv(motor_out / f"{motor}_robust_stage2_rank.csv", stage2_rows)
        _write_csv(motor_out / f"{motor}_robust_per_level.csv", all_level_rows)

        selected_payload: Dict[str, object] = {
            "motor": str(motor),
            "config_path": str(config_path),
            "checkpoint": "" if ckpt is None else str(ckpt),
            "thresholds": {
                "baseline_min_power": thr.baseline_min_power,
                "baseline_min_eta": thr.baseline_min_eta,
                "baseline_max_err": thr.baseline_max_err,
                "baseline_min_start_stop": thr.baseline_min_start_stop,
                "perturb_min_power": thr.perturb_min_power,
                "perturb_min_eta": thr.perturb_min_eta,
                "perturb_max_err": thr.perturb_max_err,
                "perturb_min_start_stop": thr.perturb_min_start_stop,
            },
            "seeds": [int(x) for x in seeds],
            "scenarios": scenarios,
            "perturb_levels": [float(x) for x in perturb_levels],
            "selected_candidate": selected,
            "baseline_candidate": baseline,
            "improved_vs_baseline": bool(improved_vs_baseline),
            "selection_policy": str(selection_policy),
            "config_applied": bool(args.apply_config),
        }
        _json_dump(motor_out / f"{motor}_robust_selected.json", selected_payload)
        all_summary.append(selected_payload)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "motors": motors,
        "seeds": seeds,
        "scenarios": scenarios,
        "perturb_levels": perturb_levels,
        "out_dir": str(out_dir),
        "results": all_summary,
    }
    _json_dump(out_dir / "robust_hardening_summary.json", summary)
    print(f"saved: {out_dir / 'robust_hardening_summary.json'}")


if __name__ == "__main__":
    main()
