from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisorConfig  # noqa: E402
from tools.step27_pipeline import (  # noqa: E402
    MOTOR_REGISTRY,
    SeedPerturbationSettings,
    _aggregate_rows,
    _build_handcrafted_candidates,
    _candidate_to_supervisor,
    _id_ref_eval_params,
    _load_env_and_agent,
    _sample_supervisor_candidate,
    _sensorless_params,
    _simulate_rows,
    _supervisor_from_env,
    _supervisor_to_candidate,
)
from tools.common_utils import json_dump as _json_dump_shared
from tools.common_utils import parse_csv_list as _parse_csv_list_shared
from tools.common_utils import parse_int_list as _parse_int_list_shared
from tools.common_utils import write_csv as _write_csv_shared


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_list(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _json_dump(path: Path, payload: object) -> None:
    _json_dump_shared(path, payload)


def _score(metrics: Dict[str, float], *, min_power: float, min_eta: float, max_err: float, min_start_stop: float) -> float:
    power = float(metrics.get("avg_power_saving_pct", 0.0))
    eta = float(metrics.get("avg_eta_gain_pct", 0.0))
    err = float(metrics.get("err_failures", 0.0))
    start_stop = float(metrics.get("start_stop_power_saving_pct", 0.0))
    peak_ratio = float(metrics.get("worst_current_peak_ratio", 1.0))
    mean_ratio = float(metrics.get("worst_current_mean_ratio", 1.0))

    penalty = 0.0
    if power < min_power:
        penalty += 30.0 * (min_power - power)
    if eta < min_eta:
        penalty += 20.0 * (min_eta - eta)
    if err > max_err:
        penalty += 8.0 * (err - max_err)
    if start_stop < min_start_stop:
        penalty += 12.0 * (min_start_stop - start_stop)
    penalty += 3.0 * max(0.0, peak_ratio - 1.15)
    penalty += 2.0 * max(0.0, mean_ratio - 1.03)
    return float(penalty)


def _pass(metrics: Dict[str, float], *, min_power: float, min_eta: float, max_err: float, min_start_stop: float) -> bool:
    return bool(
        float(metrics.get("avg_power_saving_pct", 0.0)) >= min_power
        and float(metrics.get("avg_eta_gain_pct", 0.0)) >= min_eta
        and float(metrics.get("err_failures", 0.0)) <= max_err
        and float(metrics.get("start_stop_power_saving_pct", 0.0)) >= min_start_stop
    )


def _eval_candidate(
    *,
    env_cfg: object,
    motor_key: str,
    agent: object,
    candidate: Dict[str, object],
    scenarios: List[str],
    seeds: List[int],
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
    seed_perturbation: SeedPerturbationSettings,
) -> Dict[str, float]:
    sup_cfg = _candidate_to_supervisor(candidate)
    id_ref = _id_ref_eval_params(env_cfg)
    id_ref["id_ref_alpha"] = float(candidate["id_ref_alpha"])
    id_ref["delta_id_max"] = float(candidate["delta_id_max"])
    id_ref["id_ref_gate_speed_tol_rel"] = float(candidate["id_ref_gate_speed_tol_rel"])
    id_ref["id_ref_gate_min_scale"] = float(candidate["id_ref_gate_min_scale"])
    id_ref["id_ref_gate_exponent"] = float(candidate["id_ref_gate_exponent"])
    sensorless = _sensorless_params(env_cfg)

    per_seed: List[Dict[str, float]] = []
    for seed in seeds:
        rows = _simulate_rows(
            env_cfg=env_cfg,
            motor_key=str(motor_key),
            agent=agent,
            scenarios=scenarios,
            seed=int(seed),
            window_frac=float(window_frac),
            error_tol_rel=float(error_tol_rel),
            error_tol_abs=float(error_tol_abs),
            use_total_power=bool(use_total_power),
            foc_feedback_mode=str(foc_feedback_mode),
            mic_feedback_mode=str(mic_feedback_mode),
            controller="MIC",
            id_ref_params=id_ref,
            supervisor_cfg=sup_cfg,
            sensorless=sensorless,
            seed_perturbation=seed_perturbation,
            mic_mode="ai",
            mic_rule_params=None,
        )
        per_seed.append(_aggregate_rows(rows))

    out = {
        "avg_power_saving_pct": _mean([float(x["avg_power_saving_pct"]) for x in per_seed]),
        "avg_eta_gain_pct": _mean([float(x["avg_eta_gain_pct"]) for x in per_seed]),
        "err_failures": _mean([float(x["err_failures"]) for x in per_seed]),
        "start_stop_power_saving_pct": _mean([float(x["start_stop_power_saving_pct"]) for x in per_seed]),
        "worst_current_peak_ratio": max(float(x["worst_current_peak_ratio"]) for x in per_seed),
        "worst_current_mean_ratio": max(float(x["worst_current_mean_ratio"]) for x in per_seed),
        "avg_power_saving_pct_min_seed": min(float(x["avg_power_saving_pct"]) for x in per_seed),
        "avg_eta_gain_pct_min_seed": min(float(x["avg_eta_gain_pct"]) for x in per_seed),
        "err_failures_max_seed": max(float(x["err_failures"]) for x in per_seed),
        "start_stop_power_saving_pct_min_seed": min(float(x["start_stop_power_saving_pct"]) for x in per_seed),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune AI MIC supervisor/id_ref params for selected motor using step27 metrics.")
    parser.add_argument("--motor", required=True, choices=sorted(MOTOR_REGISTRY.keys()))
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--out-dir", default="outputs/motor_tuning_step27")
    parser.add_argument("--stage1-trials", type=int, default=40)
    parser.add_argument("--stage2-topk", type=int, default=8)
    parser.add_argument("--stage1-seed", type=int, default=101)
    parser.add_argument("--stage2-seed", type=int, default=101)
    parser.add_argument("--search-seed", type=int, default=26027)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--sample-profile", default="global", choices=["global", "local_safe"])
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--no-use-total-power", dest="use_total_power", action="store_false")
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--min-avg-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--min-avg-eta-gain-pct", type=float, default=0.0)
    parser.add_argument("--max-err-failures", type=float, default=2.0)
    parser.add_argument("--min-start-stop-saving-pct", type=float, default=-0.5)
    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    if not seeds:
        raise ValueError("Empty seeds list")
    if not scenarios:
        raise ValueError("Empty scenarios list")

    motor = str(args.motor)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_seed = int(args.stage1_seed)
    stage2_seed = int(args.stage2_seed)

    env_cfg, agent, ckpt = _load_env_and_agent(
        MOTOR_REGISTRY[motor].config_path,
        foc_disable_lut=bool(args.foc_disable_lut),
        require_agent=True,
        motor_key=str(motor),
        checkpoint_registry_path=str(args.checkpoint_registry),
    )
    if agent is None:
        raise RuntimeError("AI agent was not loaded.")

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

    rng = random.Random(int(args.search_seed))
    candidates: List[Dict[str, object]] = [dict(base_cand)]
    candidates.extend(_build_handcrafted_candidates(base_cand))
    for i in range(int(args.stage1_trials)):
        candidates.append(
            _sample_supervisor_candidate(
                rng,
                idx=i + 1,
                base=base_cand,
                profile=str(args.sample_profile),
            )
        )

    seed_perturb = SeedPerturbationSettings(
        enabled=bool(args.seed_perturbation),
        level=float(max(0.0, float(args.seed_perturb_level))),
    )

    stage1_rows: List[Dict[str, object]] = []
    for i, cand in enumerate(candidates, start=1):
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=motor,
            agent=agent,
            candidate=cand,
            scenarios=scenarios,
            seeds=[stage1_seed],
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            seed_perturbation=seed_perturb,
        )
        row = {**cand, **metrics}
        row["score"] = _score(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
        )
        row["acceptance_pass"] = _pass(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
        )
        stage1_rows.append(row)
        print(
            f"[tune:{motor}][stage1] {i}/{len(candidates)} tag={row['tag']} "
            f"power={float(row['avg_power_saving_pct']):.3f}% eta={float(row['avg_eta_gain_pct']):.3f}% "
            f"start_stop={float(row['start_stop_power_saving_pct']):.3f}% err={float(row['err_failures']):.1f}"
        )

    stage1_rows.sort(key=lambda r: float(r["score"]))
    top = stage1_rows[: max(1, int(args.stage2_topk))]

    stage2_rows: List[Dict[str, object]] = []
    for i, cand in enumerate(top, start=1):
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=motor,
            agent=agent,
            candidate=cand,
            scenarios=scenarios,
            seeds=seeds,
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            seed_perturbation=seed_perturb,
        )
        row = {**cand, **metrics}
        row["score"] = _score(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
        )
        row["acceptance_pass"] = _pass(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
        )
        stage2_rows.append(row)
        print(
            f"[tune:{motor}][stage2] {i}/{len(top)} tag={row['tag']} "
            f"power={float(row['avg_power_saving_pct']):.3f}% eta={float(row['avg_eta_gain_pct']):.3f}% "
            f"start_stop={float(row['start_stop_power_saving_pct']):.3f}% err={float(row['err_failures']):.1f}"
        )

    stage2_rows.sort(key=lambda r: float(r["score"]))
    best = stage2_rows[0] if stage2_rows else stage1_rows[0]

    _write_csv(out_dir / f"{motor}_stage1_rank.csv", stage1_rows)
    _write_csv(out_dir / f"{motor}_stage2_rank.csv", stage2_rows)
    summary = {
        "motor": motor,
        "checkpoint": "" if ckpt is None else str(ckpt),
        "config_path": MOTOR_REGISTRY[motor].config_path,
        "seeds": seeds,
        "scenarios": scenarios,
        "stage1_trials": int(args.stage1_trials),
        "stage2_topk": int(args.stage2_topk),
        "stage1_seed": stage1_seed,
        "stage2_seed": stage2_seed,
        "search_seed": int(args.search_seed),
        "acceptance": {
            "min_avg_power_saving_pct": float(args.min_avg_power_saving_pct),
            "min_avg_eta_gain_pct": float(args.min_avg_eta_gain_pct),
            "max_err_failures": float(args.max_err_failures),
            "min_start_stop_saving_pct": float(args.min_start_stop_saving_pct),
        },
        "sample_profile": str(args.sample_profile),
        "best": best,
        "top_stage2": stage2_rows,
    }
    _json_dump(out_dir / f"{motor}_tuning_summary.json", summary)
    _json_dump(out_dir / f"{motor}_best_candidate.json", best)
    print(f"[tune:{motor}] done. best={best.get('tag')} score={float(best.get('score', 0.0)):.3f}")
    print(f"[tune:{motor}] summary={out_dir / f'{motor}_tuning_summary.json'}")


if __name__ == "__main__":
    main()
