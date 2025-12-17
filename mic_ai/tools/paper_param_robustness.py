from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.ai.foc_baseline import save_foc_baseline
from mic_ai.ai.train_ai_voltage import _motor_key_from_config, resolve_config_path
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.paper_eval_suite import (
    _aggregate_runs,
    _set_seed,
    _summarize_foc_single,
    _summarize_series,
    _write_csv_by_stage,
    _write_csv_overall,
)
from mic_ai.tools.paper_plots_ai_vs_foc import eval_ai_checkpoint, _load_episode_list


@dataclass(frozen=True)
class ParamCase:
    name: str
    motor_scale: Dict[str, float]
    load_scale: float = 1.0


DEFAULT_CASES: List[ParamCase] = [
    ParamCase("nominal", motor_scale={}, load_scale=1.0),
    ParamCase("Rs_p10", motor_scale={"Rs": 1.10}, load_scale=1.0),
    ParamCase("Rr_p10", motor_scale={"Rr": 1.10}, load_scale=1.0),
    ParamCase("Lm_m10", motor_scale={"Lm": 0.90}, load_scale=1.0),
    ParamCase("J_p20", motor_scale={"J": 1.20}, load_scale=1.0),
    ParamCase("load_p20", motor_scale={}, load_scale=1.20),
    ParamCase("worst", motor_scale={"Rs": 1.10, "Rr": 1.10, "Lm": 0.90, "J": 1.20}, load_scale=1.20),
]


def _apply_case(env_cfg, case: ParamCase):
    motor = env_cfg.motor
    motor_updates: Dict[str, float] = {}
    for key, scale in case.motor_scale.items():
        if not hasattr(motor, key):
            continue
        base_val = float(getattr(motor, key))
        motor_updates[key] = base_val * float(scale)
    if motor_updates:
        motor = replace(motor, **motor_updates)

    sim = env_cfg.sim
    if float(case.load_scale) != 1.0:
        sim = replace(sim, load_torque=float(sim.load_torque) * float(case.load_scale))

    return replace(env_cfg, motor=motor, sim=sim)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness sweep over motor/load parameter variations (paper tables).")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--ai-checkpoint", default="outputs/demo_ai/checkpoints/motor1/last_actor.pth")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--episodes-per-stage", type=int, default=25)
    p.add_argument("--voltage-scale", type=float, default=1.25)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--disable-noise", action="store_true", help="Disable AI measurement noise (and FOC sensor noise).")
    p.add_argument("--sigma-omega", type=float, default=0.05, help="Std of omega measurement noise (rad/s) if noise enabled.")
    p.add_argument("--sigma-i", type=float, default=0.03, help="Std of current measurement noise (A) if noise enabled.")
    p.add_argument("--cases", default="", help="Comma-separated case names to run (default: built-in set).")
    p.add_argument("--out-dir", default="outputs/paper_param_robustness")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_config = resolve_config_path(str(args.env_config))
    motor_key = _motor_key_from_config(str(env_config))

    cfg = load_ai_voltage_config()
    curriculum = get_curriculum_config(cfg)
    n_stages = len(curriculum.get("omega_pu_stages", [0.3, 0.5]))
    episodes_per_stage = int(args.episodes_per_stage)
    episodes = int(episodes_per_stage) * int(n_stages)

    base_env_cfg = make_env_from_config(str(env_config)).env_config
    vdc = float(getattr(getattr(base_env_cfg, "inverter", None), "Vdc", 0.0) or 0.0)
    v_limit_ai = float(args.voltage_scale) * (0.8 * vdc / (3.0**0.5)) if vdc > 0 else None

    if bool(args.disable_noise):
        sigma_omega = 0.0
        sigma_i = 0.0
    else:
        sigma_omega = float(args.sigma_omega)
        sigma_i = float(args.sigma_i)

    noise_enabled = (not bool(args.disable_noise)) and (sigma_omega > 0.0 or sigma_i > 0.0)

    selected = [c.strip() for c in str(args.cases).split(",") if c.strip()]
    cases = [c for c in DEFAULT_CASES if not selected or c.name in selected]
    if not cases:
        raise SystemExit("No cases selected. Use --cases with names from DEFAULT_CASES.")

    summary_rows: List[Dict[str, object]] = []

    for case in cases:
        case_dir = out_dir / f"case_{case.name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        env_cfg_case = _apply_case(base_env_cfg, case)

        ai_runs: List = []
        foc_runs: List = []
        runs_dir = case_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        for k in range(int(args.seeds)):
            seed = int(args.seed0) + k
            _set_seed(seed)
            run_dir = runs_dir / f"seed_{seed:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if noise_enabled:
                foc_json_seed = run_dir / f"foc_baseline_{motor_key}.json"
                save_foc_baseline(
                    config_name=str(env_config),
                    curriculum_config=curriculum,
                    log_path=foc_json_seed,
                    n_episodes_eval=int(episodes_per_stage),
                    episode_steps=int(args.episode_steps),
                    v_limit=v_limit_ai,
                    sigma_omega=float(sigma_omega),
                    sigma_i_abc=float(sigma_i),
                    env_cfg_override=env_cfg_case,
                )
                foc_eps_seed = _load_episode_list(foc_json_seed)
                foc_stat = _summarize_series(foc_eps_seed)
                foc_runs.append(foc_stat)

            ai_json = run_dir / f"ai_eval_{motor_key}.json"
            eval_ai_checkpoint(
                env_config=env_config,
                checkpoint=Path(args.ai_checkpoint),
                out_json=ai_json,
                episodes=episodes,
                episodes_per_stage=episodes_per_stage,
                episode_steps=int(args.episode_steps),
                voltage_scale=float(args.voltage_scale),
                disable_noise=bool(args.disable_noise),
                sigma_omega=float(sigma_omega) if noise_enabled else None,
                sigma_id=float(sigma_i) if noise_enabled else None,
                sigma_iq=float(sigma_i) if noise_enabled else None,
                env_cfg_override=env_cfg_case,
            )
            ai_eps = _load_episode_list(ai_json)
            ai_stat = _summarize_series(ai_eps)
            ai_runs.append(ai_stat)

        ai_overall, ai_by_stage = _aggregate_runs(ai_runs, n_stages=n_stages)

        if noise_enabled and foc_runs:
            foc_overall, foc_by_stage = _aggregate_runs(foc_runs, n_stages=n_stages)
        else:
            foc_json = case_dir / f"foc_baseline_{motor_key}.json"
            save_foc_baseline(
                config_name=str(env_config),
                curriculum_config=curriculum,
                log_path=foc_json,
                n_episodes_eval=int(episodes_per_stage),
                episode_steps=int(args.episode_steps),
                v_limit=v_limit_ai,
                sigma_omega=float(sigma_omega),
                sigma_i_abc=float(sigma_i),
                env_cfg_override=env_cfg_case,
            )
            foc_eps = _load_episode_list(foc_json)
            foc_overall, foc_by_stage = _summarize_foc_single(foc_eps, n_stages=n_stages)

        _write_csv_overall(case_dir / "summary_overall.csv", ai=ai_overall, foc=foc_overall)
        _write_csv_by_stage(case_dir / "summary_by_stage.csv", ai=ai_by_stage, foc=foc_by_stage, n_stages=n_stages)

        meta = {
            "case": case.name,
            "motor_scale": case.motor_scale,
            "load_scale": float(case.load_scale),
            "env_config": str(env_config),
            "ai_checkpoint": str(Path(args.ai_checkpoint).resolve()),
            "episode_steps": int(args.episode_steps),
            "episodes_per_stage": int(episodes_per_stage),
            "stages": int(n_stages),
            "episodes_total": int(episodes),
            "voltage_scale": float(args.voltage_scale),
            "noise_enabled": bool(noise_enabled),
            "sigma_omega": float(sigma_omega),
            "sigma_i": float(sigma_i),
            "seeds": [int(args.seed0) + k for k in range(int(args.seeds))],
        }
        (case_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8-sig")

        for metric in ("i_rms", "p_in_pos", "speed_err"):
            summary_rows.append(
                {
                    "case": case.name,
                    "metric": metric,
                    "ai_mean": ai_overall.get(f"{metric}_mean", 0.0),
                    "ai_ci95": ai_overall.get(f"{metric}_ci95", 0.0),
                    "foc_mean": foc_overall.get(f"{metric}_mean", 0.0),
                    "foc_ci95": foc_overall.get(f"{metric}_ci95", 0.0),
                }
            )

    summary_path = out_dir / "summary_cases.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "metric", "ai_mean", "ai_ci95", "foc_mean", "foc_ci95"])
        w.writeheader()
        w.writerows(summary_rows)

    top_meta = {
        "env_config": str(env_config),
        "ai_checkpoint": str(Path(args.ai_checkpoint).resolve()),
        "cases": [c.name for c in cases],
        "seeds": int(args.seeds),
        "seed0": int(args.seed0),
        "disable_noise": bool(args.disable_noise),
        "sigma_omega": float(sigma_omega),
        "sigma_i": float(sigma_i),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(top_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8-sig")
    print(f"Saved param robustness outputs to {out_dir}")


if __name__ == "__main__":
    main()

