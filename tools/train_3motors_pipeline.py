from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
        "eval_interval": int(args.eval_interval),
        "eval_scenarios": str(args.eval_scenarios),
        "eval_dt": None if args.eval_dt is None else float(args.eval_dt),
        "eval_t_end": None if args.eval_t_end is None else float(args.eval_t_end),
        "eval_window_frac": float(args.eval_window_frac),
        "eval_error_tol_rel": float(args.eval_error_tol_rel),
        "eval_error_tol_abs": float(args.eval_error_tol_abs),
        "eval_use_total_power": bool(args.eval_use_total_power),
        "include_energy_obs": bool(args.include_energy_obs),
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
) -> Dict[str, object]:
    res = train_id_ref(env_config=motor.config_path, init_checkpoint=init_checkpoint, **kwargs)
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
    parser.add_argument("--joint-cycles", type=int, default=2)
    parser.add_argument("--joint-cycle-episodes", type=int, default=40)
    parser.add_argument("--control-mode", default="ai_id_ref", choices=["ai_id_ref", "ai_current"])
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=200)
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
    parser.add_argument("--update-every-episodes", type=int, default=1)
    args = parser.parse_args()

    motors = _resolve_motors(args.motors)
    seeds = _parse_int_csv(args.seeds)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_dir).resolve() / f"{timestamp}_{str(args.mode).replace('-', '_')}"
    run_root.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    per_seed_shared: Dict[str, str] = {}

    base_manifest_data: Dict[str, object] | None = None
    if args.base_manifest:
        base_manifest_path = Path(str(args.base_manifest)).resolve()
        if not base_manifest_path.exists():
            raise FileNotFoundError(base_manifest_path)
        base_manifest_data = _load_json(base_manifest_path)

    for seed in seeds:
        kwargs = _base_train_kwargs(args, seed=seed)
        if str(args.mode) == "separate-per-motor":
            for motor in motors:
                row = _run_train(motor=motor, seed=seed, stage="separate", init_checkpoint=None, kwargs=kwargs)
                run_rows.append(row)
                print(f"[train3] mode=separate seed={seed} motor={motor.key} best={row['best_checkpoint']}")
            continue

        if str(args.mode) == "joint-domain-randomized":
            carry_ckpt: str | None = None
            for cycle in range(int(args.joint_cycles)):
                kwargs_joint = dict(kwargs)
                kwargs_joint["episodes"] = int(args.joint_cycle_episodes)
                for motor in motors:
                    stage = f"joint_cycle_{cycle + 1}"
                    row = _run_train(motor=motor, seed=seed, stage=stage, init_checkpoint=carry_ckpt, kwargs=kwargs_joint)
                    run_rows.append(row)
                    carry_ckpt = str(row["best_checkpoint"]) if str(row["best_checkpoint"]) else carry_ckpt
                    print(
                        f"[train3] mode=joint seed={seed} cycle={cycle + 1} motor={motor.key} best={row['best_checkpoint']}"
                    )
            per_seed_shared[str(seed)] = "" if carry_ckpt is None else str(carry_ckpt)
            continue

        # fine_tune_per_motor
        if base_manifest_data is None:
            raise ValueError("Mode fine_tune_per_motor requires --base-manifest from a joint run.")
        shared = dict(base_manifest_data.get("per_seed_shared_checkpoints", {}))
        init_ckpt_seed = str(shared.get(str(seed), ""))
        if not init_ckpt_seed:
            raise ValueError(f"Base manifest does not contain shared checkpoint for seed={seed}.")
        for motor in motors:
            row = _run_train(
                motor=motor,
                seed=seed,
                stage="fine_tune",
                init_checkpoint=init_ckpt_seed,
                kwargs=kwargs,
            )
            run_rows.append(row)
            print(f"[train3] mode=finetune seed={seed} motor={motor.key} best={row['best_checkpoint']}")

    _write_csv(run_root / "training_runs_3motors.csv", run_rows)
    manifest = {
        "timestamp": timestamp,
        "mode": str(args.mode),
        "motors": [m.key for m in motors],
        "seeds": seeds,
        "scenarios": _parse_csv_list(args.scenarios),
        "run_root": str(run_root),
        "base_manifest": None if args.base_manifest is None else str(Path(args.base_manifest).resolve()),
        "per_seed_shared_checkpoints": per_seed_shared,
        "runs": run_rows,
    }
    (run_root / "training_manifest_3motors.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[train3] manifest: {run_root / 'training_manifest_3motors.json'}")


if __name__ == "__main__":
    main()
