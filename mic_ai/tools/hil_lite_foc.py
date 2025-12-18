from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from control.vector_foc import FocController
from mic_ai.core.env import make_env_from_config
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import dq_to_abc
from simulation.scenarios import get_scenario


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIL-lite loop: split plant/controller with optional delay + noise.")
    parser.add_argument("--env-config", required=True, help="Path to config/*.py with ENV.")
    parser.add_argument("--out", default="out/hil_lite_foc.npz", help="Output .npz path (default: out/hil_lite_foc.npz).")
    parser.add_argument("--steps", type=int, default=0, help="Override number of steps (0 = use t_end/dt).")
    parser.add_argument("--dt", type=float, default=0.0, help="Override dt (0 = use ENV.sim.dt).")
    parser.add_argument("--meas-delay-steps", type=int, default=0, help="Measurement delay in control steps.")
    parser.add_argument("--cmd-delay-steps", type=int, default=0, help="Command (vd/vq) delay in control steps.")
    parser.add_argument("--sigma-omega", type=float, default=0.0, help="Speed measurement noise std (rad/s).")
    parser.add_argument("--sigma-i", type=float, default=0.0, help="Current measurement noise std per phase (A).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise.")
    return parser.parse_args(list(argv) if argv is not None else None)


def run_hil_lite_foc(
    env_config_path: str | Path,
    out_path: str | Path,
    steps: int = 0,
    dt: float = 0.0,
    meas_delay_steps: int = 0,
    cmd_delay_steps: int = 0,
    sigma_omega: float = 0.0,
    sigma_i: float = 0.0,
    seed: int = 0,
) -> Path:
    env = make_env_from_config(str(env_config_path))
    env_cfg = env.env_config
    dt_sim = float(dt) if dt and dt > 0 else float(env_cfg.sim.dt)
    n_steps = int(steps) if steps and steps > 0 else int(max(1, env_cfg.sim.t_end / dt_sim))

    omega_ref_func, load_torque_func = get_scenario(env_cfg.sim.scenario_name, env_cfg)

    controller = FocController(env_cfg.foc, env_cfg.motor, dt=dt_sim)
    controller.reset()
    inverter = IdealInverter(env_cfg.inverter)
    motor = InductionMotorModel(env_cfg.motor)

    meas_delay_steps = max(int(meas_delay_steps), 0)
    cmd_delay_steps = max(int(cmd_delay_steps), 0)

    rng = np.random.default_rng(int(seed))

    # Each measurement entry: (omega_meas, (i_a, i_b, i_c)).
    meas_queue: deque[Tuple[float, Tuple[float, float, float]]] = deque(
        [(0.0, (0.0, 0.0, 0.0))] * (meas_delay_steps + 1),
        maxlen=meas_delay_steps + 1,
    )
    # Each command entry: (v_d, v_q, theta_e_used, omega_syn).
    cmd_queue: deque[Tuple[float, float, float, float]] = deque(
        [(0.0, 0.0, 0.0, 0.0)] * (cmd_delay_steps + 1),
        maxlen=cmd_delay_steps + 1,
    )

    results = {
        "t": [],
        "omega_m": [],
        "omega_ref": [],
        "T_e": [],
        "load_torque": [],
        "i_a": [],
        "i_b": [],
        "i_c": [],
        "P_in": [],
        "P_out": [],
        # Extra signals (for debugging/analysis)
        "v_d_cmd": [],
        "v_q_cmd": [],
        "v_d_apply": [],
        "v_q_apply": [],
        "theta_e_cmd": [],
        "omega_syn_cmd": [],
    }

    t_now = 0.0
    theta_mech = 0.0
    last_torque = 0.0
    last_i_abc_true = (0.0, 0.0, 0.0)

    for step_idx in range(n_steps):
        omega_ref = float(omega_ref_func(t_now))
        load_torque = float(load_torque_func(t_now))

        omega_meas, i_abc_meas = meas_queue[0]
        v_d_cmd, v_q_cmd, theta_e_used, omega_syn_cmd, _info = controller.step(
            t=t_now,
            omega_ref=omega_ref,
            omega_m=float(omega_meas),
            i_abc=i_abc_meas,
            torque_e=float(last_torque),
            theta_mech=float(theta_mech),
        )
        cmd_queue.append((float(v_d_cmd), float(v_q_cmd), float(theta_e_used), float(omega_syn_cmd)))

        v_d_apply, v_q_apply, theta_e_apply, omega_syn_apply = cmd_queue[0]
        v_abc, (v_d_out, v_q_out) = inverter.output(v_d_apply, v_q_apply, theta_e_apply)
        state, i_d, i_q, torque_e, omega_m = motor.step(
            float(v_d_out),
            float(v_q_out),
            load_torque=float(load_torque),
            dt=float(dt_sim),
            omega_syn=float(omega_syn_apply),
        )
        theta_mech += float(omega_m) * float(dt_sim)
        i_abc_true = dq_to_abc(float(i_d), float(i_q), float(theta_e_apply))
        last_i_abc_true = tuple(float(x) for x in i_abc_true)
        last_torque = float(torque_e)

        # Measurements to controller (noise + queue delay).
        omega_meas_next = float(omega_m) + float(rng.normal(0.0, sigma_omega)) if sigma_omega > 0 else float(omega_m)
        if sigma_i > 0:
            i_abc_meas_next = tuple(float(x + rng.normal(0.0, sigma_i)) for x in last_i_abc_true)
        else:
            i_abc_meas_next = last_i_abc_true
        meas_queue.append((omega_meas_next, i_abc_meas_next))

        p_in = float(v_abc[0] * last_i_abc_true[0] + v_abc[1] * last_i_abc_true[1] + v_abc[2] * last_i_abc_true[2])
        p_out = float(last_torque * float(omega_m))

        t_now += float(dt_sim)
        results["t"].append(t_now)
        results["omega_m"].append(float(state.omega_m))
        results["omega_ref"].append(float(omega_ref))
        results["T_e"].append(float(last_torque))
        results["load_torque"].append(float(load_torque))
        results["i_a"].append(float(last_i_abc_true[0]))
        results["i_b"].append(float(last_i_abc_true[1]))
        results["i_c"].append(float(last_i_abc_true[2]))
        results["P_in"].append(p_in)
        results["P_out"].append(p_out)

        results["v_d_cmd"].append(float(v_d_cmd))
        results["v_q_cmd"].append(float(v_q_cmd))
        results["v_d_apply"].append(float(v_d_apply))
        results["v_q_apply"].append(float(v_q_apply))
        results["theta_e_cmd"].append(float(theta_e_used))
        results["omega_syn_cmd"].append(float(omega_syn_cmd))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "env": asdict(env_cfg),
        "hil": {
            "dt": dt_sim,
            "steps": n_steps,
            "meas_delay_steps": meas_delay_steps,
            "cmd_delay_steps": cmd_delay_steps,
            "sigma_omega": float(sigma_omega),
            "sigma_i": float(sigma_i),
            "seed": int(seed),
        },
    }
    meta_bytes = np.array(json.dumps(meta).encode("utf-8"), dtype=np.bytes_)
    np.savez(out_path, **{k: np.asarray(v) for k, v in results.items()}, meta=meta_bytes)
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out = run_hil_lite_foc(
        env_config_path=args.env_config,
        out_path=args.out,
        steps=args.steps,
        dt=args.dt,
        meas_delay_steps=args.meas_delay_steps,
        cmd_delay_steps=args.cmd_delay_steps,
        sigma_omega=args.sigma_omega,
        sigma_i=args.sigma_i,
        seed=args.seed,
    )
    print(f"Saved HIL-lite run to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

