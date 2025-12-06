"""
Batch identification over a set of synthetic motors and plain-text report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import (
    EnvConfig,
    FocParams,
    InverterParams,
    MotorParams,
    ScalarVfParams,
    SimulationParams,
    create_default_env,
)
from vfd_ai.core.env import DirectVoltageEnv
from vfd_ai.ident.auto_id import run_full_identification
from vfd_ai.ident.ident_result import IdentificationResult

# Default identification excitation (can be overridden per motor if fields exist)
IDENT_DEFAULTS = {
    "ident_u_d_step": 150.0,
    "ident_total_time": 1.0,
    "ident_u_q_step": 200.0,
    "ident_locked_total_time": 1.0,
    "ident_torque_ref": 1.5,
    "ident_runup_time": 0.8,
    "ident_coast_time": 0.8,
}


def build_env_from_motor(motor_dict: Dict) -> object:
    base = create_default_env()
    Ls_total = float(motor_dict["Ls"])
    Lr_total = float(motor_dict["Lr"])
    Lm_val = float(motor_dict["Lm"])
    Lm_val = min(Lm_val, 0.95 * min(Ls_total, Lr_total))  # clamp to avoid negative sigmas
    Ls_sigma = max(Ls_total - Lm_val, 1e-6)
    Lr_sigma = max(Lr_total - Lm_val, 1e-6)

    motor = MotorParams(
        Rs=float(motor_dict["Rs"]),
        Rr=float(motor_dict["Rr"]),
        Ls_sigma=Ls_sigma,
        Lr_sigma=Lr_sigma,
        Lm=Lm_val,
        J=float(motor_dict["J"]),
        B=float(motor_dict["B"]),
        p=int(motor_dict.get("p", base.motor.p)),
    )
    inverter = InverterParams(
        Vdc=float(motor_dict.get("Vdc", base.inverter.Vdc)),
        f_pwm=float(motor_dict.get("f_pwm", base.inverter.f_pwm)),
    )
    sim = SimulationParams(
        t_end=max(1.2, base.sim.t_end),
        dt=base.sim.dt,
        mode=base.sim.mode,
        scenario_name=base.sim.scenario_name,
        save_prefix=base.sim.save_prefix,
        load_torque=base.sim.load_torque,
    )
    env_cfg = EnvConfig(
        motor=motor,
        inverter=inverter,
        scalar_vf=base.scalar_vf if isinstance(base.scalar_vf, ScalarVfParams) else base.scalar_vf,
        foc=base.foc if isinstance(base.foc, FocParams) else base.foc,
        sim=sim,
    )
    env = DirectVoltageEnv(env_cfg)
    Vdc = inverter.Vdc
    ident_u_d_step = motor_dict.get("ident_u_d_step", max(80.0, 0.9 * Vdc / math.sqrt(3.0)))
    ident_u_q_step = motor_dict.get("ident_u_q_step", max(100.0, 0.95 * Vdc / math.sqrt(3.0)))
    settings = {
        **IDENT_DEFAULTS,
        "ident_u_d_step": ident_u_d_step,
        "ident_u_q_step": ident_u_q_step,
    }
    for k, v in settings.items():
        setattr(env, k, motor_dict.get(k, v))
    return env


def run_one(motor_dict: Dict) -> Tuple[IdentificationResult, Dict[str, float]]:
    # Adaptive retries by increasing test excitation/duration on failures
    u_d_step = motor_dict.get("ident_u_d_step")
    u_q_step = motor_dict.get("ident_u_q_step")
    t_d = motor_dict.get("ident_total_time", IDENT_DEFAULTS["ident_total_time"])
    t_q = motor_dict.get("ident_locked_total_time", IDENT_DEFAULTS["ident_locked_total_time"])

    last_exc: Exception | None = None
    for _ in range(4):
        md = dict(motor_dict)
        if u_d_step is not None:
            md["ident_u_d_step"] = u_d_step
        if u_q_step is not None:
            md["ident_u_q_step"] = u_q_step
        md["ident_total_time"] = t_d
        md["ident_locked_total_time"] = t_q

        env = build_env_from_motor(md)
        try:
            result = run_full_identification(env, motor_name=md.get("name", "motor"), source="simulation")
            est = result.estimated.as_dict()
            true = motor_dict
            rel_err: Dict[str, float] = {}
            for k, v in est.items():
                if v is None or k not in true:
                    continue
                t = true[k]
                if t == 0:
                    continue
                rel_err[k] = abs(float(v) - float(t)) / abs(float(t)) * 100.0
            return result, rel_err
        except Exception as exc:
            last_exc = exc
            # boost excitation and time for next attempt
            if u_d_step is None:
                u_d_step = IDENT_DEFAULTS["ident_u_d_step"]
            if u_q_step is None:
                u_q_step = IDENT_DEFAULTS["ident_u_q_step"]
            u_d_step *= 1.8
            u_q_step *= 1.4
            t_d *= 1.5
            t_q *= 1.4
            continue
    # if all attempts failed
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown failure in run_one")


def format_line(name: str, true: Dict, est: Dict, err: Dict) -> str:
    def fmt(val):
        return "" if val is None else f"{float(val):.6g}"

    fields = ["Rs", "Rr", "Ls", "Lr", "Lm", "J", "B"]
    parts = [name]
    for f in fields:
        parts.append(fmt(true.get(f)))
    for f in fields:
        parts.append(fmt(est.get(f)))
    for f in fields:
        parts.append("" if f not in err else f"{err[f]:.3f}")
    return "\t".join(parts)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch identification over motors list")
    p.add_argument("--motors-file", required=True, help="Path to motors JSONL file (one JSON per line)")
    p.add_argument("--output", required=True, help="Path to save plain-text report")
    p.add_argument("--limit", type=int, default=None, help="Limit number of motors (for quick runs)")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    motors_path = Path(args.motors_file)
    if not motors_path.exists():
        print(f"Motors file not found: {motors_path}")
        return 1
    lines = motors_path.read_text(encoding="utf-8").splitlines()
    if args.limit is not None:
        lines = lines[: args.limit]

    headers = (
        ["name"]
        + [f"{f}_true" for f in ["Rs", "Rr", "Ls", "Lr", "Lm", "J", "B"]]
        + [f"{f}_est" for f in ["Rs", "Rr", "Ls", "Lr", "Lm", "J", "B"]]
        + [f"{f}_err_percent" for f in ["Rs", "Rr", "Ls", "Lr", "Lm", "J", "B"]]
    )
    out_lines = ["\t".join(headers)]

    for idx, line in enumerate(lines, 1):
        motor = json.loads(line)
        name = motor.get("name", f"motor_{idx:03d}")
        print(f"[batch] {idx}/{len(lines)}: {name}")
        try:
            result, rel_err = run_one(motor)
            out_lines.append(format_line(name, motor, result.estimated.as_dict(), rel_err))
        except Exception as exc:  # pragma: no cover - batch resilience
            print(f"[batch] {name} FAILED: {exc}")
            out_lines.append(f"{name}\tERROR\t{exc}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[batch] Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
