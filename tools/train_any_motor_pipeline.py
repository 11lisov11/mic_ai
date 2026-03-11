from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import estimate_id_ref_from_nameplate, estimate_motor_params_from_nameplate
from mic_ai.ai.train_ai_id_ref import train as train_ai_id_ref
from mic_ai.ident.auto_id import run_full_identification
from mic_ai.ident.io import load_test_data, save_ident_result
from mic_ai.ident.motor_params import MotorParamsEstimated
from tools.common_utils import json_dump, parse_csv_list, write_csv


BENCHMARK_MOTOR_CONFIGS: Dict[str, str] = {
    "air56": "config/env_research_air56_025kw.py",
    "al31": "config/env_research_al31_4_06kw.py",
    "ao2": "config/env_research_ao2_32_4_3kw.py",
}


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    value = value.strip("_")
    return value or "motor"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8-sig")))


def _pick_first(payload: Dict[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _to_float(value: object | None, *, field: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"Missing required field: {field}")
        return float(default)
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Field '{field}' must be float-like, got={value!r}") from exc


def _to_int(value: object | None, *, field: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"Missing required field: {field}")
        return int(default)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Field '{field}' must be int-like, got={value!r}") from exc


def _normalize_nameplate(passport: Dict[str, object]) -> Dict[str, object]:
    p_w = _pick_first(passport, ("P_n", "p_n", "P_w", "p_w"))
    if p_w is None:
        p_kw = _pick_first(passport, ("P_kW", "p_kw", "power_kw", "rated_power_kw"))
        if p_kw is not None:
            p_w = float(p_kw) * 1000.0

    u_ll = _pick_first(passport, ("U_ll", "u_ll", "V_ll", "voltage_ll"))
    i_n = _pick_first(passport, ("I_n", "i_n", "current_n", "rated_current"))
    cos_phi = _pick_first(passport, ("cos_phi_n", "cos_phi", "power_factor"))
    eta = _pick_first(passport, ("eta_n", "eta", "efficiency"))
    f_n = _pick_first(passport, ("f_n", "frequency_hz", "frequency"))
    poles = _pick_first(passport, ("p", "pole_pairs"))
    n_rated = _pick_first(passport, ("n_rated", "rated_rpm", "rpm_n"))
    connection = _pick_first(passport, ("connection", "winding_connection"))
    inertia = _pick_first(passport, ("J", "inertia", "rotor_inertia"))

    p_wf = _to_float(p_w, field="P_n/P_kW")
    f_nf = _to_float(f_n, field="f_n", default=50.0)
    poles_i = _to_int(poles, field="p", default=2)
    n_sync = 60.0 * f_nf / max(poles_i, 1)
    n_rated_f = _to_float(n_rated, field="n_rated", default=n_sync * 0.97)

    nameplate = {
        "P_n": p_wf,
        "U_ll": _to_float(u_ll, field="U_ll"),
        "I_n": _to_float(i_n, field="I_n"),
        "cos_phi_n": _to_float(cos_phi, field="cos_phi_n", default=0.8),
        "eta_n": _to_float(eta, field="eta_n", default=0.85),
        "f_n": f_nf,
        "p": poles_i,
        "n_rated": n_rated_f,
        "connection": str(connection or "Y").strip().upper(),
        "J": _to_float(inertia, field="J", default=0.02),
    }
    if nameplate["connection"] not in {"Y", "D"}:
        raise ValueError("connection must be 'Y' or 'D'")
    return nameplate


def _load_ident_estimated(
    *,
    ident_json: str,
    ident_rs_leq: str,
    ident_locked_rotor_q: str,
    ident_mech_runup: str,
    motor_key: str,
    out_dir: Path,
) -> Tuple[Optional[MotorParamsEstimated], str]:
    ident_json = str(ident_json).strip()
    rs_path = str(ident_rs_leq).strip()
    lock_path = str(ident_locked_rotor_q).strip()
    mech_path = str(ident_mech_runup).strip()

    if ident_json:
        payload = _read_json(Path(ident_json).expanduser().resolve())
        raw = payload.get("estimated_params")
        if not isinstance(raw, dict):
            raise ValueError(f"ident_json has no 'estimated_params': {ident_json}")
        return MotorParamsEstimated(**raw), "ident_json"

    provided = [bool(rs_path), bool(lock_path), bool(mech_path)]
    if any(provided) and not all(provided):
        raise ValueError(
            "Provide all three identification datasets: --ident-rs-leq, --ident-locked-rotor-q, --ident-mech-runup"
        )
    if not all(provided):
        return None, ""

    data_rs = load_test_data(rs_path)
    data_lock = load_test_data(lock_path)
    data_mech = load_test_data(mech_path)
    result = run_full_identification(
        env=None,
        motor_name=str(motor_key),
        source="hardware",
        enable_refine=False,
        data_rs_leq=data_rs,
        data_locked_rotor_q=data_lock,
        data_mech_runup=data_mech,
    )
    ident_out = out_dir / "identification_result.json"
    save_ident_result(result, str(ident_out))
    return result.estimated, str(ident_out)


def _apply_ident_to_motor(
    *,
    base: Dict[str, float],
    estimated: Optional[MotorParamsEstimated],
) -> Dict[str, float]:
    motor = dict(base)
    if estimated is None:
        return motor

    if estimated.Rs is not None:
        motor["Rs"] = max(float(estimated.Rs), 1e-7)
    if estimated.Rr is not None:
        motor["Rr"] = max(float(estimated.Rr), 1e-7)
    if estimated.Lm is not None:
        motor["Lm"] = max(float(estimated.Lm), 1e-7)
    if estimated.J is not None:
        motor["J"] = max(float(estimated.J), 1e-7)
    if estimated.B is not None:
        motor["B"] = max(float(estimated.B), 1e-8)

    if estimated.Ls is not None:
        motor["Ls_sigma"] = max(float(estimated.Ls) - float(motor["Lm"]), 1e-6)
    if estimated.Lr is not None:
        motor["Lr_sigma"] = max(float(estimated.Lr) - float(motor["Lm"]), 1e-6)
    return motor


def _render_generated_config(
    *,
    motor_key: str,
    nameplate: Dict[str, object],
    motor: Dict[str, float],
    sim_dt: float,
    sim_t_end: float,
    sim_load_torque: float,
    inverter_vdc: float,
    inverter_r_out: float,
    inverter_dead_time: float,
    inverter_v_drop: float,
    id_ref: float,
    ai_id_ref_alpha: float,
    ai_delta_id_max: float,
    iq_limit: float,
    save_prefix: str,
) -> str:
    return f"""from __future__ import annotations

import math
from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate

_base = create_default_env()

NAMEPLATE_ONBOARD = {{
    "P_n": {float(nameplate["P_n"]):.10g},
    "U_ll": {float(nameplate["U_ll"]):.10g},
    "I_n": {float(nameplate["I_n"]):.10g},
    "cos_phi_n": {float(nameplate["cos_phi_n"]):.10g},
    "eta_n": {float(nameplate["eta_n"]):.10g},
    "f_n": {float(nameplate["f_n"]):.10g},
    "p": {int(nameplate["p"])},
    "n_rated": {float(nameplate["n_rated"]):.10g},
    "connection": "{str(nameplate["connection"])}",
    "J": {float(nameplate["J"]):.10g},
}}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_ONBOARD)
_motor = replace(
    _motor_est,
    Rs={float(motor["Rs"]):.10g},
    Rr={float(motor["Rr"]):.10g},
    Lm={float(motor["Lm"]):.10g},
    Ls_sigma={float(motor["Ls_sigma"]):.10g},
    Lr_sigma={float(motor["Lr_sigma"]):.10g},
    J={float(motor["J"]):.10g},
    B={float(motor["B"]):.10g},
    p={int(motor["p"])},
    I_n={float(motor["I_n"]):.10g},
)

_sim = replace(
    _base.sim,
    t_end={float(sim_t_end):.10g},
    dt={float(sim_dt):.10g},
    save_prefix="{save_prefix}",
    scenario_name="speed_step",
    load_torque={float(sim_load_torque):.10g},
)

_foc = replace(
    _base.foc,
    id_ref={float(id_ref):.10g},
    iq_limit={float(iq_limit):.10g},
)

_inverter = replace(
    _base.inverter,
    Vdc={float(inverter_vdc):.10g},
    r_out={float(inverter_r_out):.10g},
    dead_time={float(inverter_dead_time):.10g},
    v_drop={float(inverter_v_drop):.10g},
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

ai_omega_ref_pu_range = (0.3, 1.1)
ai_load_mult_range = (0.5, 1.6)
ai_drift_every_episodes = 1
ai_drift_params = ("Rs", "Rr", "Lm", "Ls_sigma", "Lr_sigma", "J", "B")
ai_drift_ranges = {{
    "Rs": (0.75, 1.25),
    "Rr": (0.75, 1.25),
    "Lm": (0.85, 1.15),
    "Ls_sigma": (0.75, 1.25),
    "Lr_sigma": (0.75, 1.25),
    "J": (0.6, 1.4),
    "B": (0.6, 1.8),
}}

# Filled by onboarding pipeline after training if desired.
ai_eval_checkpoint_path = ""
ai_eval_id_ref_alpha = {float(ai_id_ref_alpha):.10g}
ai_eval_delta_id_max = {float(ai_delta_id_max):.10g}
ai_eval_id_ref_relative = True
ai_eval_id_ref_gate_speed_tol_rel = 0.08

__all__ = ["ENV"]
"""


def _run_cmd(cmd: List[str], *, cwd: Path, dry_run: bool) -> Dict[str, object]:
    started = time.time()
    row: Dict[str, object] = {
        "cmd": cmd,
        "cwd": str(cwd),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        row["dry_run"] = True
        row["returncode"] = 0
        row["elapsed_sec"] = 0.0
        return row
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    row["dry_run"] = False
    row["returncode"] = int(proc.returncode)
    row["elapsed_sec"] = round(float(time.time() - started), 3)
    return row


def _aggregate_summary_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "power_saving_pct_mean": 0.0,
            "eta_gain_pct_mean": 0.0,
            "err_ok_rate": 0.0,
            "foc_mean_err": 0.0,
            "mic_mean_err": 0.0,
        }
    n = float(len(rows))
    power = [float(r.get("power_saving_pct", 0.0)) for r in rows]
    eta = [float(r.get("eta_gain_pct", 0.0)) for r in rows]
    err_ok = [1.0 if bool(r.get("err_ok", False)) else 0.0 for r in rows]
    foc_err = [float(r.get("foc_mean_err", 0.0)) for r in rows]
    mic_err = [float(r.get("mic_mean_err", 0.0)) for r in rows]
    return {
        "n_rows": n,
        "power_saving_pct_mean": float(sum(power) / n),
        "eta_gain_pct_mean": float(sum(eta) / n),
        "err_ok_rate": float(sum(err_ok) / n),
        "foc_mean_err": float(sum(foc_err) / n),
        "mic_mean_err": float(sum(mic_err) / n),
    }


def _run_benchmark_validation(
    *,
    checkpoint_path: Path,
    benchmark_motors: Sequence[str],
    scenarios: str,
    out_root: Path,
    delta_id_max: float,
    id_ref_alpha: float,
    seed: int,
    dt: float | None,
    t_end: float | None,
    dry_run: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    plan_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for motor in benchmark_motors:
        key = str(motor).strip().lower()
        if key not in BENCHMARK_MOTOR_CONFIGS:
            raise ValueError(f"Unknown benchmark motor: {motor}")
        out_dir = out_root / key
        cmd = [
            sys.executable,
            "-m",
            "mic_ai.tools.scenario_compare",
            "--env-config",
            BENCHMARK_MOTOR_CONFIGS[key],
            "--ai-checkpoint",
            str(checkpoint_path),
            "--ai-id-relative",
            "--delta-id-max",
            str(float(delta_id_max)),
            "--id-ref-alpha",
            str(float(id_ref_alpha)),
            "--scenarios",
            str(scenarios),
            "--window-frac",
            "0.25",
            "--error-tol-rel",
            "0.05",
            "--error-tol-abs",
            "0.0",
            "--seed",
            str(int(seed)),
            "--out-dir",
            str(out_dir),
        ]
        if dt is not None:
            cmd += ["--dt", str(float(dt))]
        if t_end is not None:
            cmd += ["--t-end", str(float(t_end))]
        row = _run_cmd(cmd, cwd=ROOT, dry_run=dry_run)
        row["motor"] = key
        plan_rows.append(row)

        if dry_run:
            continue
        if int(row.get("returncode", 1)) != 0:
            continue
        summary_json = out_dir / "summary.json"
        if not summary_json.exists():
            continue
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        typed_rows = [dict(x) for x in payload if isinstance(x, dict)]
        agg = _aggregate_summary_rows(typed_rows)
        summary_rows.append(
            {
                "motor": key,
                "power_saving_pct_mean": agg["power_saving_pct_mean"],
                "eta_gain_pct_mean": agg["eta_gain_pct_mean"],
                "err_ok_rate": agg["err_ok_rate"],
                "foc_mean_err": agg["foc_mean_err"],
                "mic_mean_err": agg["mic_mean_err"],
                "scenarios_count": int(agg["n_rows"]),
            }
        )
    return plan_rows, summary_rows


def _render_report(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Any-Motor Onboarding Report")
    lines.append("")
    lines.append(f"- created_utc: `{payload.get('created_utc', '')}`")
    lines.append(f"- dry_run: `{bool(payload.get('dry_run', False))}`")
    lines.append(f"- motor_key: `{payload.get('motor_key', '')}`")
    lines.append(f"- run_root: `{payload.get('run_root', '')}`")
    lines.append(f"- all_ok: `{bool(payload.get('all_ok', False))}`")
    lines.append("")
    lines.append("## Steps")
    for i, step in enumerate(payload.get("steps", []), start=1):
        if not isinstance(step, dict):
            continue
        lines.append(
            f"{i}. `{step.get('name', '')}` status=`{step.get('status', '')}` "
            f"rc=`{step.get('returncode', '')}`"
        )
        note = str(step.get("note", "")).strip()
        if note:
            lines.append(f"   - {note}")
    lines.append("")

    bench = payload.get("benchmark_summary_rows", [])
    if isinstance(bench, list) and bench:
        lines.append("## Benchmark Validation")
        lines.append("| Motor | Power Saving Mean, % | Eta Gain Mean, % | err_ok_rate |")
        lines.append("|---|---:|---:|---:|")
        for row in bench:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {m} | {p:+.3f} | {e:+.3f} | {r:.3f} |".format(
                    m=str(row.get("motor", "")),
                    p=float(row.get("power_saving_pct_mean", 0.0)),
                    e=float(row.get("eta_gain_pct_mean", 0.0)),
                    r=float(row.get("err_ok_rate", 0.0)),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Universal onboarding pipeline for a new motor: "
            "passport -> optional identification -> generated env config -> training -> 3-motor validation."
        )
    )
    parser.add_argument("--passport-json", required=True)
    parser.add_argument("--motor-key", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--out-dir", default="outputs/train_any_motor_pipeline")
    parser.add_argument("--generated-config-dir", default="config/generated")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-benchmark-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--ident-json", default="")
    parser.add_argument("--ident-rs-leq", default="")
    parser.add_argument("--ident-locked-rotor-q", default="")
    parser.add_argument("--ident-mech-runup", default="")

    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--delta-id-max", type=float, default=0.3)
    parser.add_argument("--id-ref-alpha", type=float, default=1.0)

    parser.add_argument("--benchmark-motors", default="air56,al31,ao2")
    parser.add_argument("--benchmark-scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--benchmark-seed", type=int, default=101)
    parser.add_argument("--benchmark-dt", type=float, default=None)
    parser.add_argument("--benchmark-t-end", type=float, default=None)

    parser.add_argument("--sim-dt", type=float, default=1e-3)
    parser.add_argument("--sim-t-end", type=float, default=2.0)
    parser.add_argument("--sim-load-torque", type=float, default=None)
    parser.add_argument("--id-ref-scale", type=float, default=0.35)
    parser.add_argument("--iq-limit-mult", type=float, default=3.0)
    parser.add_argument("--inverter-vdc", type=float, default=540.0)
    parser.add_argument("--inverter-r-out", type=float, default=0.1)
    parser.add_argument("--inverter-dead-time", type=float, default=2e-6)
    parser.add_argument("--inverter-v-drop", type=float, default=1.2)
    args = parser.parse_args()

    passport_path = Path(str(args.passport_json)).expanduser().resolve()
    if not passport_path.exists():
        raise FileNotFoundError(passport_path)
    passport_raw = _read_json(passport_path)
    nameplate = _normalize_nameplate(passport_raw)

    key_raw = str(args.motor_key).strip() or str(passport_raw.get("motor_key", "")).strip() or "custom_motor"
    motor_key = _slug(key_raw)
    run_tag = str(args.run_tag).strip() or f"{_now_tag()}_{motor_key}"
    run_root = Path(str(args.out_dir)).expanduser().resolve() / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    config_dir = Path(str(args.generated_config_dir)).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"env_onboard_{motor_key}.py"

    steps: List[Dict[str, object]] = []
    all_ok = True

    normalized_passport_path = run_root / "normalized_passport.json"
    json_dump(normalized_passport_path, nameplate)

    est_params, ident_source = _load_ident_estimated(
        ident_json=str(args.ident_json),
        ident_rs_leq=str(args.ident_rs_leq),
        ident_locked_rotor_q=str(args.ident_locked_rotor_q),
        ident_mech_runup=str(args.ident_mech_runup),
        motor_key=motor_key,
        out_dir=run_root,
    )

    motor_est = estimate_motor_params_from_nameplate(nameplate)
    base_motor = {
        "Rs": float(motor_est.Rs),
        "Rr": float(motor_est.Rr),
        "Lm": float(motor_est.Lm),
        "Ls_sigma": float(max(motor_est.Ls_sigma, 1e-6)),
        "Lr_sigma": float(max(motor_est.Lr_sigma, 1e-6)),
        "J": float(nameplate["J"]),
        "B": float(max(motor_est.B, 1e-8)),
        "p": int(nameplate["p"]),
        "I_n": float(nameplate["I_n"]),
    }
    motor_final = _apply_ident_to_motor(base=base_motor, estimated=est_params)
    json_dump(run_root / "motor_params_final.json", motor_final)

    n_rated = float(nameplate["n_rated"])
    torque_nom = float(nameplate["P_n"]) / max(2.0 * math.pi * n_rated / 60.0, 1e-6)
    sim_load_torque = float(args.sim_load_torque) if args.sim_load_torque is not None else 0.25 * torque_nom
    id_ref = float(estimate_id_ref_from_nameplate(nameplate, k_m=float(args.id_ref_scale)))
    iq_limit = max(float(nameplate["I_n"]) * float(args.iq_limit_mult), 2.0)

    config_text = _render_generated_config(
        motor_key=motor_key,
        nameplate=nameplate,
        motor=motor_final,
        sim_dt=float(args.sim_dt),
        sim_t_end=float(args.sim_t_end),
        sim_load_torque=float(sim_load_torque),
        inverter_vdc=float(args.inverter_vdc),
        inverter_r_out=float(args.inverter_r_out),
        inverter_dead_time=float(args.inverter_dead_time),
        inverter_v_drop=float(args.inverter_v_drop),
        id_ref=float(id_ref),
        ai_id_ref_alpha=float(args.id_ref_alpha),
        ai_delta_id_max=float(args.delta_id_max),
        iq_limit=float(iq_limit),
        save_prefix=f"onboard_{motor_key}",
    )
    config_path.write_text(config_text, encoding="utf-8")
    steps.append(
        {
            "name": "generate_config",
            "status": "ok",
            "returncode": 0,
            "note": f"config={config_path}; ident_source={ident_source or 'nameplate_only'}",
        }
    )

    train_result: Dict[str, str] = {}
    best_checkpoint = Path()
    if bool(args.skip_training):
        steps.append(
            {
                "name": "train_policy",
                "status": "skipped",
                "returncode": 0,
                "note": "training skipped by flag",
            }
        )
    elif bool(args.dry_run):
        steps.append(
            {
                "name": "train_policy",
                "status": "planned",
                "returncode": 0,
                "note": "dry-run: train_ai_id_ref call skipped",
            }
        )
    else:
        try:
            train_result = train_ai_id_ref(
                env_config=str(config_path),
                episodes=int(args.episodes),
                episode_steps=int(args.episode_steps),
                control_mode="ai_id_ref",
                w_speed=1.0,
                w_power=6.0,
                w_current=None,
                w_smooth=0.05,
                w_mag=0.0,
                w_shaft=2.0,
                w_eta=1.0,
                eta_clip=1.2,
                id_ref_alpha=float(args.id_ref_alpha),
                id_ref_rate_limit=None,
                ai_id_speed_tol=0.5,
                ai_id_speed_tol_rel=0.08,
                id_ref_gate_speed_tol=None,
                id_ref_gate_speed_tol_rel=0.08,
                id_ref_gate_min_scale=0.1,
                id_ref_gate_exponent=1.0,
                fast=bool(args.fast),
                time_budget_min=None,
                override_load_torque=False,
                override_omega_ref=False,
                ai_id_ref_relative=True,
                delta_id_max=float(args.delta_id_max),
                load_torque=None,
                omega_ref_override=None,
                scenarios=parse_csv_list(str(args.scenarios)),
                scenario_sample="random",
                omega_ref_range=None,
                load_torque_range=None,
                seed=int(args.seed),
                sigma_start=0.2,
                sigma_end=0.05,
                sigma_decay_episodes=100,
                power_warmup_episodes=0,
                power_ramp_episodes=50,
                eval_interval=0,
                eval_scenarios=str(args.scenarios),
                eval_dt=None,
                eval_t_end=None,
                eval_window_frac=0.25,
                eval_error_tol_rel=0.05,
                eval_error_tol_abs=0.0,
                eval_use_total_power=True,
                include_energy_obs=True,
                update_every_episodes=1 if bool(args.fast) else 4,
                init_checkpoint=str(args.init_checkpoint).strip() or None,
                output_dir=str((run_root / "ai_id_ref").resolve()),
                results_root=str((run_root / "results_run").resolve()),
            )
            best_checkpoint = Path(str(train_result.get("best", ""))).expanduser().resolve()
            ok = best_checkpoint.exists()
            all_ok = all_ok and bool(ok)
            steps.append(
                {
                    "name": "train_policy",
                    "status": "ok" if ok else "failed",
                    "returncode": 0 if ok else 2,
                    "note": f"best_checkpoint={best_checkpoint}",
                }
            )
        except Exception as exc:
            all_ok = False
            steps.append(
                {
                    "name": "train_policy",
                    "status": "failed",
                    "returncode": 2,
                    "note": str(exc),
                }
            )

    benchmark_plan_rows: List[Dict[str, object]] = []
    benchmark_summary_rows: List[Dict[str, object]] = []
    if bool(args.skip_benchmark_validation):
        steps.append(
            {
                "name": "validate_benchmarks",
                "status": "skipped",
                "returncode": 0,
                "note": "validation skipped by flag",
            }
        )
    else:
        benchmark_motors = [m.strip().lower() for m in parse_csv_list(str(args.benchmark_motors))]
        if bool(args.dry_run):
            benchmark_plan_rows, benchmark_summary_rows = _run_benchmark_validation(
                checkpoint_path=Path("dry_run_checkpoint.pth"),
                benchmark_motors=benchmark_motors,
                scenarios=str(args.benchmark_scenarios),
                out_root=run_root / "benchmark_validation",
                delta_id_max=float(args.delta_id_max),
                id_ref_alpha=float(args.id_ref_alpha),
                seed=int(args.benchmark_seed),
                dt=float(args.benchmark_dt) if args.benchmark_dt is not None else None,
                t_end=float(args.benchmark_t_end) if args.benchmark_t_end is not None else None,
                dry_run=True,
            )
            steps.append(
                {
                    "name": "validate_benchmarks",
                    "status": "planned",
                    "returncode": 0,
                    "note": "dry-run: scenario_compare calls skipped",
                }
            )
        else:
            if bool(args.skip_training):
                raise ValueError("Benchmark validation requires trained checkpoint when --skip-training is used.")
            if not best_checkpoint.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")
            benchmark_plan_rows, benchmark_summary_rows = _run_benchmark_validation(
                checkpoint_path=best_checkpoint,
                benchmark_motors=benchmark_motors,
                scenarios=str(args.benchmark_scenarios),
                out_root=run_root / "benchmark_validation",
                delta_id_max=float(args.delta_id_max),
                id_ref_alpha=float(args.id_ref_alpha),
                seed=int(args.benchmark_seed),
                dt=float(args.benchmark_dt) if args.benchmark_dt is not None else None,
                t_end=float(args.benchmark_t_end) if args.benchmark_t_end is not None else None,
                dry_run=False,
            )
            rc = 0 if all(int(r.get("returncode", 1)) == 0 for r in benchmark_plan_rows) else 3
            all_ok = all_ok and bool(rc == 0)
            steps.append(
                {
                    "name": "validate_benchmarks",
                    "status": "ok" if rc == 0 else "failed",
                    "returncode": rc,
                    "note": f"motors={','.join(benchmark_motors)}",
                }
            )

    if benchmark_plan_rows:
        write_csv(run_root / "benchmark_validation_plan.csv", benchmark_plan_rows)
        json_dump(run_root / "benchmark_validation_plan.json", benchmark_plan_rows)
    if benchmark_summary_rows:
        write_csv(run_root / "benchmark_validation_summary.csv", benchmark_summary_rows)
        json_dump(run_root / "benchmark_validation_summary.json", benchmark_summary_rows)

    payload: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "all_ok": bool(all_ok),
        "motor_key": motor_key,
        "run_root": str(run_root),
        "passport_json": str(passport_path),
        "normalized_passport_json": str(normalized_passport_path),
        "generated_config_path": str(config_path),
        "ident_source": ident_source,
        "training_result": train_result,
        "benchmark_summary_rows": benchmark_summary_rows,
        "steps": steps,
    }
    report_json = run_root / "any_motor_onboarding_report.json"
    report_md = run_root / "any_motor_onboarding_report.md"
    json_dump(report_json, payload)
    report_md.write_text(_render_report(payload), encoding="utf-8")
    print(f"saved: {report_json}")
    print(f"saved: {report_md}")

    if not bool(args.dry_run) and not bool(all_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
