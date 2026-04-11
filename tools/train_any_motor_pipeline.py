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


def _parse_float_csv(text: str) -> List[float]:
    values: List[float] = []
    seen: set[Tuple[int, int]] = set()
    for token in parse_csv_list(str(text)):
        value = float(token)
        key = (int(value * 1_000_000), int(value * 1_000))
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def _fmt_float_tag(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text or "0"


def _build_validation_pairs(
    *,
    base_alpha: float,
    base_delta: float,
    alpha_grid: str,
    delta_grid: str,
) -> List[Tuple[float, float]]:
    alphas = _parse_float_csv(alpha_grid) if str(alpha_grid).strip() else [float(base_alpha)]
    deltas = _parse_float_csv(delta_grid) if str(delta_grid).strip() else [float(base_delta)]

    pairs: List[Tuple[float, float]] = []
    seen: set[Tuple[int, int]] = set()
    base_pair = (float(base_alpha), float(base_delta))
    for alpha, delta in [base_pair, *[(a, d) for a in alphas for d in deltas]]:
        key = (int(alpha * 1_000_000), int(delta * 1_000_000))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((float(alpha), float(delta)))
    return pairs


def _evaluate_benchmark_acceptance(
    *,
    summary_rows: Sequence[Dict[str, object]],
    expected_motors: Sequence[str],
    err_ok_rate_min: float,
    power_saving_mean_min: Optional[float],
    required_pass_count: int,
) -> Dict[str, object]:
    expected = [str(m).strip().lower() for m in expected_motors if str(m).strip()]
    summary_by_motor = {str(r.get("motor", "")).strip().lower(): r for r in summary_rows if isinstance(r, dict)}

    details: List[Dict[str, object]] = []
    pass_count = 0
    mean_err_terms: List[float] = []
    mean_power_terms: List[float] = []
    mean_mic_err_terms: List[float] = []
    missing: List[str] = []

    for motor in expected:
        row = summary_by_motor.get(motor)
        if row is None:
            missing.append(motor)
            details.append(
                {
                    "motor": motor,
                    "missing": True,
                    "pass": False,
                }
            )
            continue

        err_ok_rate = float(row.get("err_ok_rate", 0.0))
        power_mean = float(row.get("power_saving_pct_mean", 0.0))
        scenarios_count = int(row.get("scenarios_count", 0))
        mic_mean_err = float(row.get("mic_mean_err", 0.0))
        pass_err = bool(err_ok_rate >= float(err_ok_rate_min))
        pass_power = True if power_saving_mean_min is None else bool(power_mean >= float(power_saving_mean_min))
        pass_scenarios = bool(scenarios_count > 0)
        passed = bool(pass_err and pass_power and pass_scenarios)
        if passed:
            pass_count += 1
        mean_err_terms.append(err_ok_rate)
        mean_power_terms.append(power_mean)
        mean_mic_err_terms.append(mic_mean_err)
        details.append(
            {
                "motor": motor,
                "missing": False,
                "err_ok_rate": err_ok_rate,
                "power_saving_pct_mean": power_mean,
                "scenarios_count": scenarios_count,
                "mic_mean_err": mic_mean_err,
                "pass_err_ok_rate": pass_err,
                "pass_power_saving": pass_power,
                "pass_scenarios": pass_scenarios,
                "pass": passed,
            }
        )

    required = int(required_pass_count)
    if required <= 0:
        required = len(expected)
    required = min(max(required, 1), max(len(expected), 1))

    mean_err = float(sum(mean_err_terms) / len(mean_err_terms)) if mean_err_terms else 0.0
    mean_power = float(sum(mean_power_terms) / len(mean_power_terms)) if mean_power_terms else 0.0
    mean_mic_err = float(sum(mean_mic_err_terms) / len(mean_mic_err_terms)) if mean_mic_err_terms else 0.0

    return {
        "expected_motors": expected,
        "summary_rows_count": int(len(summary_rows)),
        "missing_motors": missing,
        "required_pass_count": required,
        "pass_count": int(pass_count),
        "all_pass": bool(pass_count >= required and len(missing) == 0),
        "err_ok_rate_min": float(err_ok_rate_min),
        "power_saving_mean_min": None if power_saving_mean_min is None else float(power_saving_mean_min),
        "mean_err_ok_rate": mean_err,
        "mean_power_saving_pct": mean_power,
        "mean_mic_err": mean_mic_err,
        "details": details,
    }


def _acceptance_rank(payload: Dict[str, object]) -> Tuple[int, int, float, float, float]:
    return (
        1 if bool(payload.get("all_pass", False)) else 0,
        int(payload.get("pass_count", 0)),
        float(payload.get("mean_err_ok_rate", 0.0)),
        float(payload.get("mean_power_saving_pct", 0.0)),
        -float(payload.get("mean_mic_err", 0.0)),
    )


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
    selected = payload.get("selected_validation", {})
    if isinstance(selected, dict) and selected:
        lines.append(
            "- selected_validation: `attempt={a} alpha={alpha:.4f} delta={delta:.4f}`".format(
                a=int(selected.get("train_attempt", 0)),
                alpha=float(selected.get("id_ref_alpha", 0.0)),
                delta=float(selected.get("delta_id_max", 0.0)),
            )
        )
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
    acc = payload.get("benchmark_acceptance", {})
    if isinstance(acc, dict) and acc:
        lines.append("## Acceptance Gate")
        lines.append(
            "- pass_count/required: `{}/{}; all_pass={}`".format(
                int(acc.get("pass_count", 0)),
                int(acc.get("required_pass_count", 0)),
                bool(acc.get("all_pass", False)),
            )
        )
        lines.append(
            "- thresholds: `err_ok_rate_min={:.3f}, power_saving_mean_min={}`".format(
                float(acc.get("err_ok_rate_min", 0.0)),
                (
                    "disabled"
                    if acc.get("power_saving_mean_min", None) is None
                    else "{:+.3f}%".format(float(acc.get("power_saving_mean_min", 0.0)))
                ),
            )
        )
        missing = acc.get("missing_motors", [])
        if isinstance(missing, list) and missing:
            lines.append(f"- missing_motors: `{','.join(str(x) for x in missing)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Universal onboarding pipeline for a new motor: "
            "passport -> optional identification -> generated env config -> training -> benchmark validation."
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
    parser.add_argument("--max-train-attempts", type=int, default=1)
    parser.add_argument("--train-episodes-scale", type=float, default=1.5)

    parser.add_argument(
        "--benchmark-motors",
        default="air56,al31",
        help=(
            "Comma-separated benchmark motors for onboarding validation. "
            "Default follows the active 2-motor release scope; add ao2 explicitly to re-open backlog validation."
        ),
    )
    parser.add_argument("--benchmark-scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--benchmark-seed", type=int, default=101)
    parser.add_argument("--benchmark-dt", type=float, default=None)
    parser.add_argument("--benchmark-t-end", type=float, default=None)
    parser.add_argument("--benchmark-search-alpha-grid", default="")
    parser.add_argument("--benchmark-search-delta-grid", default="")
    parser.add_argument("--accept-err-ok-rate-min", type=float, default=1.0)
    parser.add_argument("--accept-power-saving-mean-min", type=float, default=None)
    parser.add_argument("--accept-required-motor-pass-count", type=int, default=0)
    parser.add_argument("--no-acceptance-gate", action="store_true")

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
    train_attempt_rows: List[Dict[str, object]] = []

    benchmark_plan_rows: List[Dict[str, object]] = []
    benchmark_summary_rows: List[Dict[str, object]] = []
    benchmark_search_rows: List[Dict[str, object]] = []
    benchmark_acceptance: Dict[str, object] = {}
    selected_validation: Dict[str, object] = {}
    selected_candidate: Optional[Dict[str, object]] = None

    benchmark_motors: List[str] = []
    for raw_motor in parse_csv_list(str(args.benchmark_motors)):
        key = str(raw_motor).strip().lower()
        if not key:
            continue
        if key not in benchmark_motors:
            benchmark_motors.append(key)
    if not benchmark_motors and not bool(args.skip_benchmark_validation):
        raise ValueError("Empty benchmark motors list")

    validation_pairs = _build_validation_pairs(
        base_alpha=float(args.id_ref_alpha),
        base_delta=float(args.delta_id_max),
        alpha_grid=str(args.benchmark_search_alpha_grid),
        delta_grid=str(args.benchmark_search_delta_grid),
    )
    max_train_attempts = max(1, int(args.max_train_attempts))
    episodes_base = max(1, int(args.episodes))
    episodes_scale = max(1.0, float(args.train_episodes_scale))
    external_checkpoint = Path(str(args.init_checkpoint).strip()).expanduser().resolve() if str(args.init_checkpoint).strip() else Path()

    if bool(args.skip_training):
        if external_checkpoint.exists():
            best_checkpoint = external_checkpoint
            train_result = {
                "best": str(best_checkpoint),
                "source": "external_init_checkpoint",
            }
        steps.append(
            {
                "name": "train_policy",
                "status": "skipped",
                "returncode": 0,
                "note": (
                    f"training skipped by flag; external_checkpoint={best_checkpoint}"
                    if best_checkpoint.exists()
                    else "training skipped by flag"
                ),
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
        init_checkpoint = str(args.init_checkpoint).strip() or None
        for attempt_idx in range(1, max_train_attempts + 1):
            episodes_now = max(1, int(round(episodes_base * (episodes_scale ** (attempt_idx - 1)))))
            attempt_output = (run_root / "ai_id_ref" / f"attempt_{attempt_idx:02d}").resolve()
            attempt_results = (run_root / "results_run" / f"attempt_{attempt_idx:02d}").resolve()

            try:
                result = train_ai_id_ref(
                    env_config=str(config_path),
                    episodes=episodes_now,
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
                    init_checkpoint=init_checkpoint,
                    output_dir=str(attempt_output),
                    results_root=str(attempt_results),
                )
                ckpt = Path(str(result.get("best", ""))).expanduser().resolve()
                ok = ckpt.exists()
                row = {
                    "train_attempt": attempt_idx,
                    "episodes": int(episodes_now),
                    "episode_steps": int(args.episode_steps),
                    "init_checkpoint": init_checkpoint or "",
                    "best_checkpoint": str(ckpt),
                    "ok": bool(ok),
                }
                train_attempt_rows.append(row)
                if not ok:
                    continue
                train_result = result
                best_checkpoint = ckpt
                init_checkpoint = str(ckpt)

                if bool(args.skip_benchmark_validation):
                    break

                attempt_best: Optional[Dict[str, object]] = None
                for alpha, delta in validation_pairs:
                    pair_tag = f"alpha_{_fmt_float_tag(alpha)}_delta_{_fmt_float_tag(delta)}"
                    pair_out = run_root / "benchmark_validation" / f"train_attempt_{attempt_idx:02d}" / pair_tag
                    pair_plan, pair_summary = _run_benchmark_validation(
                        checkpoint_path=ckpt,
                        benchmark_motors=benchmark_motors,
                        scenarios=str(args.benchmark_scenarios),
                        out_root=pair_out,
                        delta_id_max=float(delta),
                        id_ref_alpha=float(alpha),
                        seed=int(args.benchmark_seed),
                        dt=float(args.benchmark_dt) if args.benchmark_dt is not None else None,
                        t_end=float(args.benchmark_t_end) if args.benchmark_t_end is not None else None,
                        dry_run=False,
                    )
                    for r in pair_plan:
                        r["train_attempt"] = int(attempt_idx)
                        r["id_ref_alpha"] = float(alpha)
                        r["delta_id_max"] = float(delta)
                        r["checkpoint"] = str(ckpt)
                        benchmark_plan_rows.append(r)
                    for s in pair_summary:
                        s["train_attempt"] = int(attempt_idx)
                        s["id_ref_alpha"] = float(alpha)
                        s["delta_id_max"] = float(delta)

                    pair_rc_ok = all(int(r.get("returncode", 1)) == 0 for r in pair_plan)
                    pair_acceptance = _evaluate_benchmark_acceptance(
                        summary_rows=pair_summary,
                        expected_motors=benchmark_motors,
                        err_ok_rate_min=float(args.accept_err_ok_rate_min),
                        power_saving_mean_min=(
                            None
                            if args.accept_power_saving_mean_min is None
                            else float(args.accept_power_saving_mean_min)
                        ),
                        required_pass_count=int(args.accept_required_motor_pass_count),
                    )
                    pair_acceptance["train_attempt"] = int(attempt_idx)
                    pair_acceptance["id_ref_alpha"] = float(alpha)
                    pair_acceptance["delta_id_max"] = float(delta)
                    pair_acceptance["plan_rc_ok"] = bool(pair_rc_ok)
                    pair_acceptance["gate_pass"] = bool(pair_rc_ok and bool(pair_acceptance.get("all_pass", False)))

                    benchmark_search_rows.append(
                        {
                            "train_attempt": int(attempt_idx),
                            "id_ref_alpha": float(alpha),
                            "delta_id_max": float(delta),
                            "plan_rc_ok": bool(pair_rc_ok),
                            "all_pass": bool(pair_acceptance.get("all_pass", False)),
                            "pass_count": int(pair_acceptance.get("pass_count", 0)),
                            "required_pass_count": int(pair_acceptance.get("required_pass_count", 0)),
                            "mean_err_ok_rate": float(pair_acceptance.get("mean_err_ok_rate", 0.0)),
                            "mean_power_saving_pct": float(pair_acceptance.get("mean_power_saving_pct", 0.0)),
                            "mean_mic_err": float(pair_acceptance.get("mean_mic_err", 0.0)),
                            "missing_motors": ",".join(str(x) for x in pair_acceptance.get("missing_motors", [])),
                            "summary_rows_count": int(pair_acceptance.get("summary_rows_count", 0)),
                        }
                    )

                    candidate = {
                        "train_attempt": int(attempt_idx),
                        "checkpoint": str(ckpt),
                        "id_ref_alpha": float(alpha),
                        "delta_id_max": float(delta),
                        "summary_rows": pair_summary,
                        "acceptance": pair_acceptance,
                        "plan_rc_ok": bool(pair_rc_ok),
                    }
                    if attempt_best is None:
                        attempt_best = candidate
                    else:
                        left_rank = (
                            1 if bool(candidate.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(candidate.get("acceptance", {}))),
                        )
                        right_rank = (
                            1 if bool(attempt_best.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(attempt_best.get("acceptance", {}))),
                        )
                        if left_rank > right_rank:
                            attempt_best = candidate

                if attempt_best is not None:
                    if selected_candidate is None:
                        selected_candidate = attempt_best
                    else:
                        left_rank = (
                            1 if bool(attempt_best.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(attempt_best.get("acceptance", {}))),
                        )
                        right_rank = (
                            1 if bool(selected_candidate.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(selected_candidate.get("acceptance", {}))),
                        )
                        if left_rank > right_rank:
                            selected_candidate = attempt_best

                    if bool(attempt_best.get("plan_rc_ok", False)) and bool(
                        dict(attempt_best.get("acceptance", {})).get("all_pass", False)
                    ):
                        break
            except Exception as exc:
                train_attempt_rows.append(
                    {
                        "train_attempt": attempt_idx,
                        "episodes": int(episodes_now),
                        "episode_steps": int(args.episode_steps),
                        "init_checkpoint": init_checkpoint or "",
                        "best_checkpoint": "",
                        "ok": False,
                        "error": str(exc),
                    }
                )

        train_ok = best_checkpoint.exists()
        all_ok = all_ok and bool(train_ok)
        steps.append(
            {
                "name": "train_policy",
                "status": "ok" if train_ok else "failed",
                "returncode": 0 if train_ok else 2,
                "note": f"attempts={len(train_attempt_rows)}; best_checkpoint={best_checkpoint}",
            }
        )

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
        if bool(args.dry_run):
            base_alpha, base_delta = validation_pairs[0]
            benchmark_plan_rows, _ = _run_benchmark_validation(
                checkpoint_path=Path("dry_run_checkpoint.pth"),
                benchmark_motors=benchmark_motors,
                scenarios=str(args.benchmark_scenarios),
                out_root=run_root / "benchmark_validation" / "dry_run",
                delta_id_max=float(base_delta),
                id_ref_alpha=float(base_alpha),
                seed=int(args.benchmark_seed),
                dt=float(args.benchmark_dt) if args.benchmark_dt is not None else None,
                t_end=float(args.benchmark_t_end) if args.benchmark_t_end is not None else None,
                dry_run=True,
            )
            selected_validation = {
                "train_attempt": 0,
                "checkpoint": "dry_run_checkpoint.pth",
                "id_ref_alpha": float(base_alpha),
                "delta_id_max": float(base_delta),
            }
            steps.append(
                {
                    "name": "validate_benchmarks",
                    "status": "planned",
                    "returncode": 0,
                    "note": "dry-run: scenario_compare calls skipped",
                }
            )
        else:
            if selected_candidate is None:
                if not best_checkpoint.exists():
                    raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")
                fallback_best: Optional[Dict[str, object]] = None
                fallback_attempt = int(max_train_attempts) if not bool(args.skip_training) else 0
                for alpha, delta in validation_pairs:
                    pair_tag = f"alpha_{_fmt_float_tag(alpha)}_delta_{_fmt_float_tag(delta)}"
                    pair_plan, pair_summary = _run_benchmark_validation(
                        checkpoint_path=best_checkpoint,
                        benchmark_motors=benchmark_motors,
                        scenarios=str(args.benchmark_scenarios),
                        out_root=run_root / "benchmark_validation" / "fallback" / pair_tag,
                        delta_id_max=float(delta),
                        id_ref_alpha=float(alpha),
                        seed=int(args.benchmark_seed),
                        dt=float(args.benchmark_dt) if args.benchmark_dt is not None else None,
                        t_end=float(args.benchmark_t_end) if args.benchmark_t_end is not None else None,
                        dry_run=False,
                    )
                    for r in pair_plan:
                        r["train_attempt"] = int(fallback_attempt)
                        r["id_ref_alpha"] = float(alpha)
                        r["delta_id_max"] = float(delta)
                        r["checkpoint"] = str(best_checkpoint)
                        benchmark_plan_rows.append(r)
                    for s in pair_summary:
                        s["train_attempt"] = int(fallback_attempt)
                        s["id_ref_alpha"] = float(alpha)
                        s["delta_id_max"] = float(delta)

                    pair_rc_ok = all(int(r.get("returncode", 1)) == 0 for r in pair_plan)
                    pair_acceptance = _evaluate_benchmark_acceptance(
                        summary_rows=pair_summary,
                        expected_motors=benchmark_motors,
                        err_ok_rate_min=float(args.accept_err_ok_rate_min),
                        power_saving_mean_min=(
                            None
                            if args.accept_power_saving_mean_min is None
                            else float(args.accept_power_saving_mean_min)
                        ),
                        required_pass_count=int(args.accept_required_motor_pass_count),
                    )
                    pair_acceptance["train_attempt"] = int(fallback_attempt)
                    pair_acceptance["id_ref_alpha"] = float(alpha)
                    pair_acceptance["delta_id_max"] = float(delta)
                    pair_acceptance["plan_rc_ok"] = bool(pair_rc_ok)
                    pair_acceptance["gate_pass"] = bool(pair_rc_ok and bool(pair_acceptance.get("all_pass", False)))

                    benchmark_search_rows.append(
                        {
                            "train_attempt": int(fallback_attempt),
                            "id_ref_alpha": float(alpha),
                            "delta_id_max": float(delta),
                            "plan_rc_ok": bool(pair_rc_ok),
                            "all_pass": bool(pair_acceptance.get("all_pass", False)),
                            "pass_count": int(pair_acceptance.get("pass_count", 0)),
                            "required_pass_count": int(pair_acceptance.get("required_pass_count", 0)),
                            "mean_err_ok_rate": float(pair_acceptance.get("mean_err_ok_rate", 0.0)),
                            "mean_power_saving_pct": float(pair_acceptance.get("mean_power_saving_pct", 0.0)),
                            "mean_mic_err": float(pair_acceptance.get("mean_mic_err", 0.0)),
                            "missing_motors": ",".join(str(x) for x in pair_acceptance.get("missing_motors", [])),
                            "summary_rows_count": int(pair_acceptance.get("summary_rows_count", 0)),
                        }
                    )

                    candidate = {
                        "train_attempt": int(fallback_attempt),
                        "checkpoint": str(best_checkpoint),
                        "id_ref_alpha": float(alpha),
                        "delta_id_max": float(delta),
                        "summary_rows": pair_summary,
                        "acceptance": pair_acceptance,
                        "plan_rc_ok": bool(pair_rc_ok),
                    }
                    if fallback_best is None:
                        fallback_best = candidate
                    else:
                        left_rank = (
                            1 if bool(candidate.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(candidate.get("acceptance", {}))),
                        )
                        right_rank = (
                            1 if bool(fallback_best.get("plan_rc_ok", False)) else 0,
                            *_acceptance_rank(dict(fallback_best.get("acceptance", {}))),
                        )
                        if left_rank > right_rank:
                            fallback_best = candidate
                selected_candidate = fallback_best

            benchmark_summary_rows = list(selected_candidate.get("summary_rows", []))  # type: ignore[arg-type]
            benchmark_acceptance = dict(selected_candidate.get("acceptance", {}))  # type: ignore[arg-type]
            selected_validation = {
                "train_attempt": int(selected_candidate.get("train_attempt", 0)),
                "checkpoint": str(selected_candidate.get("checkpoint", "")),
                "id_ref_alpha": float(selected_candidate.get("id_ref_alpha", 0.0)),
                "delta_id_max": float(selected_candidate.get("delta_id_max", 0.0)),
            }
            rc_commands = 0 if bool(selected_candidate.get("plan_rc_ok", False)) else 3
            gate_required = not bool(args.no_acceptance_gate)
            gate_pass = bool(benchmark_acceptance.get("all_pass", False)) if gate_required else True
            rc_gate = 0 if gate_pass else 4
            rc = rc_commands if rc_commands != 0 else rc_gate
            all_ok = all_ok and bool(rc == 0)
            steps.append(
                {
                    "name": "validate_benchmarks",
                    "status": "ok" if rc == 0 else "failed",
                    "returncode": rc,
                    "note": (
                        "motors={motors}; selected_attempt={attempt}; alpha={alpha:.4f}; delta={delta:.4f}; "
                        "gate_required={gate_required}; gate_pass={gate_pass}"
                    ).format(
                        motors=",".join(benchmark_motors),
                        attempt=int(selected_validation.get("train_attempt", 0)),
                        alpha=float(selected_validation.get("id_ref_alpha", 0.0)),
                        delta=float(selected_validation.get("delta_id_max", 0.0)),
                        gate_required=gate_required,
                        gate_pass=gate_pass,
                    ),
                }
            )

    if train_attempt_rows:
        write_csv(run_root / "training_attempts.csv", train_attempt_rows)
        json_dump(run_root / "training_attempts.json", train_attempt_rows)
    if benchmark_plan_rows:
        write_csv(run_root / "benchmark_validation_plan.csv", benchmark_plan_rows)
        json_dump(run_root / "benchmark_validation_plan.json", benchmark_plan_rows)
    if benchmark_search_rows:
        write_csv(run_root / "benchmark_search_summary.csv", benchmark_search_rows)
        json_dump(run_root / "benchmark_search_summary.json", benchmark_search_rows)
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
        "training_attempts": train_attempt_rows,
        "benchmark_search_pairs": [
            {"id_ref_alpha": float(alpha), "delta_id_max": float(delta)} for alpha, delta in validation_pairs
        ],
        "selected_validation": selected_validation,
        "benchmark_summary_rows": benchmark_summary_rows,
        "benchmark_acceptance": benchmark_acceptance,
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
