from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import estimate_id_ref_from_nameplate, estimate_motor_params_from_nameplate
from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from simulation.gym_env import InductionMotorEnv


def _resolve_config_module(config: str):
    cfg = str(config).strip()
    if cfg.endswith(".py") or "\\" in cfg or "/" in cfg:
        path = Path(cfg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"config path does not exist: {path}")
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load config module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if cfg.startswith("config."):
        return importlib.import_module(cfg)
    return importlib.import_module(f"config.{cfg}")


def _find_nameplate(module) -> Dict[str, float]:
    for name in dir(module):
        if not str(name).startswith("NAMEPLATE_"):
            continue
        value = getattr(module, name)
        if isinstance(value, dict) and "P_n" in value:
            return dict(value)
    raise ValueError(f"NAMEPLATE_* dict not found in module {module.__name__}")


def _as_plain_dict(obj: object) -> Dict[str, object]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return dict(asdict(obj))
    if isinstance(obj, dict):
        return dict(obj)
    out: Dict[str, object] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, (int, float, str, bool, type(None))):
            out[key] = value
    return out


def _copy_env_with_extras(env_cfg: object, **kwargs):
    fields = set(getattr(env_cfg, "__dataclass_fields__", {}).keys())
    extras: Dict[str, object] = {}
    try:
        for name, value in vars(env_cfg).items():
            if name not in fields:
                extras[name] = value
    except Exception:
        extras = {}
    out = replace(env_cfg, **kwargs)
    for name, value in extras.items():
        if hasattr(out, name):
            continue
        try:
            object.__setattr__(out, name, value)
        except Exception:
            try:
                setattr(out, name, value)
            except Exception:
                pass
    return out


def nominal_omega_rad_s(nameplate: Dict[str, float]) -> float:
    return float(2.0 * math.pi * float(nameplate["n_rated"]) / 60.0)


def nominal_torque_nm(nameplate: Dict[str, float]) -> float:
    omega = nominal_omega_rad_s(nameplate)
    return float(float(nameplate["P_n"]) / max(omega, 1e-9))


def estimate_foc_torque_capacity_nm(env_cfg: object) -> float:
    motor = getattr(env_cfg, "motor")
    foc = getattr(env_cfg, "foc")
    p = float(getattr(motor, "p"))
    lm = float(getattr(motor, "Lm"))
    lr = float(getattr(motor, "Lr_sigma") + getattr(motor, "Lm"))
    id_ref = float(getattr(foc, "id_ref", 0.0) or 0.0)
    iq_limit = float(getattr(foc, "iq_limit", 0.0) or 0.0)
    if lr <= 1e-9:
        return 0.0
    k_t = 1.5 * p * (lm * lm / lr)
    return float(k_t * id_ref * iq_limit)


def _relative_delta(current: float, reference: float) -> float | None:
    ref = float(reference)
    cur = float(current)
    if abs(ref) <= 1e-12:
        return None
    return float((cur - ref) / ref)


def _ratio(current: float, reference: float) -> float | None:
    ref = float(reference)
    cur = float(current)
    if abs(ref) <= 1e-12:
        return None
    return float(cur / ref)


def build_param_delta_report(current_motor: object, estimated_motor: object) -> Dict[str, Dict[str, float | None]]:
    keys = ("Rs", "Rr", "Lm", "Ls_sigma", "Lr_sigma", "J", "B")
    report: Dict[str, Dict[str, float | None]] = {}
    for key in keys:
        cur = float(getattr(current_motor, key))
        est = float(getattr(estimated_motor, key))
        report[key] = {
            "current": cur,
            "estimated_from_nameplate": est,
            "ratio_to_estimated": _ratio(cur, est),
            "relative_delta_to_estimated": _relative_delta(cur, est),
        }
    return report


def _simulate_probe(
    env_cfg: object,
    scenario_name: str,
    load_torque: float,
    t_end: float,
    dt: float,
    steady_window_frac: float,
    speed_target_rad_s: float,
) -> Dict[str, float | str]:
    sim = replace(
        env_cfg.sim,
        scenario_name=str(scenario_name),
        load_torque=float(load_torque),
        t_end=float(t_end),
        dt=float(dt),
        save_prefix="nominal_diagnosis",
    )
    env_cfg_run = _copy_env_with_extras(env_cfg, sim=sim)
    env = InductionMotorEnv(env_cfg_run)
    obs = env.reset()
    steps = int(max(float(t_end) / float(dt), 1))
    omega_abs_limit = max(abs(float(speed_target_rad_s)) * 5.0, 1e3)
    torque_abs_limit = max(abs(float(load_torque)) * 20.0, 1e3)

    def _sanitize_scalar(value: float, *, abs_limit: float) -> float:
        nonlocal invalid_signal_seen
        if not math.isfinite(value) or abs(float(value)) > float(abs_limit):
            invalid_signal_seen = True
            return 0.0
        return float(value)

    omega_vals = np.zeros(steps, dtype=float)
    omega_ref_vals = np.zeros(steps, dtype=float)
    i_rms_vals = np.zeros(steps, dtype=float)
    p_el_vals = np.zeros(steps, dtype=float)
    p_mech_vals = np.zeros(steps, dtype=float)
    v_peak_vals = np.zeros(steps, dtype=float)
    torque_vals = np.zeros(steps, dtype=float)
    invalid_signal_seen = False

    for idx in range(steps):
        obs, _reward, done, info = env.step(None)
        omega = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        omega_ref = float(env.omega_ref_func(env.t))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))

        invalid_signal_seen = invalid_signal_seen or not bool(
            np.isfinite(omega) and np.isfinite(omega_ref) and np.all(np.isfinite(i_abc)) and np.all(np.isfinite(v_abc)) and np.isfinite(torque)
        )

        omega_safe = _sanitize_scalar(float(np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)), abs_limit=omega_abs_limit)
        omega_ref_safe = _sanitize_scalar(
            float(np.nan_to_num(omega_ref, nan=0.0, posinf=0.0, neginf=0.0)),
            abs_limit=omega_abs_limit,
        )
        torque_safe = _sanitize_scalar(float(np.nan_to_num(torque, nan=0.0, posinf=0.0, neginf=0.0)), abs_limit=torque_abs_limit)

        omega_vals[idx] = omega_safe
        omega_ref_vals[idx] = omega_ref_safe
        i_rms_vals[idx] = float(np.nan_to_num(calc_i_rms(i_abc), nan=0.0, posinf=0.0, neginf=0.0))
        p_el_vals[idx] = float(np.nan_to_num(calc_p_el(v_abc, i_abc), nan=0.0, posinf=0.0, neginf=0.0))
        p_mech_vals[idx] = float(np.nan_to_num(calc_p_mech(omega_safe, torque_safe), nan=0.0, posinf=0.0, neginf=0.0))
        v_peak_vals[idx] = float(np.nan_to_num(np.max(np.abs(v_abc)) if v_abc.size else 0.0, nan=0.0, posinf=0.0, neginf=0.0))
        torque_vals[idx] = torque_safe
        if done:
            omega_vals = omega_vals[: idx + 1]
            omega_ref_vals = omega_ref_vals[: idx + 1]
            i_rms_vals = i_rms_vals[: idx + 1]
            p_el_vals = p_el_vals[: idx + 1]
            p_mech_vals = p_mech_vals[: idx + 1]
            v_peak_vals = v_peak_vals[: idx + 1]
            torque_vals = torque_vals[: idx + 1]
            break

    start = int(max(0, len(omega_vals) * (1.0 - float(steady_window_frac))))
    omega_ss = float(np.mean(omega_vals[start:])) if omega_vals.size else 0.0
    omega_ref_ss = float(np.mean(omega_ref_vals[start:])) if omega_ref_vals.size else 0.0
    p_el_ss = float(np.mean(p_el_vals[start:])) if p_el_vals.size else 0.0
    p_mech_ss = float(np.mean(p_mech_vals[start:])) if p_mech_vals.size else 0.0
    i_rms_ss = float(np.mean(i_rms_vals[start:])) if i_rms_vals.size else 0.0
    eta_ss = float(p_mech_ss / p_el_ss) if p_el_ss > 1e-9 else 0.0
    speed_err = np.abs(omega_ref_vals - omega_vals)
    denom = max(float(speed_target_rad_s), 1e-9)
    v_limit = float(getattr(env_cfg_run.foc, "v_limit", 0.0) or 0.0)

    return {
        "scenario": str(scenario_name),
        "load_torque_nm": float(load_torque),
        "steps": int(omega_vals.size),
        "steady_omega_rpm": float(omega_ss * 60.0 / (2.0 * math.pi)),
        "steady_omega_ref_rpm": float(omega_ref_ss * 60.0 / (2.0 * math.pi)),
        "steady_speed_error_rel": float(abs(omega_ref_ss - omega_ss) / denom),
        "mean_speed_error_rel": float(np.mean(speed_err) / denom) if speed_err.size else 0.0,
        "peak_speed_error_rel": float(np.max(speed_err) / denom) if speed_err.size else 0.0,
        "steady_i_rms_a": i_rms_ss,
        "steady_p_el_w": p_el_ss,
        "steady_p_mech_w": p_mech_ss,
        "steady_eta": eta_ss,
        "torque_peak_nm": float(np.max(np.abs(torque_vals))) if torque_vals.size else 0.0,
        "torque_mean_abs_nm": float(np.mean(np.abs(torque_vals))) if torque_vals.size else 0.0,
        "invalid_signal_seen": bool(invalid_signal_seen),
        "phase_voltage_peak_max_v": float(np.max(v_peak_vals)) if v_peak_vals.size else 0.0,
        "phase_voltage_peak_to_foc_limit_ratio": float(np.max(v_peak_vals) / max(v_limit, 1e-9))
        if v_limit > 0.0 and v_peak_vals.size
        else None,
    }


def build_diagnosis_payload(
    module,
    *,
    run_probes: bool,
    t_end: float,
    dt: float,
    steady_window_frac: float,
    probe_load_factors: Iterable[float],
) -> Dict[str, object]:
    env_cfg = getattr(module, "ENV")
    nameplate = _find_nameplate(module)
    estimated_motor = estimate_motor_params_from_nameplate(nameplate)
    estimated_id_ref = float(estimate_id_ref_from_nameplate(nameplate))
    torque_nom = nominal_torque_nm(nameplate)
    omega_nom = nominal_omega_rad_s(nameplate)
    load_nominal_ratio = float(getattr(env_cfg.sim, "load_torque")) / max(torque_nom, 1e-9)
    torque_capacity = estimate_foc_torque_capacity_nm(env_cfg)

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_module": str(module.__name__),
        "nameplate": nameplate,
        "current_motor": _as_plain_dict(getattr(env_cfg, "motor")),
        "current_foc": _as_plain_dict(getattr(env_cfg, "foc")),
        "current_sim": _as_plain_dict(getattr(env_cfg, "sim")),
        "estimated_motor_from_nameplate": _as_plain_dict(estimated_motor),
        "nominal": {
            "omega_rated_rad_s": omega_nom,
            "omega_rated_rpm": float(nameplate["n_rated"]),
            "torque_rated_nm": torque_nom,
            "power_rated_w": float(nameplate["P_n"]),
            "estimated_id_ref_from_nameplate_a": estimated_id_ref,
        },
        "consistency": {
            "config_load_torque_nm": float(getattr(env_cfg.sim, "load_torque")),
            "config_load_to_nominal_torque_ratio": load_nominal_ratio,
            "rough_foc_torque_capacity_nm": torque_capacity,
            "rough_foc_torque_capacity_to_nominal_ratio": float(torque_capacity / max(torque_nom, 1e-9)),
            "parameter_deltas_vs_nameplate_estimate": build_param_delta_report(env_cfg.motor, estimated_motor),
        },
        "warnings": [],
    }

    warnings: List[str] = []
    if load_nominal_ratio < 0.1:
        warnings.append(
            f"configured research load is only {load_nominal_ratio:.3f} of nominal torque; runtime task is much lighter than nameplate nominal duty"
        )
    torque_cap_ratio = float(torque_capacity / max(torque_nom, 1e-9))
    if torque_cap_ratio < 0.75:
        warnings.append(
            f"rough FOC torque capacity ratio is only {torque_cap_ratio:.3f}; current id_ref/iq_limit envelope looks underpowered for nominal torque"
        )
    delta_report = payload["consistency"]["parameter_deltas_vs_nameplate_estimate"]  # type: ignore[index]
    for key in ("Rr", "Lm", "J", "B"):
        rel = delta_report[key]["relative_delta_to_estimated"]  # type: ignore[index]
        if rel is not None and abs(float(rel)) > 0.25:
            warnings.append(f"{key} deviates materially from nameplate estimate: relative_delta={float(rel):+.3f}")

    if run_probes:
        probes: List[Dict[str, object]] = []
        load_cases = [("config_load", float(getattr(env_cfg.sim, "load_torque")))]
        for factor in probe_load_factors:
            load_cases.append((f"nominal_x{factor:.2f}", float(torque_nom * float(factor))))

        for load_label, load_value in load_cases:
            for scenario_name in ("hold", "start_stop"):
                probe = _simulate_probe(
                    env_cfg,
                    scenario_name=scenario_name,
                    load_torque=load_value,
                    t_end=t_end,
                    dt=dt,
                    steady_window_frac=steady_window_frac,
                    speed_target_rad_s=omega_nom,
                )
                probe["load_label"] = load_label
                probes.append(probe)
                util = probe.get("phase_voltage_peak_to_foc_limit_ratio")
                if scenario_name == "hold" and load_label == "config_load" and util is not None and float(util) >= 0.95:
                    warnings.append(
                        f"FOC configured-light-load hold already reaches voltage limit ratio {float(util):.3f}; modulation or voltage saturation likely contributes even before nominal duty"
                    )
                if scenario_name == "hold" and load_label == "nominal_x1.00":
                    if float(probe["steady_speed_error_rel"]) > 0.2:
                        warnings.append(
                            f"FOC nominal hold shows large steady speed error ({float(probe['steady_speed_error_rel']):.3f}); nominal operating point is not reproduced"
                        )
                    if bool(probe.get("invalid_signal_seen")):
                        warnings.append(
                            "FOC nominal hold generated invalid signals; current AO2 config/controller combination becomes numerically or physically unstable near nameplate duty"
                        )
                    if util is not None and float(util) < 0.8 and float(probe["steady_speed_error_rel"]) > 0.2:
                        warnings.append(
                            "nominal hold fails without approaching FOC voltage limit; this points more to model/controller torque mismatch than to PWM modulation saturation"
                        )
        payload["foc_probes"] = probes

    payload["warnings"] = warnings
    return payload


def _to_md(payload: Dict[str, object]) -> str:
    nominal = payload["nominal"]
    cons = payload["consistency"]
    lines: List[str] = []
    lines.append("# Motor Nominal Consistency Diagnosis")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- config_module: `{payload['config_module']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- rated_torque_nm: `{float(nominal['torque_rated_nm']):.6f}`")
    lines.append(f"- config_load_torque_nm: `{float(cons['config_load_torque_nm']):.6f}`")
    lines.append(f"- config_load_to_nominal_torque_ratio: `{float(cons['config_load_to_nominal_torque_ratio']):.6f}`")
    lines.append(f"- rough_foc_torque_capacity_nm: `{float(cons['rough_foc_torque_capacity_nm']):.6f}`")
    lines.append(
        f"- rough_foc_torque_capacity_to_nominal_ratio: `{float(cons['rough_foc_torque_capacity_to_nominal_ratio']):.6f}`"
    )
    lines.append("")
    lines.append("## Warnings")
    warnings = list(payload.get("warnings", []))
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Parameter Delta vs Nameplate Estimate")
    lines.append("| param | current | estimated | ratio | rel_delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, row in payload["consistency"]["parameter_deltas_vs_nameplate_estimate"].items():
        ratio = row["ratio_to_estimated"]
        rel = row["relative_delta_to_estimated"]
        ratio_txt = "n/a" if ratio is None else f"{float(ratio):.6f}"
        rel_txt = "n/a" if rel is None else f"{float(rel):+.6f}"
        lines.append(
            f"| {key} | {float(row['current']):.6f} | {float(row['estimated_from_nameplate']):.6f} | {ratio_txt} | {rel_txt} |"
        )
    probes = list(payload.get("foc_probes", []))
    if probes:
        lines.append("")
        lines.append("## FOC Probe Summary")
        lines.append("| scenario | load_label | load_nm | omega_ss_rpm | err_rel_ss | p_mech_w | eta_ss | v_peak/foc_limit |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for probe in probes:
            util = probe.get("phase_voltage_peak_to_foc_limit_ratio")
            util_txt = "n/a" if util is None else f"{float(util):.6f}"
            lines.append(
                f"| {probe['scenario']} | {probe['load_label']} | {float(probe['load_torque_nm']):.6f} | "
                f"{float(probe['steady_omega_rpm']):.6f} | {float(probe['steady_speed_error_rel']):.6f} | "
                f"{float(probe['steady_p_mech_w']):.6f} | {float(probe['steady_eta']):.6f} | {util_txt} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether a motor research config is physically consistent with its nameplate.")
    parser.add_argument("--config", required=True, help="Config module or path, e.g. env_research_ao2_32_4_3kw or config/env_research_ao2_32_4_3kw.py")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON and Markdown outputs.")
    parser.add_argument("--skip-probes", action="store_true", help="Skip FOC hold/start_stop probes and emit only static diagnosis.")
    parser.add_argument("--probe-t-end", type=float, default=2.0)
    parser.add_argument("--probe-dt", type=float, default=1e-3)
    parser.add_argument("--steady-window-frac", type=float, default=0.25)
    parser.add_argument(
        "--probe-load-factors",
        default="0.25,1.0",
        help="Comma-separated nominal torque factors for extra FOC probes. Config load is always included.",
    )
    args = parser.parse_args()

    load_factors = [float(item.strip()) for item in str(args.probe_load_factors).split(",") if item.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    module = _resolve_config_module(str(args.config))
    payload = build_diagnosis_payload(
        module,
        run_probes=not bool(args.skip_probes),
        t_end=float(args.probe_t_end),
        dt=float(args.probe_dt),
        steady_window_frac=float(args.steady_window_frac),
        probe_load_factors=load_factors,
    )

    json_path = out_dir / "motor_nominal_consistency.json"
    md_path = out_dir / "motor_nominal_consistency.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_md(payload), encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
