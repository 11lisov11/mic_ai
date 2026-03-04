from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MOTOR_MODULES: Dict[str, str] = {
    "air56": "config.env_research_air56_025kw",
    "al31": "config.env_research_al31_4_06kw",
    "ao2": "config.env_research_ao2_32_4_3kw",
}


def _safe_get(obj: object, name: str, default: object = None) -> object:
    return getattr(obj, name, default)


def _obj_to_dict(obj: object) -> Dict[str, object]:
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
        val = getattr(obj, key)
        if callable(val):
            continue
        if isinstance(val, (int, float, str, bool, type(None))):
            out[key] = val
    return out


def _pick(d: Dict[str, object], keys: List[str]) -> Dict[str, object]:
    return {k: d.get(k) for k in keys if k in d}


def _collect_module(mod_name: str) -> Dict[str, object]:
    mod = importlib.import_module(mod_name)
    env = getattr(mod, "ENV")
    nameplate_key = next((k for k in dir(mod) if k.startswith("NAMEPLATE_")), "")
    nameplate = dict(getattr(mod, nameplate_key)) if nameplate_key else {}

    motor_d = _obj_to_dict(_safe_get(env, "motor"))
    inv_d = _obj_to_dict(_safe_get(env, "inverter"))
    sim_d = _obj_to_dict(_safe_get(env, "sim"))
    foc_d = _obj_to_dict(_safe_get(env, "foc"))

    return {
        "module": mod_name,
        "nameplate": nameplate,
        "motor_model": _pick(motor_d, ["Rs", "Rr", "Lm", "Ls_sigma", "Lr_sigma", "J", "B", "p", "f_nom", "U_ll_nom", "I_n"]),
        "inverter_model": _pick(inv_d, ["Vdc", "r_out", "dead_time", "v_drop"]),
        "sim_setup": _pick(sim_d, ["dt", "t_end", "scenario_name", "load_torque", "save_prefix"]),
        "foc_setup": _pick(foc_d, ["id_ref", "iq_limit", "kp_speed", "ki_speed"]),
        "loss_model": {
            "loss_inv_r": getattr(mod, "loss_inv_r", None),
            "loss_core_k": getattr(mod, "loss_core_k", None),
            "loss_core_omega_exp": getattr(mod, "loss_core_omega_exp", None),
            "loss_core_psi_exp": getattr(mod, "loss_core_psi_exp", None),
            "id_ref_lut_path": getattr(mod, "id_ref_lut_path", None),
        },
        "ai_eval": {
            "checkpoint_path": getattr(mod, "ai_eval_checkpoint_path", None),
            "id_ref_alpha": getattr(mod, "ai_eval_id_ref_alpha", None),
            "delta_id_max": getattr(mod, "ai_eval_delta_id_max", None),
            "id_ref_gate_speed_tol_rel": getattr(mod, "ai_eval_id_ref_gate_speed_tol_rel", None),
            "supervisor_enabled": getattr(mod, "ai_eval_supervisor_enabled", None),
            "supervisor_objective": getattr(mod, "ai_eval_sup_objective", None),
        },
    }


def _to_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Physical Config Policy (3 Motors)")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Source of truth: research configs in `config/` for AIR56, AL31, AO2.")
    lines.append("- Purpose: фиксировать assumptions физической модели, потерь, инвертора и базового FOC для reproducible сравнения.")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("| Motor | Module | Pn, W | Un, V | In, A | eta_n | cos_phi_n | dt, s | loss_inv_r | loss_core_k |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for motor, item in payload["motors"].items():
        np = dict(item.get("nameplate", {}))
        sim = dict(item.get("sim_setup", {}))
        loss = dict(item.get("loss_model", {}))
        lines.append(
            f"| {motor.upper()} | {item.get('module','')} | "
            f"{float(np.get('P_n', 0.0)):.1f} | {float(np.get('U_ll', 0.0)):.1f} | {float(np.get('I_n', 0.0)):.3f} | "
            f"{float(np.get('eta_n', 0.0)):.3f} | {float(np.get('cos_phi_n', 0.0)):.3f} | "
            f"{float(sim.get('dt', 0.0)):.6f} | {float(loss.get('loss_inv_r', 0.0)):.6f} | {float(loss.get('loss_core_k', 0.0)):.6f} |"
        )
    lines.append("")

    for motor, item in payload["motors"].items():
        lines.append(f"## {motor.upper()}")
        lines.append(f"- module: `{item.get('module','')}`")
        lines.append(f"- nameplate: `{json.dumps(item.get('nameplate', {}), ensure_ascii=False)}`")
        lines.append(f"- motor_model: `{json.dumps(item.get('motor_model', {}), ensure_ascii=False)}`")
        lines.append(f"- inverter_model: `{json.dumps(item.get('inverter_model', {}), ensure_ascii=False)}`")
        lines.append(f"- sim_setup: `{json.dumps(item.get('sim_setup', {}), ensure_ascii=False)}`")
        lines.append(f"- foc_setup: `{json.dumps(item.get('foc_setup', {}), ensure_ascii=False)}`")
        lines.append(f"- loss_model: `{json.dumps(item.get('loss_model', {}), ensure_ascii=False)}`")
        lines.append(f"- ai_eval: `{json.dumps(item.get('ai_eval', {}), ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified physical-config policy report for AIR56/AL31/AO2.")
    parser.add_argument("--out-json", default="docs/physical_config_policy_3motors.json")
    parser.add_argument("--out-md", default="docs/physical_config_policy_3motors.md")
    args = parser.parse_args()

    motors: Dict[str, object] = {}
    for key, mod_name in MOTOR_MODULES.items():
        motors[key] = _collect_module(mod_name)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "motors": motors,
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
