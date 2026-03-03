from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checkpoint_registry import resolve_checkpoint_path
from tools.step27_pipeline import DEFAULT_CHECKPOINT_REGISTRY, MOTOR_REGISTRY


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_nameplate(module) -> Dict[str, float]:
    for key in dir(module):
        if key.startswith("NAMEPLATE_"):
            value = getattr(module, key)
            if isinstance(value, dict) and "P_n" in value:
                return dict(value)
    raise ValueError(f"NAMEPLATE_* dict was not found in module {module.__name__}")


def _resolve_checkpoint(config_path: str, motor_key: str, registry_path: str) -> str | None:
    module = _load_module(Path(config_path).resolve())
    ckpt = getattr(module, "ai_eval_checkpoint_path", None)
    resolved = resolve_checkpoint_path(
        env_checkpoint=str(ckpt) if isinstance(ckpt, str) and ckpt.strip() else None,
        motor_key=str(motor_key),
        config_path=str(config_path),
        registry_path=str(registry_path),
        # Keep backward-compatible behavior for passport script:
        # prefer config checkpoint first, registry second.
        prefer_registry=False,
    )
    return None if resolved is None else str(resolved)


def _run_drive_characteristics(
    *,
    config_path: str,
    checkpoint: str | None,
    out_dir: Path,
    load_factors: str,
    omega_ref_pu: float,
    t_end: float,
    window_frac: float,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mic_ai.tools.drive_characteristics_ai",
        "--env-config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--omega-ref-pu",
        str(float(omega_ref_pu)),
        "--load-factors",
        str(load_factors),
        "--window-frac",
        str(window_frac),
        "--t-end",
        str(t_end),
    ]
    if checkpoint:
        cmd.extend(["--ai-checkpoint", str(checkpoint), "--ai-mode", "ai_id_ref"])
    subprocess.run(cmd, check=True)


def _safe_pct(delta: float, base: float) -> float:
    if abs(float(base)) < 1e-12:
        return 0.0
    return 100.0 * float(delta) / float(base)


def _pick_nominal_row(df: pd.DataFrame, policy: str) -> pd.Series | None:
    part = df[df["policy"].astype(str) == str(policy)].copy()
    if part.empty:
        return None
    part["load_factor_abs_err"] = (pd.to_numeric(part["load_factor"], errors="coerce") - 1.0).abs()
    part = part.sort_values("load_factor_abs_err")
    return part.iloc[0]


def _build_rows_for_motor(
    *,
    motor_key: str,
    config_path: str,
    raw_dir: Path,
    checkpoint_registry: str,
    load_factors: str,
    t_end: float,
    window_frac: float,
) -> List[Dict[str, object]]:
    ckpt = _resolve_checkpoint(config_path, motor_key, checkpoint_registry)
    omega_used: float | None = None
    last_error: Exception | None = None
    for omega in (1.0, 0.9, 0.8, 0.7):
        try:
            _run_drive_characteristics(
                config_path=config_path,
                checkpoint=ckpt,
                out_dir=raw_dir,
                load_factors=load_factors,
                omega_ref_pu=float(omega),
                t_end=t_end,
                window_frac=window_frac,
            )
            omega_used = float(omega)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if omega_used is None and last_error is not None:
        raise last_error

    csv_path = raw_dir / "load_characteristics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    module = _load_module(Path(config_path).resolve())
    plate = _resolve_nameplate(module)
    p_nom_kw = float(plate["P_n"]) / 1000.0
    i_nom = float(plate["I_n"])
    n_nom = float(plate["n_rated"])
    eta_nom_pct = float(plate["eta_n"]) * 100.0
    cos_nom = float(plate["cos_phi_n"])

    df = pd.read_csv(csv_path)
    policies = ["FOC"]
    if "MIC_AI" in set(df["policy"].astype(str).unique().tolist()):
        policies.append("MIC_AI")
    elif "MIC_RULE" in set(df["policy"].astype(str).unique().tolist()):
        policies.append("MIC_RULE")

    rows: List[Dict[str, object]] = []
    for policy in policies:
        row = _pick_nominal_row(df, policy)
        if row is None:
            continue
        p2 = float(row["p2_kw"])
        i1 = float(row["i_rms"])
        n2 = float(row["n2_rpm"])
        eta_pct = float(row["eta_pct"])
        cos_phi = float(row["cos_phi"])
        rows.append(
            {
                "motor": motor_key,
                "policy": policy,
                "load_factor": float(row["load_factor"]),
                "p2_kw_model": p2,
                "p2_kw_nameplate": p_nom_kw,
                "p2_kw_delta_pct": _safe_pct(p2 - p_nom_kw, p_nom_kw),
                "i1_a_model": i1,
                "i1_a_nameplate": i_nom,
                "i1_a_delta_pct": _safe_pct(i1 - i_nom, i_nom),
                "n2_rpm_model": n2,
                "n2_rpm_nameplate": n_nom,
                "n2_rpm_delta_pct": _safe_pct(n2 - n_nom, n_nom),
                "eta_pct_model": eta_pct,
                "eta_pct_nameplate": eta_nom_pct,
                "eta_pct_delta_abs": eta_pct - eta_nom_pct,
                "cos_phi_model": cos_phi,
                "cos_phi_nameplate": cos_nom,
                "cos_phi_delta_abs": cos_phi - cos_nom,
                "checkpoint_used": "" if ckpt is None else str(ckpt),
                "omega_ref_pu_used": float(omega_used) if omega_used is not None else float("nan"),
                "load_csv": str(csv_path.resolve()),
            }
        )
    return rows


def _to_md(rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Against-Passport Table (3 motors)")
    lines.append("")
    lines.append("| motor | policy | load_factor | omega_ref_pu | P2 delta % | I1 delta % | n2 delta % | eta delta abs p.p. | cos(phi) delta abs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {motor} | {policy} | {load_factor:.3f} | {omega_ref_pu_used:.2f} | {p2_kw_delta_pct:+.2f} | {i1_a_delta_pct:+.2f} | {n2_rpm_delta_pct:+.2f} | {eta_pct_delta_abs:+.2f} | {cos_phi_delta_abs:+.4f} |".format(
                **r
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build passport comparison table for AIR56/AL31/AO2 at nominal load point.")
    parser.add_argument("--motors", default="air56,al31,ao2")
    parser.add_argument("--checkpoint-registry", default=DEFAULT_CHECKPOINT_REGISTRY)
    parser.add_argument("--load-factors", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--t-end", type=float, default=1.5)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--out-root", default="paper/ieee_2026/data/passport")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    motors = [m.strip() for m in str(args.motors).split(",") if m.strip()]
    for m in motors:
        if m not in MOTOR_REGISTRY:
            raise ValueError(f"Unknown motor key: {m}")

    tag = str(args.tag).strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).resolve() / tag
    raw_root = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for motor in motors:
        spec = MOTOR_REGISTRY[motor]
        motor_raw = raw_root / motor
        motor_raw.mkdir(parents=True, exist_ok=True)
        try:
            rows.extend(
                _build_rows_for_motor(
                    motor_key=motor,
                    config_path=spec.config_path,
                    raw_dir=motor_raw,
                    checkpoint_registry=str(args.checkpoint_registry),
                    load_factors=str(args.load_factors),
                    t_end=float(args.t_end),
                    window_frac=float(args.window_frac),
                )
            )
        except Exception as exc:
            failures.append({"motor": motor, "error": str(exc)})
            print(f"[passport][WARN] failed motor={motor}: {exc}")

    df = pd.DataFrame(rows)
    csv_path = out_root / "passport_compare_3motors.csv"
    md_path = out_root / "passport_compare_3motors.md"
    json_path = out_root / "passport_compare_3motors.json"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_md(rows), encoding="utf-8")
    payload = {"rows": rows, "failures": failures}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[passport] saved: {csv_path}")
    print(f"[passport] saved: {md_path}")
    print(f"[passport] saved: {json_path}")


if __name__ == "__main__":
    main()
