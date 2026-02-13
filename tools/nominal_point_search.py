from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import EnvConfig, create_default_env
from drivers import SimDriver


def _load_env_config(path: str | None) -> EnvConfig:
    if path is None:
        return create_default_env()
    from mic_ai.core.env import make_env_from_config

    env = make_env_from_config(path)
    return env.env_config


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _run_foc(env_cfg: EnvConfig, dt: float, t_end: float, seed: int) -> Dict[str, np.ndarray]:
    sim = SimDriver(env_cfg)
    sim.reset(seed=seed)
    sim.set_mode("FOC")
    n_steps = int(max(t_end / dt, 1))
    t = np.zeros(n_steps, dtype=float)
    omega = np.zeros(n_steps, dtype=float)
    omega_ref = np.zeros(n_steps, dtype=float)
    for k in range(n_steps):
        sim.step()
        obs = sim.read_obs()
        t[k] = obs["t"]
        omega[k] = obs["omega"]
        omega_ref[k] = obs["omega_ref"]
        if sim.get_last_fault():
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            break
    return {"t": t, "omega": omega, "omega_ref": omega_ref}


def _mean_abs_err(series: Dict[str, np.ndarray], window_frac: float) -> float:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    err = np.abs(series["omega_ref"][sl] - series["omega"][sl])
    return float(np.mean(err)) if err.size else 0.0


def _parse_grid(text: str | None) -> Sequence[float]:
    if not text:
        return ()
    text = text.strip()
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("Grid range must be start:stop:step")
        start, stop, step = (float(p) for p in parts)
        if step == 0.0:
            raise ValueError("Grid step must be non-zero")
        values = []
        v = start
        if step > 0:
            while v <= stop + 1e-9:
                values.append(round(v, 10))
                v += step
        else:
            while v >= stop - 1e-9:
                values.append(round(v, 10))
                v += step
        return values
    values = [float(p.strip()) for p in text.split(",") if p.strip()]
    return values


def _best_omega_pu(
    env_cfg: EnvConfig,
    omega_grid: Iterable[float],
    dt: float,
    t_end: float,
    window_frac: float,
    err_tol: float,
) -> Dict[str, float]:
    results: List[Dict[str, float]] = []
    for omega_pu in omega_grid:
        sim_cfg = replace(env_cfg.sim, scenario_name=f"hold:{omega_pu}", dt=float(dt), t_end=float(t_end))
        cfg = replace(env_cfg, sim=sim_cfg)
        series = _run_foc(cfg, float(dt), float(t_end), seed=123)
        err = _mean_abs_err(series, float(window_frac))
        omega_ref = float(series["omega_ref"][-1]) if series["omega_ref"].size else 0.0
        err_rel = err / max(abs(omega_ref), 1e-9)
        results.append({"omega_pu": float(omega_pu), "err": err, "err_rel": err_rel})

    ok = [r for r in results if r["err_rel"] <= err_tol]
    has_ok = bool(ok)
    if has_ok:
        best = max(ok, key=lambda r: r["omega_pu"])
    else:
        best = min(results, key=lambda r: r["err_rel"]) if results else {"omega_pu": 0.0, "err": 0.0, "err_rel": 0.0}

    err_rel_1 = next((r["err_rel"] for r in results if abs(r["omega_pu"] - 1.0) < 1e-6), None)
    ok_at_1_0 = err_rel_1 is not None and err_rel_1 <= err_tol
    return {
        "max_ok_omega_pu": float(best["omega_pu"]),
        "err_rel_at_max": float(best["err_rel"]),
        "err_rel_at_1_0": float(err_rel_1) if err_rel_1 is not None else float("nan"),
        "has_ok": bool(has_ok),
        "ok_at_1_0": bool(ok_at_1_0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max omega_pu achievable at nominal load.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--load-torque", type=float, default=None)
    parser.add_argument("--iq-limit", type=float, default=None)
    parser.add_argument("--iq-grid", default=None, help="Comma list or start:stop:step")
    parser.add_argument("--load-grid", default=None, help="Comma list or start:stop:step")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--err-tol", type=float, default=0.02)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    env_cfg = _load_env_config(args.env_config)
    if args.iq_limit is not None:
        env_cfg = replace(env_cfg, foc=replace(env_cfg.foc, iq_limit=float(args.iq_limit)))
    base_load = float(args.load_torque) if args.load_torque is not None else float(env_cfg.sim.load_torque)

    omega_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    iq_grid = _parse_grid(args.iq_grid)
    load_grid = _parse_grid(args.load_grid)

    if iq_grid or load_grid:
        if not iq_grid:
            iq_grid = [float(getattr(env_cfg.foc, "iq_limit", 0.0) or 0.0)]
        if not load_grid:
            load_grid = [float(base_load)]
        rows: List[Dict[str, float]] = []
        for iq in iq_grid:
            cfg_iq = replace(env_cfg, foc=replace(env_cfg.foc, iq_limit=float(iq)))
            for load in load_grid:
                cfg_load = replace(cfg_iq, sim=replace(cfg_iq.sim, load_torque=float(load)))
                best = _best_omega_pu(
                    cfg_load,
                    omega_grid,
                    float(args.dt),
                    float(args.t_end),
                    float(args.window_frac),
                    float(args.err_tol),
                )
                rows.append(
                    {
                        "iq_limit": float(iq),
                        "load_torque": float(load),
                        "max_ok_omega_pu": float(best["max_ok_omega_pu"]),
                        "err_rel_at_max": float(best["err_rel_at_max"]),
                        "err_rel_at_1_0": float(best["err_rel_at_1_0"]),
                        "has_ok": bool(best["has_ok"]),
                        "ok_at_1_0": bool(best["ok_at_1_0"]),
                    }
                )

        header = "| iq_limit | load_torque | max_ok_omega_pu | err_rel_at_max | err_rel_at_1.0 | ok_at_1.0 |"
        sep = "| --- | --- | --- | --- | --- | --- |"
        lines = [header, sep]
        for r in rows:
            lines.append(
                "| {iq_limit:.3g} | {load_torque:.3g} | {max_ok_omega_pu:.3g} | {err_rel_at_max:.3g} | {err_rel_at_1_0:.3g} | {ok_at_1_0} |".format(
                    **r
                )
            )
        text = "\n".join(lines)
        if args.save_path:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_path).write_text(text, encoding="utf-8")
        print(text)
    else:
        results: List[Dict[str, float]] = []
        sim_cfg_base = replace(env_cfg.sim, load_torque=float(base_load))
        env_cfg = replace(env_cfg, sim=sim_cfg_base)
        for omega_pu in omega_grid:
            sim_cfg = replace(env_cfg.sim, scenario_name=f"hold:{omega_pu}", dt=float(args.dt), t_end=float(args.t_end))
            cfg = replace(env_cfg, sim=sim_cfg)
            series = _run_foc(cfg, float(args.dt), float(args.t_end), seed=123)
            err = _mean_abs_err(series, float(args.window_frac))
            omega_ref = float(series["omega_ref"][-1]) if series["omega_ref"].size else 0.0
            err_rel = err / max(abs(omega_ref), 1e-9)
            results.append({"omega_pu": omega_pu, "err": err, "err_rel": err_rel})

        for r in results:
            status = "OK" if r["err_rel"] <= float(args.err_tol) else "FAIL"
            print(f"omega_pu={r['omega_pu']:.1f} err={r['err']:.3g} rel={r['err_rel']:.3g} -> {status}")


if __name__ == "__main__":
    main()
