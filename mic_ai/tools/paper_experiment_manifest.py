from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mic_ai.ai.ai_voltage_config import get_curriculum_config, get_voltage_scale, load_ai_voltage_config
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS, _motor_key_from_config, resolve_config_path
from mic_ai.core.env import make_env_from_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root()), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def _pkg_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "matplotlib", "gym"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", None)
            if ver:
                versions[name] = str(ver)
        except Exception:
            pass
    return versions


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _omega_nominal_rad_s(pole_pairs: int) -> float:
    p = max(int(pole_pairs), 1)
    return float(2.0 * np.pi * 10.0 / p)


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(int(chunk_bytes))
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _checkpoint_meta(ckpt: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(ckpt), "exists": ckpt.exists()}
    if not ckpt.exists():
        return info
    try:
        st = ckpt.stat()
        info.update({"bytes": int(st.st_size), "mtime": float(st.st_mtime)})
    except Exception:
        pass

    digest = _sha256(ckpt)
    if digest is not None:
        info["sha256"] = digest

    try:
        import torch  # type: ignore
    except Exception:
        return info

    try:
        state = torch.load(ckpt, map_location="cpu")
        if not isinstance(state, dict):
            return info

        hidden_sizes: List[int] = []
        w0 = state.get("actor_body.0.weight")
        w2 = state.get("actor_body.2.weight")
        if hasattr(w0, "shape") and hasattr(w2, "shape"):
            try:
                hidden_sizes = [int(w0.shape[0]), int(w2.shape[0])]
            except Exception:
                hidden_sizes = []
        if hidden_sizes:
            info["hidden_sizes"] = hidden_sizes

        param_count = 0
        for v in state.values():
            if hasattr(v, "numel"):
                try:
                    param_count += int(v.numel())
                except Exception:
                    pass
        if param_count > 0:
            info["param_count"] = int(param_count)
    except Exception:
        pass
    return info


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export experiment manifest for paper (reproducibility).")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--ai-checkpoint", default="outputs/demo_ai/checkpoints/motor1/last_actor.pth")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--episodes-per-stage", type=int, default=25)
    p.add_argument("--voltage-scale", type=float, default=None, help="If omitted, uses default from ai_voltage_config.")
    p.add_argument("--disable-noise", action="store_true")
    p.add_argument("--sigma-omega", type=float, default=0.05)
    p.add_argument("--sigma-i", type=float, default=0.03)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/paper_manifest")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_config = resolve_config_path(str(args.env_config))
    motor_key = _motor_key_from_config(str(env_config))
    env_cfg = make_env_from_config(str(env_config)).env_config

    cfg = load_ai_voltage_config()
    curriculum = get_curriculum_config(cfg)
    omega_pu_stages = list(curriculum.get("omega_pu_stages", [0.3, 0.5]))
    stage_episode_boundaries_cfg = list(curriculum.get("stage_episode_boundaries", []))
    piecewise_steps = list(curriculum.get("piecewise_steps", (150, 300)))
    piecewise_multipliers = list(curriculum.get("piecewise_multipliers", (1.0, 0.8, 1.0)))

    vdc = _safe_float(getattr(getattr(env_cfg, "inverter", None), "Vdc", 0.0))
    if args.voltage_scale is None:
        v_scale = float(get_voltage_scale(cfg, motor_key))
    else:
        v_scale = float(args.voltage_scale)
    v_limit = float(v_scale) * (0.8 * vdc / np.sqrt(3.0)) if vdc > 0 else None

    iq_limit = _safe_float(getattr(getattr(env_cfg, "foc", None), "iq_limit", 0.0))
    pole_pairs = int(getattr(getattr(env_cfg, "motor", None), "p", 1) or 1)
    omega_nominal = _omega_nominal_rad_s(pole_pairs)
    omega_stage_base = [float(pu) * float(omega_nominal) for pu in omega_pu_stages]

    noise_enabled = not bool(args.disable_noise)
    sigma_omega = 0.0 if not noise_enabled else float(args.sigma_omega)
    sigma_i = 0.0 if not noise_enabled else float(args.sigma_i)

    ckpt = Path(args.ai_checkpoint).resolve()
    ckpt_info = _checkpoint_meta(ckpt)

    dt = _safe_float(getattr(getattr(env_cfg, "sim", None), "dt", 0.0))
    episode_duration_s = float(dt) * float(int(args.episode_steps)) if dt > 0 else None
    stage_episode_boundaries_eval = (
        [int(args.episodes_per_stage) * (k + 1) for k in range(max(len(omega_pu_stages) - 1, 0))]
        if int(args.episodes_per_stage) > 0
        else []
    )

    manifest: Dict[str, Any] = {
        "project": "MIC_AI",
        "git_head": _git_head(),
        "cwd": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": _pkg_versions(),
        "env_config_path": str(env_config),
        "motor_key": motor_key,
        "env_config": asdict(env_cfg) if hasattr(env_cfg, "__dataclass_fields__") else None,
        "episode_steps": int(args.episode_steps),
        "episode_duration_s": float(episode_duration_s) if episode_duration_s is not None else None,
        "episodes_per_stage": int(args.episodes_per_stage),
        "stage_episode_boundaries_eval": stage_episode_boundaries_eval,
        "seeds": {"seed0": int(args.seed0), "n": int(args.seeds), "values": [int(args.seed0) + k for k in range(int(args.seeds))]},
        "limits": {
            "Vdc": float(vdc),
            "voltage_scale": float(v_scale),
            "v_limit_line_to_neutral_peak": float(v_limit) if v_limit is not None else None,
            "iq_limit": float(iq_limit) if iq_limit else None,
            "foc_v_limit": _safe_float(getattr(getattr(env_cfg, "foc", None), "v_limit", 0.0)) or None,
        },
        "noise": {"enabled": bool(noise_enabled), "sigma_omega": float(sigma_omega), "sigma_i": float(sigma_i)},
        "agent": {"action_space": "[-1, 1]^2", "features": list(FEATURE_KEYS)},
        "omega_ref_profile": {
            "omega_nominal_rad_s": float(omega_nominal),
            "omega_pu_stages": omega_pu_stages,
            "omega_stage_base_rad_s": omega_stage_base,
            "stage_episode_boundaries_cfg": stage_episode_boundaries_cfg,
            "piecewise_steps": piecewise_steps,
            "piecewise_multipliers": piecewise_multipliers,
        },
        "load": {"load_torque": _safe_float(getattr(getattr(env_cfg, "sim", None), "load_torque", 0.0))},
        "sim": {"dt": float(dt), "t_end": _safe_float(getattr(getattr(env_cfg, "sim", None), "t_end", 0.0))},
        "ai_checkpoint": ckpt_info,
    }

    json_path = out_dir / f"manifest_{motor_key}.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8-sig")

    md_lines = [
        "# Manifest эксперимента",
        "",
        f"- `git_head`: `{manifest.get('git_head')}`",
        f"- `python`: `{manifest['python']['version']}`",
        f"- `env_config`: `{manifest['env_config_path']}`",
        f"- `ai_checkpoint`: `{ckpt_info.get('path')}`",
        "",
        "## Ограничения",
        f"- `Vdc`: {manifest['limits']['Vdc']} В",
        f"- `voltage_scale`: {manifest['limits']['voltage_scale']}",
        f"- `v_limit` (L-N peak): {manifest['limits']['v_limit_line_to_neutral_peak']} В",
        f"- `iq_limit`: {manifest['limits']['iq_limit']} А",
        "",
        "## Профиль ω_ref",
        f"- `omega_nominal`: {manifest['omega_ref_profile']['omega_nominal_rad_s']} рад/с",
        f"- `omega_pu_stages`: {manifest['omega_ref_profile']['omega_pu_stages']}",
        f"- `piecewise_steps`: {manifest['omega_ref_profile']['piecewise_steps']}",
        f"- `piecewise_multipliers`: {manifest['omega_ref_profile']['piecewise_multipliers']}",
        "",
        "## Оценивание",
        f"- `episode_steps`: {manifest['episode_steps']}",
        f"- `episode_duration_s`: {manifest['episode_duration_s']}",
        f"- `episodes_per_stage`: {manifest['episodes_per_stage']}",
        f"- `stage_episode_boundaries_eval`: {manifest['stage_episode_boundaries_eval']}",
        f"- `seeds`: {manifest['seeds']['values']}",
        "",
        "## Шум измерений",
        f"- enabled: {manifest['noise']['enabled']}",
        f"- `sigma_omega`: {manifest['noise']['sigma_omega']} рад/с",
        f"- `sigma_i`: {manifest['noise']['sigma_i']} А",
    ]
    md_path = out_dir / f"manifest_{motor_key}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8-sig")

    print(f"Saved manifest: {json_path}")
    print(f"Saved manifest: {md_path}")


if __name__ == "__main__":
    main()
