from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def resolve_registry_path(registry_path: str | None) -> Path | None:
    if not registry_path:
        return None
    path = Path(str(registry_path)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _normalize_registry_sources(payload: Dict[str, object]) -> List[Dict[str, object]]:
    sources: List[Dict[str, object]] = []
    for key in ("motors", "configs", "by_motor"):
        src = payload.get(key)
        if isinstance(src, dict):
            sources.append(src)
    if not sources:
        sources.append(payload)
    return sources


def load_checkpoint_registry(registry_path: str | None) -> Dict[str, str]:
    path = resolve_registry_path(registry_path)
    if path is None or (not path.exists()):
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint registry must be a JSON object: {path}")

    registry: Dict[str, str] = {}
    for src in _normalize_registry_sources(payload):
        for key, value in src.items():
            if isinstance(key, str) and isinstance(value, str):
                registry[key.strip().lower()] = value.strip()
    return registry


def _resolve_checkpoint_path(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _candidate_keys(*, motor_key: str | None, config_path: str | None) -> List[str]:
    keys: List[str] = []
    if motor_key:
        keys.append(str(motor_key).strip().lower())
    if config_path:
        cfg = Path(str(config_path))
        keys.append(cfg.stem.lower())
        keys.append(cfg.name.lower())
    out: List[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def resolve_checkpoint_candidates(
    *,
    env_checkpoint: str | None,
    motor_key: str | None,
    config_path: str | None,
    registry_path: str | None,
    prefer_registry: bool = True,
) -> List[Tuple[str, Path]]:
    registry_candidates: List[Tuple[str, Path]] = []
    env_candidates: List[Tuple[str, Path]] = []

    registry = load_checkpoint_registry(registry_path)
    for key in _candidate_keys(motor_key=motor_key, config_path=config_path):
        raw = registry.get(key)
        if raw:
            registry_candidates.append((f"registry:{key}", _resolve_checkpoint_path(raw)))

    if env_checkpoint:
        env_candidates.append(("env.ai_eval_checkpoint_path", _resolve_checkpoint_path(env_checkpoint)))

    return registry_candidates + env_candidates if prefer_registry else env_candidates + registry_candidates


def resolve_checkpoint_path(
    *,
    env_checkpoint: str | None,
    motor_key: str | None,
    config_path: str | None,
    registry_path: str | None,
    prefer_registry: bool = True,
) -> Path | None:
    for _src, path in resolve_checkpoint_candidates(
        env_checkpoint=env_checkpoint,
        motor_key=motor_key,
        config_path=config_path,
        registry_path=registry_path,
        prefer_registry=prefer_registry,
    ):
        if path.exists():
            return path
    return None
