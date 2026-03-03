from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from tools.step27_pipeline import _resolve_checkpoint


def test_resolve_checkpoint_fallback_to_registry(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "best_actor.pth"
    ckpt.write_bytes(b"ok")
    registry = tmp_path / "checkpoint_registry.json"
    registry.write_text(
        json.dumps({"motors": {"air56": "best_actor.pth"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    env_cfg = SimpleNamespace(ai_eval_checkpoint_path=None)
    got = _resolve_checkpoint(
        env_cfg,
        motor_key="air56",
        config_path="config/env_research_air56_025kw.py",
        registry_path=str(registry),
    )
    assert got == ckpt.resolve()


def test_resolve_checkpoint_prefers_registry_path(tmp_path: Path, monkeypatch) -> None:
    ckpt_env = tmp_path / "env_actor.pth"
    ckpt_env.write_bytes(b"env")
    ckpt_registry = tmp_path / "registry_actor.pth"
    ckpt_registry.write_bytes(b"registry")
    registry = tmp_path / "checkpoint_registry.json"
    registry.write_text(
        json.dumps({"motors": {"air56": "registry_actor.pth"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    env_cfg = SimpleNamespace(ai_eval_checkpoint_path="env_actor.pth")
    got = _resolve_checkpoint(
        env_cfg,
        motor_key="air56",
        config_path="config/env_research_air56_025kw.py",
        registry_path=str(registry),
    )
    assert got == ckpt_registry.resolve()
