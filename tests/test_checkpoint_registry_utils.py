from __future__ import annotations

import json
from pathlib import Path

from tools.checkpoint_registry import (
    load_checkpoint_registry,
    resolve_checkpoint_candidates,
    resolve_checkpoint_path,
)


def test_load_checkpoint_registry_accepts_by_motor_and_bom(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    payload = {"by_motor": {"air56": "a.pth"}, "configs": {"env.py": "b.pth"}}
    # Explicitly write with BOM-compatible encoding path (module reads with utf-8-sig).
    registry.write_text(json.dumps(payload), encoding="utf-8-sig")
    got = load_checkpoint_registry(str(registry))
    assert got["air56"] == "a.pth"
    assert got["env.py"] == "b.pth"


def test_resolve_checkpoint_prefers_registry_when_requested(tmp_path: Path, monkeypatch) -> None:
    reg_ckpt = tmp_path / "registry_actor.pth"
    env_ckpt = tmp_path / "env_actor.pth"
    reg_ckpt.write_bytes(b"reg")
    env_ckpt.write_bytes(b"env")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"motors": {"air56": "registry_actor.pth"}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    got = resolve_checkpoint_path(
        env_checkpoint="env_actor.pth",
        motor_key="air56",
        config_path="config/env_research_air56_025kw.py",
        registry_path=str(registry),
        prefer_registry=True,
    )
    assert got == reg_ckpt.resolve()


def test_resolve_checkpoint_candidate_order_env_first(tmp_path: Path, monkeypatch) -> None:
    reg_ckpt = tmp_path / "registry_actor.pth"
    env_ckpt = tmp_path / "env_actor.pth"
    reg_ckpt.write_bytes(b"reg")
    env_ckpt.write_bytes(b"env")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"motors": {"air56": "registry_actor.pth"}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    candidates = resolve_checkpoint_candidates(
        env_checkpoint="env_actor.pth",
        motor_key="air56",
        config_path="config/env_research_air56_025kw.py",
        registry_path=str(registry),
        prefer_registry=False,
    )
    assert candidates[0][1] == env_ckpt.resolve()
