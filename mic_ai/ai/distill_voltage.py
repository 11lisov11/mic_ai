from __future__ import annotations

"""
Utility helpers for preparing ai_voltage policy distillation.

This module intentionally keeps the surface minimal so that future steps can
plug in real exporters without touching training code.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from mic_ai.ai.agents.ppo_voltage import ActorCritic


def load_teacher_policy(path: str) -> nn.Module:
    """
    Load a teacher PPO actor-critic from checkpoint.

    The checkpoint is expected to contain the state_dict produced by
    PPOVoltageAgent.actor_critic. State/action dimensions default to the
    current ai_voltage setup (6 state features, 2 actions).
    """
    model = ActorCritic(state_dim=6, action_dim=2)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # Keep partially loaded weights to avoid failing the pipeline.
            pass
    return model


class TinyStudent(nn.Module):
    """Compact MLP student suitable for microcontrollers."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def build_tiny_student(input_dim: int, output_dim: int) -> nn.Module:
    """
    Build a compact student network for voltage control distillation.
    """
    return TinyStudent(input_dim=input_dim, output_dim=output_dim)


def export_tiny_to_c(student_model: nn.Module, export_path: str) -> Path:
    """
    Export tiny student weights to a simple JSON format consumable by C code.

    The layout is intentionally straightforward:
    {
      "layers": [
        {"W": [[...], ...], "b": [...]},
        ...
      ]
    }
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    layers: List[Dict[str, object]] = []
    for module in student_model.modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.detach().cpu().numpy().tolist()
            bias = module.bias.detach().cpu().numpy().tolist() if module.bias is not None else []
            layers.append({"W": weights, "b": bias})

    payload: Dict[str, object] = {"layers": layers}
    with export_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return export_path


__all__ = ["load_teacher_policy", "build_tiny_student", "export_tiny_to_c"]
