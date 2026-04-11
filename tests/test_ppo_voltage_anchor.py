from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent


def test_ppo_voltage_actor_anchor_penalizes_policy_drift() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    agent = PPOVoltageAgent(
        feature_keys=["omega_norm", "iq_norm"],
        action_dim=1,
        device="cpu",
        hidden_sizes=(8, 8),
        lr=1e-3,
        entropy_coef=0.0,
        train_epochs=1,
        minibatch_frac=1.0,
    )
    agent.set_actor_anchor_from_current(0.5)

    assert agent.actor_anchor_net is not None
    assert agent.actor_anchor_coef == 0.5
    assert all(not p.requires_grad for p in agent.actor_anchor_net.parameters())

    with torch.no_grad():
        agent.net.actor_head.bias.add_(0.5)

    obs = {"omega_norm": 0.2, "iq_norm": -0.1}
    action, logp, value = agent.act(obs)
    agent.store(obs, action, logp, reward=0.0, done=True, value=value)
    metrics = agent.update(last_value=0.0)

    assert metrics["anchor_loss"] > 0.0
    assert agent.last_anchor_loss > 0.0


def test_ppo_voltage_update_sanitizes_nan_transition_data() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    agent = PPOVoltageAgent(
        feature_keys=["omega_norm", "iq_norm"],
        action_dim=2,
        device="cpu",
        hidden_sizes=(8, 8),
        lr=1e-3,
        entropy_coef=0.0,
        train_epochs=1,
        minibatch_frac=1.0,
    )

    obs = {"omega_norm": float("nan"), "iq_norm": float("inf")}
    action = np.array([float("nan"), float("inf")], dtype=np.float32)
    agent.store(obs, action, logprob=float("nan"), reward=float("nan"), done=False, value=float("nan"))
    agent.store({"omega_norm": 0.1, "iq_norm": -0.2}, np.array([0.0, 0.0], dtype=np.float32), logprob=0.0, reward=0.0, done=True, value=0.0)

    metrics = agent.update(last_value=float("nan"))

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["anchor_loss"])
