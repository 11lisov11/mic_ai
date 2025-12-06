from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List


class SimpleAdaptiveAgent:
    """
    Minimal REINFORCE agent with linear policy over hand-crafted features.
    The agent outputs a single action delta_iq_rel in [-1, 1] and updates once per episode.
    """

    def __init__(self, feature_keys: Iterable[str], lr: float = 1e-3, action_scale: float = 1.0):
        self.feature_keys = list(feature_keys)
        self.lr = lr
        self.action_scale = action_scale
        self.theta: np.ndarray | None = None
        self.episode: List[Dict[str, np.ndarray]] = []

    def _features(self, obs: Dict[str, float]) -> np.ndarray:
        # Collect phi from obs in the provided order + bias term.
        phi = np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)
        phi = np.concatenate([phi, np.array([1.0], dtype=np.float32)])
        return phi

    def start_episode(self) -> None:
        self.episode.clear()

    def act(self, obs: Dict[str, float]) -> float:
        phi = self._features(obs)
        if self.theta is None:
            self.theta = np.zeros_like(phi)

        z = float(np.dot(self.theta, phi))
        delta = float(np.tanh(z)) * self.action_scale  # in [-1, 1]
        logp_approx = z

        self.episode.append({"phi": phi, "logp": logp_approx, "reward": 0.0})

        return delta

    def record_reward(self, r: float) -> None:
        if not self.episode:
            return
        self.episode[-1]["reward"] = float(r)

    def update_after_episode(self) -> float:
        if not self.episode:
            return 0.0

        rewards = np.array([e["reward"] for e in self.episode], dtype=np.float32)

        # Returns without discount (episodes are short).
        R = np.zeros_like(rewards)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G += rewards[t]
            R[t] = G

        R_mean = float(R.mean())
        R_std = float(R.std() + 1e-8)
        R_hat = (R - R_mean) / R_std

        logps = np.array([e["logp"] for e in self.episode], dtype=np.float32)
        phis = np.stack([e["phi"] for e in self.episode], axis=0)

        weights = R_hat[:, None]
        grad = -np.mean(weights * phis, axis=0)

        # Gradient clipping.
        grad = np.clip(grad, -1.0, 1.0)

        if self.theta is None:
            self.theta = np.zeros(phis.shape[1], dtype=np.float32)
        self.theta -= self.lr * grad

        # Keep parameters bounded to avoid runaway.
        self.theta = np.clip(self.theta, -5.0, 5.0)

        loss = float(np.mean(-R_hat * logps))

        self.episode.clear()
        return loss


__all__ = ["SimpleAdaptiveAgent"]
