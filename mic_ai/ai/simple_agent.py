from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _init_layer(in_dim: int, out_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / np.sqrt(max(in_dim, 1))
    w = np.random.randn(out_dim, in_dim).astype(np.float32) * scale
    b = np.zeros(out_dim, dtype=np.float32)
    return w, b


def _forward_mlp(
    x: np.ndarray,
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    output_activation: str | None = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h = x
    activations = [h]
    for idx, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w.T + b
        last = idx == len(weights) - 1
        if not last:
            h = np.maximum(h, 0.0)
        elif output_activation == "tanh":
            h = np.tanh(h)
        activations.append(h)
    return h, activations


def _backward_mlp(
    activations: List[np.ndarray],
    grad_out: np.ndarray,
    weights: List[np.ndarray],
    output_activation: str | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    grad_w = [np.zeros_like(w) for w in weights]
    grad_b = [np.zeros_like(b) for b in weights]
    grad = grad_out
    for idx in reversed(range(len(weights))):
        last = idx == len(weights) - 1
        if last and output_activation == "tanh":
            grad = grad * (1.0 - activations[idx + 1] ** 2)
        elif not last:
            grad = grad * (activations[idx + 1] > 0).astype(np.float32)
        h_prev = activations[idx]
        grad_w[idx] = grad.T @ h_prev
        grad_b[idx] = grad.sum(axis=0)
        grad = grad @ weights[idx]
    return grad_w, grad_b


class ActorCriticAgent:
    """
    Lightweight on-policy actor-critic (A2C-style) in numpy.
    Actor: obs -> 64 -> 64 -> tanh(action)
    Critic: obs -> 64 -> 64 -> value
    Uses TD error delta = r + gamma * V(s') - V(s) with gradient clipping.
    """

    def __init__(
        self,
        feature_keys: Iterable[str],
        action_dim: int = 1,
        hidden_sizes: Sequence[int] | None = (64, 64),
        lr_actor: float = 7e-4,
        lr_critic: float = 7e-4,
        lr: float | None = None,
        gamma: float = 0.99,
        sigma: float = 0.1,
        max_grad_norm: float = 5.0,
    ):
        if lr is not None:
            lr_actor = lr
            lr_critic = lr
        self.feature_keys = list(feature_keys)
        self.act_dim = max(1, int(action_dim))
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes else [64, 64]
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.sigma = float(sigma)
        self.max_grad_norm = float(max_grad_norm)

        dims_actor = [len(self.feature_keys)] + self.hidden_sizes + [self.act_dim]
        self.actor_w: List[np.ndarray] = []
        self.actor_b: List[np.ndarray] = []
        for i in range(len(dims_actor) - 1):
            w, b = _init_layer(dims_actor[i], dims_actor[i + 1])
            self.actor_w.append(w)
            self.actor_b.append(b)

        dims_critic = [len(self.feature_keys)] + self.hidden_sizes + [1]
        self.critic_w: List[np.ndarray] = []
        self.critic_b: List[np.ndarray] = []
        for i in range(len(dims_critic) - 1):
            w, b = _init_layer(dims_critic[i], dims_critic[i + 1])
            self.critic_w.append(w)
            self.critic_b.append(b)

        self.buffer: List[Dict[str, np.ndarray | float]] = []

    def _features(self, obs: Dict[str, float]) -> np.ndarray:
        return np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)

    def start_episode(self) -> None:
        self.buffer.clear()

    def value(self, obs: Dict[str, float]) -> float:
        x = self._features(obs)[None, :]
        v, _ = _forward_mlp(x, self.critic_w, self.critic_b)
        return float(v.squeeze())

    def act(self, obs: Dict[str, float]) -> float | np.ndarray:
        x = self._features(obs)[None, :]
        mu, acts_pi = _forward_mlp(x, self.actor_w, self.actor_b, output_activation="tanh")
        mu = mu.astype(np.float32).squeeze(0)

        noise = self.sigma * np.random.randn(self.act_dim).astype(np.float32)
        pre_tanh = mu + noise
        action = np.tanh(pre_tanh)

        logprob = -0.5 * float(
            np.sum((noise / (self.sigma + 1e-8)) ** 2 + np.log(2.0 * np.pi * (self.sigma**2 + 1e-8)))
        )

        v_pred, acts_v = _forward_mlp(x, self.critic_w, self.critic_b)

        self.buffer.append(
            {
                "obs": x.squeeze(0),
                "action": action.astype(np.float32),
                "pre_tanh": pre_tanh.astype(np.float32),
                "mu": mu.astype(np.float32),
                "logprob": float(logprob),
                "acts_pi": acts_pi,
                "acts_v": acts_v,
                "value": float(v_pred.squeeze()),
                "reward": 0.0,
                "next_value": 0.0,
            }
        )
        return action if self.act_dim > 1 else float(action[0])

    def record_reward(self, r: float, next_obs: Dict[str, float] | None = None) -> None:
        if not self.buffer:
            return
        entry = self.buffer[-1]
        entry["reward"] = float(r)
        if next_obs is not None:
            entry["next_value"] = float(self.value(next_obs))

    def update_after_episode(self) -> Dict[str, float]:
        if not self.buffer:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        actor_grad_w = [np.zeros_like(w) for w in self.actor_w]
        actor_grad_b = [np.zeros_like(b) for b in self.actor_b]
        critic_grad_w = [np.zeros_like(w) for w in self.critic_w]
        critic_grad_b = [np.zeros_like(b) for b in self.critic_b]

        deltas: List[float] = []
        logprobs: List[float] = []

        for data in self.buffer:
            delta = float(data["reward"] + self.gamma * data.get("next_value", 0.0) - data["value"])
            deltas.append(delta)
            logprobs.append(float(data["logprob"]))

            grad_mu = (-delta) * (data["pre_tanh"] - data["mu"]) / (self.sigma**2 + 1e-8)
            grad_mu = np.asarray(grad_mu, dtype=np.float32)[None, :]
            g_w_pi, g_b_pi = _backward_mlp(data["acts_pi"], grad_mu, self.actor_w, output_activation="tanh")
            for i in range(len(actor_grad_w)):
                actor_grad_w[i] += g_w_pi[i]
                actor_grad_b[i] += g_b_pi[i]

            grad_v = np.array([[-2.0 * delta]], dtype=np.float32)
            g_w_v, g_b_v = _backward_mlp(data["acts_v"], grad_v, self.critic_w, output_activation=None)
            for i in range(len(critic_grad_w)):
                critic_grad_w[i] += g_w_v[i]
                critic_grad_b[i] += g_b_v[i]

        flat = np.concatenate(
            [
                *[g.flatten() for g in actor_grad_w],
                *[g.flatten() for g in actor_grad_b],
                *[g.flatten() for g in critic_grad_w],
                *[g.flatten() for g in critic_grad_b],
            ]
        )
        norm = np.linalg.norm(flat)
        scale = 1.0
        if norm > self.max_grad_norm and norm > 0:
            scale = self.max_grad_norm / norm

        batch = max(len(self.buffer), 1)
        for i in range(len(self.actor_w)):
            self.actor_w[i] -= self.lr_actor * scale * actor_grad_w[i] / batch
            self.actor_b[i] -= self.lr_actor * scale * actor_grad_b[i] / batch
        for i in range(len(self.critic_w)):
            self.critic_w[i] -= self.lr_critic * scale * critic_grad_w[i] / batch
            self.critic_b[i] -= self.lr_critic * scale * critic_grad_b[i] / batch

        actor_loss = -float(np.mean(np.array(logprobs, dtype=np.float32) * np.array(deltas, dtype=np.float32)))
        critic_loss = float(np.mean(np.square(deltas)))
        self.buffer.clear()
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}

    def reset_parameters(self) -> None:
        dims_actor = [len(self.feature_keys)] + self.hidden_sizes + [self.act_dim]
        for i in range(len(dims_actor) - 1):
            w, b = _init_layer(dims_actor[i], dims_actor[i + 1])
            self.actor_w[i] = w
            self.actor_b[i] = b

        dims_critic = [len(self.feature_keys)] + self.hidden_sizes + [1]
        for i in range(len(dims_critic) - 1):
            w, b = _init_layer(dims_critic[i], dims_critic[i + 1])
            self.critic_w[i] = w
            self.critic_b[i] = b


# Backwards-compatibility alias to keep existing imports working.
SimpleAdaptiveAgent = ActorCriticAgent

__all__ = ["ActorCriticAgent", "SimpleAdaptiveAgent"]
