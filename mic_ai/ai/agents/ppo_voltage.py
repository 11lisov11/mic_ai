from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


def _mlp(in_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.Tanh())
        last = h
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.actor_body = _mlp(state_dim, hidden_sizes)
        self.actor_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic_body = _mlp(state_dim, hidden_sizes)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_feat = self.actor_body(state)
        mu = torch.tanh(self.actor_head(actor_feat))
        std = torch.exp(self.log_std)
        critic_feat = self.critic_body(state)
        value = self.critic_head(critic_feat).squeeze(-1)
        return mu, std, value


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    logprob: float
    reward: float
    done: float
    value: float


class PPOVoltageAgent:
    def __init__(
        self,
        feature_keys: Iterable[str],
        action_dim: int = 2,
        device: str = "cpu",
        hidden_sizes: Tuple[int, ...] = (128, 128),
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.003,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_epochs: int = 5,
        minibatch_frac: float = 1.0,
    ):
        self.feature_keys = list(feature_keys)
        self.state_dim = len(self.feature_keys)
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.train_epochs = train_epochs
        self.minibatch_frac = minibatch_frac

        self.net = ActorCritic(self.state_dim, self.action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.buffer: List[Transition] = []
        self.action_std_override: float | None = None
        self.total_steps = 0
        self.last_actor_loss: float = 0.0
        self.last_value_loss: float = 0.0

    def set_action_std(self, std: float) -> None:
        self.action_std_override = float(std)

    def _to_tensor(self, obs: Dict[str, float]) -> torch.Tensor:
        arr = np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

    def act(self, obs: Dict[str, float]) -> Tuple[np.ndarray, float, float]:
        self.total_steps += 1
        state_t = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.net(state_t)
        if self.action_std_override is not None:
            std = torch.ones_like(std) * self.action_std_override
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy().astype(np.float32), float(logprob.item()), float(value.item())

    def store(self, state: Dict[str, float], action: np.ndarray, logprob: float, reward: float, done: bool, value: float) -> None:
        state_arr = np.asarray([state.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.buffer.append(
            Transition(
                state=state_arr,
                action=np.asarray(action, dtype=np.float32),
                logprob=float(logprob),
                reward=float(reward),
                done=float(done),
                value=float(value),
            )
        )

    def _compute_returns_advantages(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = [tr.reward for tr in self.buffer]
        dones = [tr.done for tr in self.buffer]
        values = [tr.value for tr in self.buffer] + [last_value]
        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        if not self.buffer:
            return {"actor_loss": self.last_actor_loss, "value_loss": self.last_value_loss}

        states = np.stack([tr.state for tr in self.buffer], axis=0)
        actions = np.stack([tr.action for tr in self.buffer], axis=0)
        old_logprobs = np.array([tr.logprob for tr in self.buffer], dtype=np.float32)
        returns, advantages = self._compute_returns_advantages(last_value=last_value)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        dataset_size = len(self.buffer)
        minibatch_size = max(1, int(dataset_size * self.minibatch_frac))

        for _ in range(self.train_epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, minibatch_size):
                batch_idx = idx[start : start + minibatch_size]
                s_t = torch.as_tensor(states[batch_idx], device=self.device)
                a_t = torch.as_tensor(actions[batch_idx], device=self.device)
                old_log = torch.as_tensor(old_logprobs[batch_idx], device=self.device)
                ret_t = torch.as_tensor(returns[batch_idx], device=self.device)
                adv_t = torch.as_tensor(advantages[batch_idx], device=self.device)

                mu, std, values = self.net(s_t)
                dist = torch.distributions.Normal(mu, std)
                logprob = dist.log_prob(a_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logprob - old_log)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -(torch.min(ratio * adv_t, clipped_ratio * adv_t)).mean()

                value_loss = nn.functional.mse_loss(values, ret_t)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                self.last_actor_loss = float(policy_loss.detach().cpu())
                self.last_value_loss = float(value_loss.detach().cpu())

        self.buffer.clear()
        return {"actor_loss": self.last_actor_loss, "value_loss": self.last_value_loss}


__all__ = ["PPOVoltageAgent"]
