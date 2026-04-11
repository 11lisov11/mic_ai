from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


def _sanitize_np_array(values: np.ndarray, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)


def _sanitize_tensor(
    tensor: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> torch.Tensor:
    out = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    if min_value is not None or max_value is not None:
        out = torch.clamp(
            out,
            min=min_value if min_value is not None else None,
            max=max_value if max_value is not None else None,
        )
    return out


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
        self.last_anchor_loss: float = 0.0
        self.actor_anchor_net: ActorCritic | None = None
        self.actor_anchor_coef: float = 0.0

    def set_action_std(self, std: float) -> None:
        self.action_std_override = float(std)

    def set_actor_anchor_from_current(self, coef: float) -> None:
        coef = float(coef)
        self.actor_anchor_coef = max(0.0, coef)
        self.last_anchor_loss = 0.0
        if self.actor_anchor_coef <= 0.0:
            self.actor_anchor_net = None
            return
        self.actor_anchor_net = copy.deepcopy(self.net).to(self.device)
        self.actor_anchor_net.eval()
        for param in self.actor_anchor_net.parameters():
            param.requires_grad_(False)

    def _to_tensor(self, obs: Dict[str, float]) -> torch.Tensor:
        arr = _sanitize_np_array([obs.get(k, 0.0) for k in self.feature_keys])
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

    def act(self, obs: Dict[str, float]) -> Tuple[np.ndarray, float, float]:
        self.total_steps += 1
        state_t = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.net(state_t)
            mu = _sanitize_tensor(mu, nan=0.0, posinf=1.0, neginf=-1.0, min_value=-1.0, max_value=1.0)
            std = _sanitize_tensor(std, nan=0.1, posinf=10.0, neginf=1e-6, min_value=1e-6, max_value=10.0)
            value = _sanitize_tensor(value, nan=0.0, posinf=0.0, neginf=0.0)
        if self.action_std_override is not None:
            std = torch.ones_like(std) * self.action_std_override
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -1.0, 1.0)
        action_np = _sanitize_np_array(action.squeeze(0).cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
        logprob_val = float(np.nan_to_num(float(logprob.item()), nan=0.0, posinf=0.0, neginf=0.0))
        value_val = float(np.nan_to_num(float(value.item()), nan=0.0, posinf=0.0, neginf=0.0))
        return action_np.astype(np.float32), logprob_val, value_val

    def store(self, state: Dict[str, float], action: np.ndarray, logprob: float, reward: float, done: bool, value: float) -> None:
        state_arr = _sanitize_np_array([state.get(k, 0.0) for k in self.feature_keys])
        action_arr = _sanitize_np_array(action, nan=0.0, posinf=1.0, neginf=-1.0)
        self.buffer.append(
            Transition(
                state=state_arr,
                action=action_arr,
                logprob=float(np.nan_to_num(float(logprob), nan=0.0, posinf=0.0, neginf=0.0)),
                reward=float(np.nan_to_num(float(reward), nan=0.0, posinf=0.0, neginf=0.0)),
                done=float(np.nan_to_num(float(done), nan=0.0, posinf=1.0, neginf=0.0)),
                value=float(np.nan_to_num(float(value), nan=0.0, posinf=0.0, neginf=0.0)),
            )
        )

    def _compute_returns_advantages(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = [float(np.nan_to_num(tr.reward, nan=0.0, posinf=0.0, neginf=0.0)) for tr in self.buffer]
        dones = [float(np.nan_to_num(tr.done, nan=0.0, posinf=1.0, neginf=0.0)) for tr in self.buffer]
        values = [float(np.nan_to_num(tr.value, nan=0.0, posinf=0.0, neginf=0.0)) for tr in self.buffer] + [
            float(np.nan_to_num(last_value, nan=0.0, posinf=0.0, neginf=0.0))
        ]
        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return _sanitize_np_array(returns), _sanitize_np_array(advantages)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        if not self.buffer:
            return {
                "actor_loss": self.last_actor_loss,
                "value_loss": self.last_value_loss,
                "anchor_loss": self.last_anchor_loss,
            }

        states = _sanitize_np_array(np.stack([tr.state for tr in self.buffer], axis=0))
        actions = _sanitize_np_array(np.stack([tr.action for tr in self.buffer], axis=0), nan=0.0, posinf=1.0, neginf=-1.0)
        old_logprobs = _sanitize_np_array([tr.logprob for tr in self.buffer])
        returns, advantages = self._compute_returns_advantages(last_value=last_value)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = _sanitize_np_array((advantages - adv_mean) / adv_std)

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
                mu = _sanitize_tensor(mu, nan=0.0, posinf=1.0, neginf=-1.0, min_value=-1.0, max_value=1.0)
                std = _sanitize_tensor(std, nan=0.1, posinf=10.0, neginf=1e-6, min_value=1e-6, max_value=10.0)
                values = _sanitize_tensor(values, nan=0.0, posinf=0.0, neginf=0.0)
                dist = torch.distributions.Normal(mu, std)
                logprob = _sanitize_tensor(dist.log_prob(a_t).sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
                entropy = _sanitize_tensor(dist.entropy().sum(dim=-1).mean(), nan=0.0, posinf=0.0, neginf=0.0)

                ratio = _sanitize_tensor(torch.exp(logprob - old_log), nan=1.0, posinf=1.0, neginf=1.0)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = _sanitize_tensor(-(torch.min(ratio * adv_t, clipped_ratio * adv_t)).mean(), nan=0.0, posinf=0.0, neginf=0.0)

                value_loss = _sanitize_tensor(nn.functional.mse_loss(values, ret_t), nan=0.0, posinf=0.0, neginf=0.0)
                if self.actor_anchor_net is not None and self.actor_anchor_coef > 0.0:
                    with torch.no_grad():
                        anchor_mu, _, _ = self.actor_anchor_net(s_t)
                        anchor_mu = _sanitize_tensor(anchor_mu, nan=0.0, posinf=1.0, neginf=-1.0, min_value=-1.0, max_value=1.0)
                    anchor_loss = _sanitize_tensor(nn.functional.mse_loss(mu, anchor_mu), nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    anchor_loss = torch.zeros((), device=self.device)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.actor_anchor_coef * anchor_loss
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                self.last_actor_loss = float(policy_loss.detach().cpu())
                self.last_value_loss = float(value_loss.detach().cpu())
                self.last_anchor_loss = float(anchor_loss.detach().cpu())

        self.buffer.clear()
        return {
            "actor_loss": self.last_actor_loss,
            "value_loss": self.last_value_loss,
            "anchor_loss": self.last_anchor_loss,
        }


__all__ = ["PPOVoltageAgent"]
