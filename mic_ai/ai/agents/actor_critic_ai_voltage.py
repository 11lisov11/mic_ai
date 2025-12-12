from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.Tanh())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ActorCriticVoltageAgent:
    def __init__(
        self,
        feature_keys: Iterable[str],
        action_dim: int = 2,
        hidden_sizes: Sequence[int] | None = (64, 64),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        sigma: float = 0.15,
        max_grad_norm: float = 5.0,
        clip_eps: float = 0.2,
        entropy_coef: float = 1e-3,
        ppo_epochs: int = 6,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.feature_keys = list(feature_keys)
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes else [64, 64]
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = max(int(ppo_epochs), 1)
        self.batch_size = max(int(batch_size), 1)
        self.device = torch.device(device)

        obs_dim = len(self.feature_keys)
        self.actor = _mlp(obs_dim, self.hidden_sizes, self.action_dim).to(self.device)
        self.critic = _mlp(obs_dim, self.hidden_sizes, 1).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.action_std = float(sigma)
        self.buffer: List[Dict[str, torch.Tensor]] = []

    def _features(self, obs: Dict[str, float]) -> torch.Tensor:
        arr = np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def start_episode(self) -> None:
        self.buffer.clear()

    def act(self, obs: Dict[str, float]) -> np.ndarray:
        x = self._features(obs).unsqueeze(0)
        with torch.no_grad():
            mu = self.actor(x)
            std = torch.full_like(mu, self.action_std)
            dist = Normal(mu, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            logprob = dist.log_prob(raw_action) - torch.log1p(-action.pow(2) + 1e-6)
            logprob = logprob.sum(dim=-1)
            value = self.critic(x).squeeze(-1)

        self.buffer.append(
            {
                "obs": x.squeeze(0),
                "action": action.squeeze(0),
                "logprob": logprob.squeeze(0),
                "value": value.squeeze(0),
                "reward": torch.tensor(0.0, device=self.device),
            }
        )
        return action.squeeze(0).cpu().numpy()

    def record_reward(self, r: float, next_obs: Dict[str, float] | None = None) -> None:
        if not self.buffer:
            return
        self.buffer[-1]["reward"] = torch.tensor(float(r), dtype=torch.float32, device=self.device)

    def _compute_returns_and_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.stack([entry["reward"] for entry in self.buffer])
        values = torch.stack([entry["value"] for entry in self.buffer])
        returns = torch.zeros_like(rewards, device=self.device)
        G = torch.tensor(0.0, device=self.device)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.detach(), advantages.detach()

    def update_after_episode(self) -> Dict[str, float]:
        if not self.buffer:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs = torch.stack([entry["obs"] for entry in self.buffer])
        actions = torch.stack([entry["action"] for entry in self.buffer])
        logprobs_old = torch.stack([entry["logprob"] for entry in self.buffer])
        values = torch.stack([entry["value"] for entry in self.buffer]).squeeze(-1)
        returns, advantages = self._compute_returns_and_advantages()

        actor_losses: List[float] = []
        critic_losses: List[float] = []
        batch_size = max(self.batch_size, 1)

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(len(self.buffer))
            for start in range(0, len(self.buffer), batch_size):
                batch_idx = idx[start : start + batch_size]
                obs_b = obs[batch_idx]
                actions_b = actions[batch_idx]
                returns_b = returns[batch_idx]
                adv_b = advantages[batch_idx]
                old_logprob_b = logprobs_old[batch_idx]

                mu = self.actor(obs_b)
                std = torch.full_like(mu, self.action_std)
                dist = Normal(mu, std)
                raw_action = torch.atanh(actions_b.clamp(-0.999, 0.999))
                logprob = dist.log_prob(raw_action) - torch.log1p(-actions_b.pow(2) + 1e-6)
                logprob = logprob.sum(dim=-1)

                ratio = torch.exp(logprob - old_logprob_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss_actor = policy_loss - self.entropy_coef * entropy

                value_pred = self.critic(obs_b).squeeze(-1)
                loss_critic = nn.functional.mse_loss(value_pred, returns_b)

                self.actor_opt.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(float(loss_actor.detach().cpu()))
                critic_losses.append(float(loss_critic.detach().cpu()))

        self.buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
        }
