from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([state, action], dim=-1))


@dataclass
class ReplaySample:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer: Deque[ReplaySample] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(
            ReplaySample(
                state=np.asarray(state, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=float(done),
            )
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states = np.stack([self.buffer[i].state for i in idx], axis=0)
        actions = np.stack([self.buffer[i].action for i in idx], axis=0)
        rewards = np.array([self.buffer[i].reward for i in idx], dtype=np.float32)[:, None]
        next_states = np.stack([self.buffer[i].next_state for i in idx], axis=0)
        dones = np.array([self.buffer[i].done for i in idx], dtype=np.float32)[:, None]
        return states, actions, rewards, next_states, dones


class TD3VoltageAgent:
    def __init__(
        self,
        feature_keys: Iterable[str],
        action_dim: int = 2,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_size: int = 200_000,
        batch_size: int = 256,
        warmup_steps: int = 2000,
        noise_std: float = 0.05,
        target_noise_std: float = 0.1,
        target_noise_clip: float = 0.3,
        policy_delay: int = 2,
        lr: float = 3e-4,
        max_grad_norm: float = 5.0,
        action_scale: float = 0.7,
    ):
        self.feature_keys = list(feature_keys)
        self.state_dim = len(self.feature_keys)
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.noise_std = noise_std
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm
        self.action_scale = float(max(min(action_scale, 1.0), 0.0))

        self.actor = Actor(self.state_dim, action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(self.state_dim, action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(self.state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
        )

        self.replay = ReplayBuffer(replay_size, self.state_dim, self.action_dim)
        self.total_steps = 0
        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0

    def _to_tensor(self, obs: Dict[str, float]) -> torch.Tensor:
        arr = np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

    def act(self, obs: Dict[str, float], noise: bool = True) -> np.ndarray:
        self.total_steps += 1
        state = self._to_tensor(obs).unsqueeze(0)
        if self.total_steps <= self.warmup_steps:
            action = np.random.uniform(-0.3, 0.3, size=(self.action_dim,))
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().numpy().squeeze(0)
            if noise:
                action = action + np.random.normal(0.0, self.noise_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        action = action * self.action_scale
        action = np.clip(action, -1.0, 1.0)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        return action.astype(np.float32)

    def store(self, obs: Dict[str, float], action: np.ndarray, reward: float, next_obs: Dict[str, float], done: bool) -> None:
        self.replay.add(
            state=np.array([obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32),
            action=action.astype(np.float32),
            reward=reward,
            next_state=np.array([next_obs.get(k, 0.0) for k in self.feature_keys], dtype=np.float32),
            done=done,
        )

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def train_step(self) -> Dict[str, float]:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return {"actor_loss": self.last_actor_loss, "critic_loss": self.last_critic_loss}

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states, device=self.device, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            noise = torch.randn_like(actions_t) * self.target_noise_std
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            next_actions = torch.clamp(self.actor_target(next_states_t) + noise, -1.0, 1.0) * self.action_scale
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            target_q1 = self.critic1_target(next_states_t, next_actions)
            target_q2 = self.critic2_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)
        critic_loss = nn.functional.mse_loss(current_q1, target_value.detach()) + nn.functional.mse_loss(current_q2, target_value.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), self.max_grad_norm)
        self.critic_opt.step()
        self.last_critic_loss = float(critic_loss.detach().cpu())

        # Delayed actor update
        if self.total_steps % self.policy_delay == 0:
            actor_actions = torch.clamp(self.actor(states_t), -1.0, 1.0) * self.action_scale
            actor_actions = torch.clamp(actor_actions, -1.0, 1.0)
            actor_loss = -self.critic1(states_t, actor_actions).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            self.last_actor_loss = float(actor_loss.detach().cpu())

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        return {"actor_loss": self.last_actor_loss, "critic_loss": self.last_critic_loss}


__all__ = ["TD3VoltageAgent"]
