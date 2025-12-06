from __future__ import annotations

import numpy as np
from typing import Dict


class SimpleCuriosityModule:
    """
    Measures novelty of observations via simple finite differences.
    """

    def __init__(self, weight: float = 0.1):
        self.weight = weight
        self.prev_obs: Dict[str, float] | None = None

    def compute_intrinsic_reward(self, obs: Dict[str, float], obs_next: Dict[str, float]) -> float:
        if self.prev_obs is None:
            self.prev_obs = obs
            return 0.0
        keys = ["i_d", "i_q"]
        diffs = []
        for k in keys:
            if k in obs and k in obs_next:
                diffs.append(obs_next[k] - obs[k])
        if not diffs:
            self.prev_obs = obs_next
            return 0.0
        value = float(np.linalg.norm(diffs))
        r_int = self.weight * value
        self.prev_obs = obs_next
        return r_int


__all__ = ["SimpleCuriosityModule"]
