import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import env_demo_true_motor1
from mic_ai.ai.ai_env import (
    AiEnvConfig,
    MicAiAIEnv,
    compute_safe_core_loss,
    compute_safe_l2_norm,
    compute_safe_obs_raw,
)
from simulation.gym_env import InductionMotorEnv


def test_ai_env_id_ref_step() -> None:
    base_env = InductionMotorEnv(env_demo_true_motor1.ENV)
    ai_cfg = AiEnvConfig(
        episode_steps=5,
        dt=env_demo_true_motor1.ENV.sim.dt,
        omega_ref=2.0,
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode="ai_id_ref",
    )
    env = MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None)
    obs = env.reset()
    obs_next, reward, done, info = env.step(0.0)
    assert isinstance(obs_next, dict)
    assert math.isfinite(float(reward))
    assert "id_ref_cmd" in info
    assert "p_in" in info
    assert "eta_episode_norm" in obs_next


def test_ai_env_voltage_uses_total_power() -> None:
    base_env = InductionMotorEnv(env_demo_true_motor1.ENV)
    ai_cfg = AiEnvConfig(
        episode_steps=5,
        dt=env_demo_true_motor1.ENV.sim.dt,
        omega_ref=2.0,
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode="ai_voltage",
    )
    env = MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None)
    env.reset()
    _obs_next, _reward, _done, info = env.step([0.0, 0.0])
    assert "p_in_total" in info
    assert abs(float(info["p_in"]) - float(info["p_in_total"])) < 1e-6


def test_compute_safe_core_loss_clamps_overflow() -> None:
    loss = compute_safe_core_loss(
        loss_core_k=1.0,
        omega_core=1e308,
        psi_s=1e308,
        loss_core_omega_exp=2.0,
        loss_core_psi_exp=2.0,
    )
    assert math.isfinite(loss)
    assert loss == 1e12


def test_compute_safe_l2_norm_clamps_overflow() -> None:
    norm = compute_safe_l2_norm(1e308, -1e308)
    assert math.isfinite(norm)
    assert norm == pytest.approx(math.sqrt(2.0) * 1e6)


def test_compute_safe_obs_raw_clamps_overflow() -> None:
    obs_raw = compute_safe_obs_raw(1e308, float("inf"), float("-inf"), float("nan"))
    assert obs_raw.dtype == np.float32
    assert np.isfinite(obs_raw).all()
    assert obs_raw.tolist() == [1e6, 1e6, -1e6, 0.0]


def test_current_rms_uses_safe_l2_path() -> None:
    base_env = InductionMotorEnv(env_demo_true_motor1.ENV)
    ai_cfg = AiEnvConfig(
        episode_steps=5,
        dt=env_demo_true_motor1.ENV.sim.dt,
        omega_ref=2.0,
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode="ai_id_ref",
    )
    env = MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None)
    current_rms = env._current_rms((1e308, -1e308, 1e308))
    assert math.isfinite(current_rms)
    assert current_rms == pytest.approx(math.sqrt(3.0) * 1e6 / math.sqrt(3.0))
