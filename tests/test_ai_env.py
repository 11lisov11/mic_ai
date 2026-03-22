import math

from config import env_demo_true_motor1
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
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
