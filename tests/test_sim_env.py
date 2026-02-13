import math

from config import env_demo_true_motor1_physical
from simulation.gym_env import InductionMotorEnv


def test_sim_env_step_keys() -> None:
    env = InductionMotorEnv(env_demo_true_motor1_physical.ENV)
    obs = env.reset()
    obs, _r, done, info = env.step(None)
    assert "p_in_total" in info
    assert "p_inv_loss" in info
    assert "p_core_loss" in info
    assert math.isfinite(info.get("p_in_total", 0.0))
    assert len(obs) == 8
