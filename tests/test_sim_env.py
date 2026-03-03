import math

from config import env_demo_true_motor1_physical
from models.induction_motor import MotorState
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


def test_sim_env_numeric_guard_for_overflowing_motor_state() -> None:
    env = InductionMotorEnv(env_demo_true_motor1_physical.ENV)
    env.reset()

    def _fake_step(*_args, **_kwargs):
        state = MotorState(
            psi_ds=1e300,
            psi_qs=1e300,
            psi_dr=0.0,
            psi_qr=0.0,
            omega_m=1e300,
        )
        env.motor.state = state
        return state, 1e300, 1e300, 10.0, 1e300

    env.motor.step = _fake_step  # type: ignore[assignment]

    _obs, _r, _done, info = env.step(None)
    assert bool(info.get("invalid_state", False))
    assert math.isfinite(float(info.get("p_mech_loss", 0.0)))
    assert math.isfinite(float(info.get("p_core_loss", 0.0)))
    assert math.isfinite(float(info.get("p_in_total", 0.0)))
