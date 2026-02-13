import math

from config.env import create_default_env
from control.aff_foc import AffFocController
from control.ebs_foc import EbsFocController
from control.esc_foc import EscFocController
from control.hybrid_v3_foc import HybridV3FocController
from control.lmc_foc import LmcFocController
from control.load_map_foc import LoadMapFocController
from control.scalar_vf import ScalarVfController
from control.v3_ternary import V3Controller
from control.vector_foc import FocController


def _basic_step(controller, omega_ref: float = 0.0, load_torque: float | None = None) -> None:
    kwargs = {
        "t": 0.0,
        "omega_ref": omega_ref,
        "omega_m": 0.0,
        "i_abc": (0.0, 0.0, 0.0),
        "torque_e": 0.0,
        "theta_mech": 0.0,
    }
    if load_torque is not None:
        kwargs["load_torque"] = float(load_torque)
    v_d, v_q, theta_e, omega_syn, info = controller.step(**kwargs)
    assert math.isfinite(v_d)
    assert math.isfinite(v_q)
    assert isinstance(info, dict)


def test_control_controllers_smoke() -> None:
    env = create_default_env()
    omega_base = 2.0 * math.pi * env.scalar_vf.f_max / env.motor.p
    dt = env.sim.dt

    _basic_step(FocController(env.foc, env.motor, dt))
    _basic_step(V3Controller(env.foc, env.motor, dt, omega_base))
    _basic_step(AffFocController(env.foc, env.motor, dt))
    _basic_step(LoadMapFocController(env.foc, env.motor, dt))
    _basic_step(LmcFocController(env.foc, env.motor, dt, omega_base))
    _basic_step(EbsFocController(env.foc, env.motor, dt, omega_base))
    _basic_step(EscFocController(env.foc, env.motor, dt, omega_base))
    _basic_step(HybridV3FocController(env.foc, env.motor, dt, omega_base, env.sim.load_torque), load_torque=env.sim.load_torque)

    scalar = ScalarVfController(env.scalar_vf, dt, env.motor.p, env.inverter.Vdc)
    _basic_step(scalar)
