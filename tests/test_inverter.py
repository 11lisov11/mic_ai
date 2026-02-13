import math

from config.env import InverterParams
from models.inverter_ideal import IdealInverter


def test_inverter_nonideal_drop() -> None:
    params = InverterParams(Vdc=300.0, f_pwm=10_000.0, r_out=0.2, dead_time=2e-6, v_drop=1.0)
    inv = IdealInverter(params)
    v_abc, (v_d, v_q) = inv.output(120.0, 0.0, 0.0, i_abc=(2.0, -1.0, 0.5))
    # Expect some drop versus ideal command.
    assert abs(v_abc[0]) < 120.0
    assert math.isfinite(v_d)
    assert math.isfinite(v_q)


def test_inverter_saturation() -> None:
    params = InverterParams(Vdc=100.0, f_pwm=10_000.0)
    inv = IdealInverter(params)
    v_abc, (v_d, v_q) = inv.output(200.0, 0.0, 0.0)
    v_max = params.Vdc / math.sqrt(3.0)
    assert abs(v_d) <= v_max + 1e-6
