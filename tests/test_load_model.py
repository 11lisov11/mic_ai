from sim.load_servo import ServoLoadConfig, ServoLoadModel


def test_torque_mode_limits_and_dynamics() -> None:
    config = ServoLoadConfig(mode="torque", tau_load=0.1, t_max=2.0)
    model = ServoLoadModel(config)
    dt = 0.01

    history = []
    for _ in range(10):  # 0.1 s total
        history.append(model.step(dt, omega=0.0, command=10.0))

    assert max(history) <= config.t_max + 1e-12
    t_at_tau = history[-1]
    assert 0.6 * config.t_max <= t_at_tau <= 0.7 * config.t_max

    t_final = None
    for _ in range(200):
        t_final = model.step(dt, omega=0.0, command=10.0)
    assert t_final is not None
    assert t_final <= config.t_max + 1e-12
    assert t_final >= 0.99 * config.t_max


def test_speed_mode_saturates_and_is_positive() -> None:
    config = ServoLoadConfig(
        mode="speed",
        tau_load=0.05,
        t_max=1.5,
        speed_kp=0.02,
        speed_ki=0.2,
        speed_int_limit=10.0,
    )
    model = ServoLoadModel(config)
    dt = 0.01

    t_final = None
    for _ in range(200):
        t_final = model.step(dt, omega=0.0, command=100.0)
    assert t_final is not None
    assert 0.0 <= t_final <= config.t_max + 1e-12
    assert t_final >= 0.9 * config.t_max
