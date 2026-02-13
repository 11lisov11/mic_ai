from mic_ai.ai.train_ai_voltage import build_env, resolve_config_path


def test_build_env_scenario_override_false() -> None:
    env = build_env(
        resolve_config_path("env_demo_true_motor1"),
        episode_steps=10,
        override_omega_ref=False,
        override_load_torque=False,
    )
    t_end = float(getattr(env.base_env.env.sim, "t_end", 1.0))
    t_step = 0.1 * t_end
    omega0 = float(env.base_env.omega_ref_func(0.0))
    omega1 = float(env.base_env.omega_ref_func(t_step * 1.5))
    assert omega0 != omega1


def test_build_env_scenario_override_true() -> None:
    env = build_env(
        resolve_config_path("env_demo_true_motor1"),
        episode_steps=10,
        override_omega_ref=True,
        override_load_torque=True,
    )
    omega0 = float(env.base_env.omega_ref_func(0.0))
    omega1 = float(env.base_env.omega_ref_func(1.0))
    assert omega0 == omega1
