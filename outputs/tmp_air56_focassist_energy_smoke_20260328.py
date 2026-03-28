from config.env_research_air56_025kw import *  # noqa: F401,F403

# AIR56 foc_assist smoke: keep tracking tight while exposing power-sensitive reward.
foc_assist_reward_mode = "energy"
w_foc_speed = 5.0
w_foc_power = 1.5
w_foc_current = 0.1
w_foc_action = 0.01
foc_speed_tol = 0.5
p_el_tau = 0.02
