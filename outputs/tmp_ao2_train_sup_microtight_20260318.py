from config.env_research_ao2_32_4_3kw import *  # noqa: F401,F403

# Temporary aligned training supervisor for AO2 continuation research.
ai_eval_supervisor_enabled = True
ai_eval_sup_objective = "p_in"
ai_eval_sup_speed_tol_rel = 0.1225
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.1
ai_eval_sup_update = 26
ai_eval_sup_dither = 0.0108
ai_eval_sup_step = 0.0051
ai_eval_sup_bias_max = 0.1415
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.98
ai_eval_sup_idle_enable = False
ai_eval_sup_idle_omega_min = 0.05
ai_eval_sup_idle_action = -0.25
ai_eval_sup_idle_blend = 0.10
ai_eval_sup_idle_exit_boost = 0
ai_eval_sup_idle_exit_action = 0.90
ai_eval_sup_idle_bias_decay = 0.98
