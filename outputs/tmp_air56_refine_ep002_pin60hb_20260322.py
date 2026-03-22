from config.env_research_air56_025kw import *  # noqa: F401,F403

# Targeted AIR56 refinement around actor_ep002 + pin60_hard_b.
ai_eval_sup_objective = "p_in"
ai_eval_sup_speed_tol_rel = 0.0755
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.0608
ai_eval_sup_update = 18
ai_eval_sup_dither = 0.0254
ai_eval_sup_step = 0.0104
ai_eval_sup_bias_max = 0.1446
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.9886974417190302
ai_eval_sup_idle_enable = True
ai_eval_sup_idle_omega_min = 0.0608
ai_eval_sup_idle_action = -0.604
ai_eval_sup_idle_blend = 1.0
ai_eval_sup_idle_exit_boost = 4
ai_eval_sup_idle_exit_action = 0.887
ai_eval_sup_idle_bias_decay = 0.952
ai_eval_id_ref_alpha = 0.190
ai_eval_delta_id_max = 0.116
ai_eval_id_ref_gate_speed_tol_rel = 0.128
ai_eval_id_ref_gate_min_scale = 0.118
ai_eval_id_ref_gate_exponent = 0.962
