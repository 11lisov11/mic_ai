from config.env_research_air56_025kw import *

ai_eval_checkpoint_path = r"outputs/air56_round1_subset_ep006_012_20260320/actor_ep008.pth"
ai_eval_id_ref_alpha = 0.181
ai_eval_delta_id_max = 0.106
ai_eval_id_ref_relative = True
ai_eval_id_ref_allow_positive_delta = True
ai_eval_id_ref_gate_speed_tol_rel = 0.121
ai_eval_id_ref_gate_min_scale = 0.126
ai_eval_id_ref_gate_exponent = 0.95

ai_eval_supervisor_enabled = True
ai_eval_sup_objective = "p_in"
ai_eval_sup_speed_tol_rel = 0.0768
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.07740003039871146
ai_eval_sup_update = 18
ai_eval_sup_dither = 0.025
ai_eval_sup_step = 0.0095
ai_eval_sup_bias_max = 0.1454
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.9886974417190302
ai_eval_sup_idle_enable = True
ai_eval_sup_idle_omega_min = 0.0612
ai_eval_sup_idle_action = -0.61
ai_eval_sup_idle_blend = 1.0
ai_eval_sup_idle_exit_boost = 4
ai_eval_sup_idle_exit_action = 0.89
ai_eval_sup_idle_bias_decay = 0.95
