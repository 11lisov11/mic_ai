# PROJECT MASTER EXECUTION LOG (2026-03-03, cycle 2)

## Done in this cycle
- AIR56 checkpoint sweep performed (`outputs/progress_air56_checkpoint_screen_20260303`).
- AIR56 fixed-seed acceptance locked with config profile `manual_safe_03` and checkpoint `actor_ep200`.
- AL31 fixed-seed eta-lock applied with profile `manual_safe_01`.
- Full step27 fixed-seed run completed:
  - `outputs/progress_step27_ai_config_locked_20260303/step27_final_pi_vs_foc_vs_mic.csv`
  - `outputs/progress_step27_ai_config_locked_20260303/step27_stats_motor_controller.csv`
  - `outputs/progress_step27_ai_config_locked_20260303/step27_air56_acceptance.json`
- Full step28 protocol run completed (no drift, AI mode):
  - `outputs/progress_step28_ieee_ai_config_locked_nodrift_20260303/`
- IEEE package built:
  - `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/`
- Passport comparison rebuilt:
  - `paper/ieee_2026/data/passport/20260303_config_locked/`

## Key fixed-seed metrics (MIC vs FOC)
From `outputs/progress_step27_ai_config_locked_20260303/step27_stats_motor_controller.csv`:

- AIR56:
  - `avg_power_saving_pct_mean = +0.5763%`
  - `avg_eta_gain_pct_mean = +0.1887%`
  - `err_failures_mean = 2.0`
  - `start_stop_power_saving_pct_mean = -0.2117%`
  - AIR56 acceptance: mean pass + worst-case pass.

- AL31:
  - `avg_power_saving_pct_mean = +2.0846%`
  - `avg_eta_gain_pct_mean = +0.0067%`
  - `err_failures_mean = 0.0`

- AO2:
  - `avg_power_saving_pct_mean = +0.0918%`
  - `avg_eta_gain_pct_mean = +0.0191%`
  - `err_failures_mean = 0.0`

- Global MIC (3 motors):
  - `avg_power_saving_pct_mean = +0.9176%`
  - `avg_eta_gain_pct_mean = +0.0715%`
  - `err_failures_mean = 0.667`

## Remaining risks (open)
- Robustness under seed perturbation (`seed_perturb_level=0.2`) is not locked:
  - `outputs/progress_step28_ieee_ai_config_locked_20260303/` shows regressions for AIR56/AL31.
- AO2 passport run still overflows in some operating points:
  - see `paper/ieee_2026/data/passport/20260303_config_locked/passport_compare_3motors.json`.

## Config/runtime changes applied
- `config/env_research_air56_025kw.py` updated to locked safe profile (`manual_safe_03` parameters).
- `config/env_research_al31_4_06kw.py` updated to locked safe profile (`manual_safe_01` parameters).
- AIR56 runtime checkpoint replaced:
  - `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth`
  - backup: `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor_pre_ep200_20260303.pth`
