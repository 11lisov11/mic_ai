# AO2 Hardening Sweep (v2)

- generated_utc: `2026-03-11T06:33:26.856542+00:00`
- stage1_csv: `C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_ao2_hardening_sweep_smoke0\ao2_hardening\ao2_stage1_rank.csv`
- stage2_csv: `C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_ao2_hardening_sweep_smoke0\ao2_hardening\ao2_stage2_rank.csv`
- shortlist_count: `2`
- selected_tag: `manual_safe_02`

## Acceptance (selected)
- power_margin_pass (>=0.20%): `False`
- eta_pass (>=0): `False`
- err_pass (<=2): `True`
- start_stop_pass (>=-0.5%): `True`

## Shortlist (top-3)
| rank | tag | avg_power_saving_pct | avg_eta_gain_pct | err_failures | start_stop_power_saving_pct | v2_score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | manual_safe_02 | -1.222 | -1.197 | 1.00 | +0.000 | 80.838 |
| 2 | manual_safe_01 | -1.840 | -1.893 | 1.00 | +0.000 | 119.505 |

## Selected Candidate
```json
{
  "tag": "manual_safe_02",
  "source": "manual",
  "objective": "p_in",
  "speed_tol_rel": 0.1093960318013046,
  "speed_tol_abs": 0.0,
  "omega_min_pu": 0.1,
  "update_steps": 20,
  "dither_amp": 0.0119286168216908,
  "bias_step": 0.006925078632367,
  "bias_max": 0.1369543194809376,
  "shaft_eps": 10.0,
  "reset_decay": 0.98,
  "objective_clip": 10.0,
  "idle_enable": false,
  "idle_omega_pu": 0.0592083072341568,
  "idle_action": -0.5,
  "idle_exit_boost_steps": 10,
  "idle_exit_action": 0.9768517402041176,
  "idle_bias_decay": 0.96,
  "id_ref_alpha": 0.0688172572674108,
  "delta_id_max": 0.0456814310819514,
  "id_ref_gate_speed_tol_rel": 0.12,
  "id_ref_gate_min_scale": 0.15,
  "id_ref_gate_exponent": 1.0327244378662854,
  "avg_power_saving_pct": -1.222226466306231,
  "avg_eta_gain_pct": -1.1966643391838172,
  "err_failures": 1.0,
  "start_stop_power_saving_pct": 0.0,
  "worst_current_peak_ratio": 1.0110747659229338,
  "worst_current_mean_ratio": 1.03778010053708,
  "avg_power_saving_pct_min_seed": -1.222226466306231,
  "avg_eta_gain_pct_min_seed": -1.1966643391838172,
  "err_failures_max_seed": 1.0,
  "start_stop_power_saving_pct_min_seed": 0.0,
  "score": 66.60008077286327,
  "acceptance_pass": false,
  "v2_score": 80.83790563699975
}
```
