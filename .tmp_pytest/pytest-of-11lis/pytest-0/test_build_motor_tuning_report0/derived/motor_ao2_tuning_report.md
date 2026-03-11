# AO2 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.100 | +0.100 | +0.010 | +0.010 | 0.00 | 0.00 | -0.010 | -0.010 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.050 | +0.050 | +0.010 | +0.010 | 0.00 | 0.00 | -0.010 | -0.010 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.050`
- avg_power_saving_pct_min: `+0.050`
- avg_eta_gain_pct_mean: `+0.010`
- avg_eta_gain_pct_min: `+0.010`
- err_failures_mean: `0.00`
- err_failures_max: `0.00`
- start_stop_power_saving_pct_mean: `-0.010`
- start_stop_power_saving_pct_min: `-0.010`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `True`
- acceptance_pass: `True`
