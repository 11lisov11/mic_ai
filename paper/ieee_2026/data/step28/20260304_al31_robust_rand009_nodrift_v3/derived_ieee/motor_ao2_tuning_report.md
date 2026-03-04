# AO2 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.371 | +0.371 | +4.245 | +4.245 | 2.00 | 2.00 | -0.347 | -0.347 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.371 | +0.371 | +4.245 | +4.245 | 2.00 | 2.00 | -0.347 | -0.347 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.371`
- avg_power_saving_pct_min: `+0.371`
- avg_eta_gain_pct_mean: `+4.245`
- avg_eta_gain_pct_min: `+4.245`
- err_failures_mean: `2.00`
- err_failures_max: `2.00`
- start_stop_power_saving_pct_mean: `-0.347`
- start_stop_power_saving_pct_min: `-0.347`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `True`
- acceptance_pass: `True`
