# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.517 | +0.358 | +0.000 | -0.018 | 0.20 | 1.00 | +1.959 | +1.571 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.517 | +0.358 | +0.000 | -0.018 | 0.20 | 1.00 | +1.959 | +1.571 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.517`
- avg_power_saving_pct_min: `+0.358`
- avg_eta_gain_pct_mean: `+0.000`
- avg_eta_gain_pct_min: `-0.018`
- err_failures_mean: `0.20`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+1.959`
- start_stop_power_saving_pct_min: `+1.571`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `False`
- acceptance_pass: `False`
