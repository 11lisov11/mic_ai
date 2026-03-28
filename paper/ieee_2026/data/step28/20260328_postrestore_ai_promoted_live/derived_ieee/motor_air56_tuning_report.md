# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.571 | +0.246 | +0.002 | -0.031 | 0.20 | 1.00 | +2.330 | +1.163 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.571 | +0.246 | +0.002 | -0.031 | 0.20 | 1.00 | +2.330 | +1.163 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.571`
- avg_power_saving_pct_min: `+0.246`
- avg_eta_gain_pct_mean: `+0.002`
- avg_eta_gain_pct_min: `-0.031`
- err_failures_mean: `0.20`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+2.330`
- start_stop_power_saving_pct_min: `+1.163`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `False`
- acceptance_pass: `False`
