# AO2 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +8.520 | +0.839 | +17.445 | +1.757 | 2.67 | 3.00 | +31.989 | +1.494 |
| mode2_foc_sensorless_vs_mic_sensorless | +8.520 | +0.839 | +17.445 | +1.757 | 2.67 | 3.00 | +31.989 | +1.494 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+8.520`
- avg_power_saving_pct_min: `+0.839`
- avg_eta_gain_pct_mean: `+17.445`
- avg_eta_gain_pct_min: `+1.757`
- err_failures_mean: `2.67`
- err_failures_max: `3.00`
- start_stop_power_saving_pct_mean: `+31.989`
- start_stop_power_saving_pct_min: `+1.494`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
