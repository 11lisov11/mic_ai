# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +1.024 | +0.901 | +0.123 | +0.104 | 0.60 | 1.00 | +1.835 | +1.528 |
| mode2_foc_sensorless_vs_mic_sensorless | +1.024 | +0.901 | +0.123 | +0.104 | 0.60 | 1.00 | +1.835 | +1.528 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+1.024`
- avg_power_saving_pct_min: `+0.901`
- avg_eta_gain_pct_mean: `+0.123`
- avg_eta_gain_pct_min: `+0.104`
- err_failures_mean: `0.60`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+1.835`
- start_stop_power_saving_pct_min: `+1.528`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `True`
- acceptance_pass: `True`
