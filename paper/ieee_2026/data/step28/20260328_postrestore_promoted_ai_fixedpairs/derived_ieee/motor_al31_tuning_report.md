# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.621 | +0.354 | +0.006 | -0.000 | 0.00 | 0.00 | +2.192 | +1.047 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.621 | +0.354 | +0.006 | -0.000 | 0.00 | 0.00 | +2.192 | +1.047 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.621`
- avg_power_saving_pct_min: `+0.354`
- avg_eta_gain_pct_mean: `+0.006`
- avg_eta_gain_pct_min: `-0.000`
- err_failures_mean: `0.00`
- err_failures_max: `0.00`
- start_stop_power_saving_pct_mean: `+2.192`
- start_stop_power_saving_pct_min: `+1.047`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `False`
- acceptance_pass: `False`
