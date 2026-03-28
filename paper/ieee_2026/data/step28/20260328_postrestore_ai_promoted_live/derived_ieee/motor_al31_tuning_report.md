# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.330 | -1.510 | +0.001 | -0.007 | 0.60 | 1.00 | +1.148 | -6.013 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.330 | -1.510 | +0.001 | -0.007 | 0.60 | 1.00 | +1.148 | -6.013 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.330`
- avg_power_saving_pct_min: `-1.510`
- avg_eta_gain_pct_mean: `+0.001`
- avg_eta_gain_pct_min: `-0.007`
- err_failures_mean: `0.60`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+1.148`
- start_stop_power_saving_pct_min: `-6.013`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `False`
- acceptance_pass: `False`
