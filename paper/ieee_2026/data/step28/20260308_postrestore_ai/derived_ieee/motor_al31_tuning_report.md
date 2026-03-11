# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.134 | -0.486 | -0.031 | -0.032 | 3.00 | 3.00 | +0.600 | -1.907 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.134 | -0.486 | -0.031 | -0.032 | 3.00 | 3.00 | +0.600 | -1.907 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.134`
- avg_power_saving_pct_min: `-0.486`
- avg_eta_gain_pct_mean: `-0.031`
- avg_eta_gain_pct_min: `-0.032`
- err_failures_mean: `3.00`
- err_failures_max: `3.00`
- start_stop_power_saving_pct_mean: `+0.600`
- start_stop_power_saving_pct_min: `-1.907`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
