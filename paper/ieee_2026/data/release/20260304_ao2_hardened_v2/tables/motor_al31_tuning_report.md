# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.049 | -1.554 | -0.002 | -0.008 | 1.00 | 1.00 | +0.115 | -6.172 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.049 | -1.554 | -0.002 | -0.008 | 1.00 | 1.00 | +0.115 | -6.172 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.049`
- avg_power_saving_pct_min: `-1.554`
- avg_eta_gain_pct_mean: `-0.002`
- avg_eta_gain_pct_min: `-0.008`
- err_failures_mean: `1.00`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+0.115`
- start_stop_power_saving_pct_min: `-6.172`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
