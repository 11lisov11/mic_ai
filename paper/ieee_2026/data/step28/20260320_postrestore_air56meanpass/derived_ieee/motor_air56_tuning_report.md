# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | -28.037 | -39.607 | -10.057 | -14.949 | 3.00 | 3.00 | -15.769 | -16.450 |
| mode2_foc_sensorless_vs_mic_sensorless | -28.037 | -39.607 | -10.057 | -14.949 | 3.00 | 3.00 | -15.769 | -16.450 |

## Worst-case across modes
- avg_power_saving_pct_mean: `-28.037`
- avg_power_saving_pct_min: `-39.607`
- avg_eta_gain_pct_mean: `-10.057`
- avg_eta_gain_pct_min: `-14.949`
- err_failures_mean: `3.00`
- err_failures_max: `3.00`
- start_stop_power_saving_pct_mean: `-15.769`
- start_stop_power_saving_pct_min: `-16.450`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
