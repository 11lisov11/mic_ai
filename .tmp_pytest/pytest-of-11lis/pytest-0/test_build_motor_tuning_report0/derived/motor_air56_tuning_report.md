# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode2_foc_sensorless_vs_mic_sensorless | +0.700 | +0.700 | +0.100 | +0.100 | 1.00 | 2.00 | -0.200 | -0.200 |
| mode1_foc_encoder_vs_mic_sensorless | +0.600 | +0.600 | +0.100 | +0.100 | 1.00 | 2.00 | -0.200 | -0.200 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.600`
- avg_power_saving_pct_min: `+0.600`
- avg_eta_gain_pct_mean: `+0.100`
- avg_eta_gain_pct_min: `+0.100`
- err_failures_mean: `1.00`
- err_failures_max: `2.00`
- start_stop_power_saving_pct_mean: `-0.200`
- start_stop_power_saving_pct_min: `-0.200`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `True`
- acceptance_pass: `True`
