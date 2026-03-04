# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.350 | +0.156 | +0.176 | +0.163 | 2.20 | 3.00 | -0.910 | -1.505 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.350 | +0.156 | +0.176 | +0.163 | 2.20 | 3.00 | -0.910 | -1.505 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.350`
- avg_power_saving_pct_min: `+0.156`
- avg_eta_gain_pct_mean: `+0.176`
- avg_eta_gain_pct_min: `+0.163`
- err_failures_mean: `2.20`
- err_failures_max: `3.00`
- start_stop_power_saving_pct_mean: `-0.910`
- start_stop_power_saving_pct_min: `-1.505`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
