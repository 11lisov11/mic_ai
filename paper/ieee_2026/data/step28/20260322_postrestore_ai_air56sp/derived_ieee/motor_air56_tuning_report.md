# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.551 | +0.493 | -0.006 | -0.029 | 0.40 | 1.00 | +2.130 | +1.750 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.551 | +0.493 | -0.006 | -0.029 | 0.40 | 1.00 | +2.130 | +1.750 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.551`
- avg_power_saving_pct_min: `+0.493`
- avg_eta_gain_pct_mean: `-0.006`
- avg_eta_gain_pct_min: `-0.029`
- err_failures_mean: `0.40`
- err_failures_max: `1.00`
- start_stop_power_saving_pct_mean: `+2.130`
- start_stop_power_saving_pct_min: `+1.750`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
