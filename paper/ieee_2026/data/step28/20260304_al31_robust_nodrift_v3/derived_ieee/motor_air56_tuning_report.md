# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.576 | +0.576 | +0.189 | +0.189 | 2.00 | 2.00 | -0.212 | -0.212 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.576 | +0.576 | +0.189 | +0.189 | 2.00 | 2.00 | -0.212 | -0.212 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.576`
- avg_power_saving_pct_min: `+0.576`
- avg_eta_gain_pct_mean: `+0.189`
- avg_eta_gain_pct_min: `+0.189`
- err_failures_mean: `2.00`
- err_failures_max: `2.00`
- start_stop_power_saving_pct_mean: `-0.212`
- start_stop_power_saving_pct_min: `-0.212`

## Acceptance
- mean_pass: `True`
- worst_case_pass: `True`
- acceptance_pass: `True`
