# AO2 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.370 | +0.193 | -2.019 | -2.265 | 2.00 | 2.00 | -0.078 | -0.201 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.370 | +0.193 | -2.019 | -2.265 | 2.00 | 2.00 | -0.078 | -0.201 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.370`
- avg_power_saving_pct_min: `+0.193`
- avg_eta_gain_pct_mean: `-2.019`
- avg_eta_gain_pct_min: `-2.265`
- err_failures_mean: `2.00`
- err_failures_max: `2.00`
- start_stop_power_saving_pct_mean: `-0.078`
- start_stop_power_saving_pct_min: `-0.201`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
