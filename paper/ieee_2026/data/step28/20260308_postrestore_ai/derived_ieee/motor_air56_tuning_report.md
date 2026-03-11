# AIR56 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +0.304 | +0.104 | -0.093 | -0.099 | 1.67 | 2.00 | +1.148 | +0.416 |
| mode2_foc_sensorless_vs_mic_sensorless | +0.304 | +0.104 | -0.093 | -0.099 | 1.67 | 2.00 | +1.148 | +0.416 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+0.304`
- avg_power_saving_pct_min: `+0.104`
- avg_eta_gain_pct_mean: `-0.093`
- avg_eta_gain_pct_min: `-0.099`
- err_failures_mean: `1.67`
- err_failures_max: `2.00`
- start_stop_power_saving_pct_mean: `+1.148`
- start_stop_power_saving_pct_min: `+0.416`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
