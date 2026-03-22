# AO2 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | -329.571 | -442.597 | -62.003 | -78.999 | 3.80 | 4.00 | -129.243 | -377.143 |
| mode2_foc_sensorless_vs_mic_sensorless | -329.571 | -442.597 | -62.003 | -78.999 | 3.80 | 4.00 | -129.243 | -377.143 |

## Worst-case across modes
- avg_power_saving_pct_mean: `-329.571`
- avg_power_saving_pct_min: `-442.597`
- avg_eta_gain_pct_mean: `-62.003`
- avg_eta_gain_pct_min: `-78.999`
- err_failures_mean: `3.80`
- err_failures_max: `4.00`
- start_stop_power_saving_pct_mean: `-129.243`
- start_stop_power_saving_pct_min: `-377.143`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
