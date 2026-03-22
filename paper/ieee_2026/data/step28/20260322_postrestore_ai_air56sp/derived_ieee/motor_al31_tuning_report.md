# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | +1.499 | +0.111 | -0.003 | -0.012 | 0.00 | 0.00 | +5.797 | +0.100 |
| mode2_foc_sensorless_vs_mic_sensorless | +1.499 | +0.111 | -0.003 | -0.012 | 0.00 | 0.00 | +5.797 | +0.100 |

## Worst-case across modes
- avg_power_saving_pct_mean: `+1.499`
- avg_power_saving_pct_min: `+0.111`
- avg_eta_gain_pct_mean: `-0.003`
- avg_eta_gain_pct_min: `-0.012`
- err_failures_mean: `0.00`
- err_failures_max: `0.00`
- start_stop_power_saving_pct_mean: `+5.797`
- start_stop_power_saving_pct_min: `+0.100`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
