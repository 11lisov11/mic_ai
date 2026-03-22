# AL31 tuning report (from frozen step28)

| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode1_foc_encoder_vs_mic_sensorless | -13.891 | -15.648 | +0.050 | +0.041 | 4.00 | 4.00 | -55.818 | -63.026 |
| mode2_foc_sensorless_vs_mic_sensorless | -13.891 | -15.648 | +0.050 | +0.041 | 4.00 | 4.00 | -55.818 | -63.026 |

## Worst-case across modes
- avg_power_saving_pct_mean: `-13.891`
- avg_power_saving_pct_min: `-15.648`
- avg_eta_gain_pct_mean: `+0.050`
- avg_eta_gain_pct_min: `+0.041`
- err_failures_mean: `4.00`
- err_failures_max: `4.00`
- start_stop_power_saving_pct_mean: `-55.818`
- start_stop_power_saving_pct_min: `-63.026`

## Acceptance
- mean_pass: `False`
- worst_case_pass: `False`
- acceptance_pass: `False`
