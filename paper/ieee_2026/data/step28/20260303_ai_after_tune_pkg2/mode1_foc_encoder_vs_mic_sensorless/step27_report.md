# Step27 Pipeline Report

- Motors: `air56,al31,ao2`
- Scenarios: `speed_step,ramp,load_step,start_stop`
- Seeds: `101,202,303,404,505`
- Seed perturbation: enabled=`True` level=`0.200`

## AIR56 Acceptance Criteria

- avg_power_saving_pct > `0.5`; avg_eta_gain_pct >= `0.0`; err_failures <= `2.0`; start_stop >= `-0.5`
- Mean pass: `False`
- Worst-case pass: `False`

## PI vs FOC vs MIC (All Motors, All Seeds)

| Controller | Avg Power Saving, % (mean/std/min) | Avg Eta Gain, % (mean/std/min) | Err Failures (mean/max) | Start-stop Saving, % (mean/min) |
|---|---:|---:|---:|---:|
| PI | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| FOC | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| MIC | 0.417/0.800/-0.945 | -0.834/1.107/-2.698 | 1.200/2.000 | 1.297/-3.904 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.183/0.103/0.081 | -0.114/0.034/-0.168 | 1.400/2.000 | 0.461/0.184 |
| al31 | 0.956/1.204/-0.945 | 0.004/0.005/-0.001 | 0.200/1.000 | 3.769/-3.904 |
| ao2 | 0.111/0.147/-0.065 | -2.392/0.162/-2.698 | 2.000/2.000 | -0.339/-0.436 |

## Reproducibility

- table_sha256: `36cde81522427170e79bda77ff94909ec6ada844e702ee5e8cdf582b1f45fa82`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are not fully satisfied; additional targeted tuning is still required.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `-0.834`%.
