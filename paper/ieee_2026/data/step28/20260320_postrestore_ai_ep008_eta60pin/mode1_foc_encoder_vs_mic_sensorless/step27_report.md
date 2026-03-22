# Step27 Pipeline Report

- Motors: `air56,al31,ao2`
- Scenarios: `speed_step,ramp,load_step,start_stop`
- Seeds: `101,202,303,404,505`
- Seed perturbation: enabled=`True` level=`0.200`

## AIR56 Acceptance Criteria

- avg_power_saving_pct > `0.5`; avg_eta_gain_pct >= `0.0`; err_failures <= `2.0`; start_stop >= `-0.5`
- Mean pass: `True`
- Worst-case pass: `False`

## PI vs FOC vs MIC (All Motors, All Seeds)

| Controller | Avg Power Saving, % (mean/std/min) | Avg Eta Gain, % (mean/std/min) | Err Failures (mean/max) | Start-stop Saving, % (mean/min) |
|---|---:|---:|---:|---:|
| PI | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| FOC | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| MIC | 0.728/0.924/0.018 | 0.105/0.208/-0.018 | 0.067/1.000 | 2.817/-0.157 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.517/0.108/0.358 | 0.000/0.022/-0.018 | 0.200/1.000 | 1.959/1.571 |
| al31 | 1.499/1.261/0.111 | -0.003/0.009/-0.012 | 0.000/0.000 | 5.797/0.100 |
| ao2 | 0.168/0.095/0.018 | 0.319/0.247/0.051 | 0.000/0.000 | 0.694/-0.157 |

## Reproducibility

- table_sha256: `31e9dbb3ef60b9ca40a536aae3bd6616f5bdb336f59bae7a4d9f06e6a3e9188d`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `0.105`%.
