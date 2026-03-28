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
| MIC | 0.356/0.672/-1.510 | 0.107/0.207/-0.031 | 0.267/1.000 | 1.391/-6.013 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.571/0.215/0.246 | 0.002/0.026/-0.031 | 0.200/1.000 | 2.330/1.163 |
| al31 | 0.330/1.103/-1.510 | 0.001/0.009/-0.007 | 0.600/1.000 | 1.148/-6.013 |
| ao2 | 0.168/0.095/0.018 | 0.319/0.247/0.051 | 0.000/0.000 | 0.694/-0.157 |

## Reproducibility

- table_sha256: `223f3115313d732b004a54b381accadd33837405288358432257e2389a2610b1`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `0.107`%.
