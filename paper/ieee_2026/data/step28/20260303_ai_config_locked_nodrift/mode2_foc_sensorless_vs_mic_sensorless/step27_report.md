# Step27 Pipeline Report

- Motors: `air56,al31,ao2`
- Scenarios: `speed_step,ramp,load_step,start_stop`
- Seeds: `101,202,303,404,505`
- Seed perturbation: enabled=`False` level=`0.000`

## AIR56 Acceptance Criteria

- avg_power_saving_pct > `0.5`; avg_eta_gain_pct >= `0.0`; err_failures <= `2.0`; start_stop >= `-0.5`
- Mean pass: `True`
- Worst-case pass: `True`

## PI vs FOC vs MIC (All Motors, All Seeds)

| Controller | Avg Power Saving, % (mean/std/min) | Avg Eta Gain, % (mean/std/min) | Err Failures (mean/max) | Start-stop Saving, % (mean/min) |
|---|---:|---:|---:|---:|
| PI | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| FOC | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| MIC | 0.918/0.849/0.092 | 0.072/0.083/0.007 | 0.667/2.000 | 2.670/-0.212 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.576/0.000/0.576 | 0.189/0.000/0.189 | 2.000/2.000 | -0.212/-0.212 |
| al31 | 2.085/0.000/2.085 | 0.007/0.000/0.007 | 0.000/0.000 | 8.235/8.235 |
| ao2 | 0.092/0.000/0.092 | 0.019/0.000/0.019 | 0.000/0.000 | -0.014/-0.014 |

## Reproducibility

- table_sha256: `e74b3a5a14cba32dec8f3f7d17413a1df57dba17f5d9371d3968c3040ba90154`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `0.072`%.
