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
| MIC | 0.480/0.084/0.371 | 1.479/1.957/0.005 | 1.333/2.000 | 0.418/-0.347 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.576/0.000/0.576 | 0.189/0.000/0.189 | 2.000/2.000 | -0.212/-0.212 |
| al31 | 0.492/0.000/0.492 | 0.005/0.000/0.005 | 0.000/0.000 | 1.814/1.814 |
| ao2 | 0.371/0.000/0.371 | 4.245/0.000/4.245 | 2.000/2.000 | -0.347/-0.347 |

## Reproducibility

- table_sha256: `c1487a1cc5545d72d35b5f2db4f82e75a578975ea85f0ca42ccef4f26d8b596e`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `1.479`%.
