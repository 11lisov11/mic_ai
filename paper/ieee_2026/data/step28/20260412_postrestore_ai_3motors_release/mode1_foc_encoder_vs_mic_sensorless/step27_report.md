# Step27 Pipeline Report

- Motors: `air56,al31,ao2`
- Scenarios: `speed_step,ramp,load_step,start_stop`
- Seeds: `101,202,303,404,505`
- Seed perturbation: enabled=`True` level=`0.200`

## AIR56 Acceptance Criteria

- avg_power_saving_pct > `0.5`; avg_eta_gain_pct >= `0.0`; err_failures <= `2.0`; start_stop >= `-0.5`
- Mean pass: `True`
- Worst-case pass: `True`

## PI vs FOC vs MIC (All Motors, All Seeds)

| Controller | Avg Power Saving, % (mean/std/min) | Avg Eta Gain, % (mean/std/min) | Err Failures (mean/max) | Start-stop Saving, % (mean/min) |
|---|---:|---:|---:|---:|
| PI | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| FOC | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| MIC | 0.849/0.367/0.356 | 1.028/1.737/-0.000 | 0.200/1.000 | 1.911/0.000 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 1.024/0.116/0.901 | 0.123/0.019/0.104 | 0.600/1.000 | 1.835/1.528 |
| al31 | 1.048/0.415/0.388 | 0.007/0.006/-0.000 | 0.000/0.000 | 3.897/1.190 |
| ao2 | 0.476/0.089/0.356 | 2.955/1.865/0.850 | 0.000/0.000 | 0.000/0.000 |

## Reproducibility

- table_sha256: `32e7a3d7a5fe8d340d92552f96b8b8486edfc184b6997e46ffcc20fa4167f1eb`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `1.028`%.
