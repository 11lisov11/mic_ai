# Step27 Pipeline Report

- Motors: `air56,al31,ao2`
- Scenarios: `speed_step,ramp,load_step,start_stop`
- Seeds: `101,202,303`
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
| MIC | 2.986/5.855/-0.486 | 5.774/11.270/-0.099 | 2.444/3.000 | 11.246/-1.907 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | 0.304/0.153/0.104 | -0.093/0.004/-0.099 | 1.667/2.000 | 1.148/0.416 |
| al31 | 0.134/0.657/-0.486 | -0.031/0.002/-0.032 | 3.000/3.000 | 0.600/-1.907 |
| ao2 | 8.520/7.512/0.839 | 17.445/13.293/1.757 | 2.667/3.000 | 31.989/1.494 |

## Reproducibility

- table_sha256: `b975f3d7ff145b7ebade3db54edb0b84b8f162c126d6a924c28d4f808860eea2`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are not fully satisfied; additional targeted tuning is still required.
Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI.
The observed MIC mean eta-gain relative to PI is `5.774`%.
