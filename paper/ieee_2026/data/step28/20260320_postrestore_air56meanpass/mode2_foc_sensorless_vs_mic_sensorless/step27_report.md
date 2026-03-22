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
| MIC | -123.833/152.112/-442.597 | -24.003/28.585/-78.999 | 3.600/4.000 | -66.943/-377.143 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | -28.037/13.574/-39.607 | -10.057/4.757/-14.949 | 3.000/3.000 | -15.769/-16.450 |
| al31 | -13.891/1.559/-15.648 | 0.050/0.008/0.041 | 4.000/4.000 | -55.818/-63.026 |
| ao2 | -329.571/75.069/-442.597 | -62.003/14.545/-78.999 | 3.800/4.000 | -129.243/-377.143 |

## Reproducibility

- table_sha256: `87856fd588767c1607c51988808dacbfa205b6d83b3c1ae6131064bc93da66a4`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are not fully satisfied; additional targeted tuning is still required.
Across the 3-motor benchmark, MIC does not yet exceed sensorless FOC in mean power-saving margin relative to PI.
The observed MIC mean eta-gain relative to PI is `-24.003`%.
