# Step27 Pipeline Report

- Motors: `air56`
- Scenarios: `speed_step`
- Seeds: `101`
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
| MIC | -19.924/0.000/-19.924 | -9.834/0.000/-9.834 | 1.000/1.000 | 0.000/0.000 |

## MIC by Motor (mean/std/min)

| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |
|---|---:|---:|---:|---:|
| air56 | -19.924/0.000/-19.924 | -9.834/0.000/-9.834 | 1.000/1.000 | 0.000/0.000 |

## Reproducibility

- table_sha256: `3b6942b52ed59a2fe1325f21ea94fb73c4f0af3526d3e4b238e010105f2bbf75`
- stable_vs_previous: `None`

## Short Scientific Conclusion

AIR56 mean acceptance constraints are not fully satisfied; additional targeted tuning is still required.
Across the 3-motor benchmark, MIC does not yet exceed sensorless FOC in mean power-saving margin relative to PI.
The observed MIC mean eta-gain relative to PI is `-9.834`%.
