# Theory Validation Report

- csv_path: `C:\mic_theory\paper\pgups_2026\fig\working_characteristics_air56_foc_mic_table.csv`
- passed: `True`
- hard_fail_count: `0`
- warn_fail_count: `0`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=15 |
| FOC:eta_bounds | error | True | eta_min=0.000, eta_max=79.764 |
| FOC:cosphi_bounds | error | True | cos_min=0.1676, cos_max=0.8117 |
| FOC:m2_monotonic | error | True | violations=0 |
| FOC:i1_monotonic | warn | True | violations=0 |
| FOC:n2_non_increasing | warn | True | upward_jumps_gt3rpm=0 |
| FOC:eta_peak_location | warn | True | peak_rel=1.000 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.5690 |
| MIC_AI:finite_rows | error | True | finite_rows=15 |
| MIC_AI:eta_bounds | error | True | eta_min=0.000, eta_max=79.953 |
| MIC_AI:cosphi_bounds | error | True | cos_min=0.1672, cos_max=0.8297 |
| MIC_AI:m2_monotonic | error | True | violations=0 |
| MIC_AI:i1_monotonic | warn | True | violations=0 |
| MIC_AI:n2_non_increasing | warn | True | upward_jumps_gt3rpm=0 |
| MIC_AI:eta_peak_location | warn | True | peak_rel=1.000 |
| MIC_AI:cosphi_low_to_mid_rise | warn | True | delta=0.5684 |
