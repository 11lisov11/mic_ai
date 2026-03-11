# Theory Validation Report: AL31

- csv_path: `C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_build_theory_validation_r0\passport\tag_test\raw\al31\working_characteristics_filtered.csv`
- passed: `True`
- hard_fail_count: `0`
- warn_fail_count: `0`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=5 |
| FOC:eta_bounds | error | True | eta_min=62.000, eta_max=77.000 |
| FOC:cosphi_bounds | error | True | cos_min=0.3200, cos_max=0.8000 |
| FOC:m2_monotonic | error | True | violations=0 |
| FOC:i1_monotonic | warn | True | violations=0 |
| FOC:n2_non_increasing | warn | True | upward_jumps_gt3rpm=0 |
| FOC:n2_spike_detector | error | True | spikes=0, threshold=25.000 |
| FOC:eta_peak_location | warn | True | peak_rel=0.833 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.2000 |
| FOC:eta_spike_detector | error | True | spikes=0, threshold=9.000 |
| FOC:cosphi_spike_detector | error | True | spikes=0, threshold=0.1500 |
| FOC:p2_le_p1 | error | True | violations=0 |
| MIC:finite_rows | error | True | finite_rows=5 |
| MIC:eta_bounds | error | True | eta_min=62.000, eta_max=77.000 |
| MIC:cosphi_bounds | error | True | cos_min=0.3200, cos_max=0.8000 |
| MIC:m2_monotonic | error | True | violations=0 |
| MIC:i1_monotonic | warn | True | violations=0 |
| MIC:n2_non_increasing | warn | True | upward_jumps_gt3rpm=0 |
| MIC:n2_spike_detector | error | True | spikes=0, threshold=25.000 |
| MIC:eta_peak_location | warn | True | peak_rel=0.833 |
| MIC:cosphi_low_to_mid_rise | warn | True | delta=0.2000 |
| MIC:eta_spike_detector | error | True | spikes=0, threshold=9.000 |
| MIC:cosphi_spike_detector | error | True | spikes=0, threshold=0.1500 |
| MIC:p2_le_p1 | error | True | violations=0 |
