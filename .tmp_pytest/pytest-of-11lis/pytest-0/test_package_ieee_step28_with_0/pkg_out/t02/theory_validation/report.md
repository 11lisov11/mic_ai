# Theory Validation Report

- csv_path: `C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_package_ieee_step28_with_0\working_chars.csv`
- passed: `True`
- hard_fail_count: `0`
- warn_fail_count: `0`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=4 |
| FOC:eta_bounds | error | True | eta_min=62.000, eta_max=77.000 |
| FOC:cosphi_bounds | error | True | cos_min=0.3200, cos_max=0.7900 |
| FOC:m2_monotonic | error | True | violations=0 |
| FOC:i1_monotonic | warn | True | violations=0 |
| FOC:n2_non_increasing | warn | True | upward_jumps_gt3rpm=0 |
| FOC:n2_spike_detector | error | True | spikes=0, threshold=25.000 |
| FOC:eta_peak_location | warn | True | peak_rel=1.000 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.2000 |
| FOC:eta_spike_detector | error | True | spikes=0, threshold=3.000 |
| FOC:cosphi_spike_detector | error | True | spikes=0, threshold=0.0800 |
| FOC:p2_le_p1 | error | True | violations=0 |
