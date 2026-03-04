# Theory Validation Report: AO2

- csv_path: `C:\mic_theory\paper\ieee_2026\data\passport\20260304_al31_robust_rand009_nodrift_v3\raw\ao2\working_characteristics.csv`
- passed: `False`
- hard_fail_count: `10`
- warn_fail_count: `2`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=20 |
| FOC:eta_bounds | error | True | eta_min=0.000, eta_max=102.000 |
| FOC:cosphi_bounds | error | True | cos_min=0.0000, cos_max=0.4382 |
| FOC:m2_monotonic | error | False | violations=4 |
| FOC:i1_monotonic | error | False | violations=3 |
| FOC:n2_non_increasing | error | False | upward_jumps_gt3rpm=8 |
| FOC:n2_spike_detector | error | False | spikes=10, threshold=25.000 |
| FOC:eta_peak_location | warn | True | peak_rel=0.938 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.3635 |
| FOC:eta_spike_detector | error | False | spikes=4, threshold=3.000 |
| FOC:cosphi_spike_detector | error | False | spikes=3, threshold=0.1953 |
| FOC:p2_le_p1 | error | False | violations=1 |
| MIC_AI:finite_rows | error | True | finite_rows=20 |
| MIC_AI:eta_bounds | error | True | eta_min=0.000, eta_max=0.000 |
| MIC_AI:cosphi_bounds | error | True | cos_min=0.0000, cos_max=0.2331 |
| MIC_AI:m2_monotonic | error | True | violations=0 |
| MIC_AI:i1_monotonic | error | False | violations=6 |
| MIC_AI:n2_non_increasing | error | False | upward_jumps_gt3rpm=4 |
| MIC_AI:n2_spike_detector | error | True | spikes=0, threshold=25.000 |
| MIC_AI:eta_peak_location | warn | False | peak_rel=0.000 |
| MIC_AI:cosphi_low_to_mid_rise | warn | False | delta=-0.0489 |
| MIC_AI:eta_spike_detector | error | True | spikes=0, threshold=3.000 |
| MIC_AI:cosphi_spike_detector | error | False | spikes=4, threshold=0.0800 |
| MIC_AI:p2_le_p1 | error | True | violations=0 |
