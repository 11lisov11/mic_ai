# Theory Validation Report: AIR56

- csv_path: `C:\mic_theory\paper\ieee_2026\data\passport\20260304_al31_robust_rand009_nodrift_v3\raw\air56\working_characteristics_filtered.csv`
- passed: `False`
- hard_fail_count: `7`
- warn_fail_count: `0`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=5 |
| FOC:eta_bounds | error | True | eta_min=21.763, eta_max=64.706 |
| FOC:cosphi_bounds | error | True | cos_min=0.2950, cos_max=0.5731 |
| FOC:m2_monotonic | error | True | violations=0 |
| FOC:i1_monotonic | error | False | violations=3 |
| FOC:n2_non_increasing | warn | True | upward_jumps_gt3rpm=1 |
| FOC:n2_spike_detector | error | False | spikes=2, threshold=25.000 |
| FOC:eta_peak_location | warn | True | peak_rel=1.000 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.0648 |
| FOC:eta_spike_detector | error | False | spikes=3, threshold=3.000 |
| FOC:cosphi_spike_detector | error | True | spikes=0, threshold=0.0800 |
| FOC:p2_le_p1 | error | True | violations=0 |
| MIC_AI:finite_rows | error | True | finite_rows=5 |
| MIC_AI:eta_bounds | error | True | eta_min=19.278, eta_max=61.582 |
| MIC_AI:cosphi_bounds | error | True | cos_min=0.3944, cos_max=0.7109 |
| MIC_AI:m2_monotonic | error | True | violations=0 |
| MIC_AI:i1_monotonic | error | False | violations=3 |
| MIC_AI:n2_non_increasing | warn | True | upward_jumps_gt3rpm=1 |
| MIC_AI:n2_spike_detector | error | False | spikes=2, threshold=25.000 |
| MIC_AI:eta_peak_location | warn | True | peak_rel=1.000 |
| MIC_AI:cosphi_low_to_mid_rise | warn | True | delta=0.1141 |
| MIC_AI:eta_spike_detector | error | False | spikes=3, threshold=3.000 |
| MIC_AI:cosphi_spike_detector | error | False | spikes=2, threshold=0.0800 |
| MIC_AI:p2_le_p1 | error | True | violations=0 |
