# Theory Validation Report: AL31

- csv_path: `C:\mic_theory\paper\ieee_2026\data\passport\20260304_al31_robust_rand009_nodrift_v3\raw\al31\working_characteristics_filtered.csv`
- passed: `False`
- hard_fail_count: `12`
- warn_fail_count: `0`

| Check | Severity | Pass | Details |
|---|---|---|---|
| FOC:finite_rows | error | True | finite_rows=20 |
| FOC:eta_bounds | error | True | eta_min=9.394, eta_max=81.600 |
| FOC:cosphi_bounds | error | True | cos_min=0.2257, cos_max=0.8247 |
| FOC:m2_monotonic | error | False | violations=8 |
| FOC:i1_monotonic | error | False | violations=9 |
| FOC:n2_non_increasing | error | False | upward_jumps_gt3rpm=9 |
| FOC:n2_spike_detector | error | False | spikes=18, threshold=25.000 |
| FOC:eta_peak_location | warn | True | peak_rel=0.773 |
| FOC:cosphi_low_to_mid_rise | warn | True | delta=0.1579 |
| FOC:eta_spike_detector | error | False | spikes=10, threshold=3.000 |
| FOC:cosphi_spike_detector | error | False | spikes=10, threshold=0.0800 |
| FOC:p2_le_p1 | error | True | violations=0 |
| MIC_AI:finite_rows | error | True | finite_rows=20 |
| MIC_AI:eta_bounds | error | True | eta_min=10.242, eta_max=82.150 |
| MIC_AI:cosphi_bounds | error | True | cos_min=0.1431, cos_max=0.8287 |
| MIC_AI:m2_monotonic | error | False | violations=8 |
| MIC_AI:i1_monotonic | error | False | violations=9 |
| MIC_AI:n2_non_increasing | error | False | upward_jumps_gt3rpm=9 |
| MIC_AI:n2_spike_detector | error | False | spikes=18, threshold=25.000 |
| MIC_AI:eta_peak_location | warn | True | peak_rel=0.614 |
| MIC_AI:cosphi_low_to_mid_rise | warn | True | delta=0.0920 |
| MIC_AI:eta_spike_detector | error | False | spikes=10, threshold=3.000 |
| MIC_AI:cosphi_spike_detector | error | False | spikes=12, threshold=0.0800 |
| MIC_AI:p2_le_p1 | error | True | violations=0 |
