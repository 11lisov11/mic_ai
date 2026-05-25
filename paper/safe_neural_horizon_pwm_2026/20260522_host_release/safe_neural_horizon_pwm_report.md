# Safe Neural Horizon PWM Host Research Report

- status: `host_simulation_matrix_only`
- hardware_claim: `False`
- mc_trials: `3`
- steps_per_trial: `60`
- seed: `7`

## Scope

This is a host-level simulation report. It is not MCU, HIL, or bench evidence.
Controller names ending with `_proxy` are lightweight comparison proxies, not final strong baselines.

## Scenario Matrix

### start_no_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 83.957 | 1.148 | 19.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 84.023 | 0.816 | 35.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 83.701 | 1.460 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 84.008 | 1.524 | 46.667 | 1.000 | 2.000 | 0 |
| dtc_svm_proxy | 83.985 | 1.595 | 22.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 84.037 | 0.285 | 75.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 84.015 | 1.072 | 30.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.010 | 1.356 | 27.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.016 | 1.172 | 32.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 83.876 | 2.444 | 20.333 | 0.883 | 7.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### start_with_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 84.110 | 1.189 | 25.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 84.154 | 0.845 | 38.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 84.037 | 1.535 | 21.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 84.079 | 1.717 | 39.333 | 1.000 | 3.333 | 0 |
| dtc_svm_proxy | 84.070 | 1.537 | 19.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 84.150 | 0.294 | 74.333 | 1.000 | 3.333 | 0 |
| sensorless_adaptive_foc_proxy | 84.685 | 1.115 | 30.333 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.216 | 1.202 | 27.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.085 | 1.324 | 30.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 84.010 | 2.489 | 16.333 | 0.883 | 10.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### ramp_to_rated
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 92.078 | 1.262 | 26.333 | 0.967 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 92.084 | 0.833 | 37.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 92.004 | 1.369 | 18.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 92.004 | 1.757 | 38.000 | 1.000 | 1.000 | 0 |
| dtc_svm_proxy | 92.148 | 1.256 | 19.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 92.191 | 0.285 | 75.667 | 1.000 | 2.333 | 0 |
| sensorless_adaptive_foc_proxy | 92.203 | 1.181 | 29.667 | 0.817 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 92.067 | 1.347 | 33.333 | 0.967 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 94.569 | 1.118 | 31.000 | 0.967 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 91.269 | 2.740 | 12.000 | 0.717 | 16.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### load_step
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 84.028 | 1.286 | 25.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 84.077 | 0.870 | 39.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 83.962 | 1.536 | 23.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 83.644 | 1.663 | 46.000 | 1.000 | 3.667 | 0 |
| dtc_svm_proxy | 83.111 | 1.765 | 17.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 84.088 | 0.287 | 75.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 83.955 | 1.020 | 29.333 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.022 | 1.221 | 31.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.039 | 1.487 | 27.667 | 0.983 | 0.667 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 83.937 | 2.616 | 15.333 | 0.883 | 12.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### load_shed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 94.437 | 1.284 | 20.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 94.344 | 0.821 | 37.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 94.247 | 1.335 | 22.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 94.312 | 1.701 | 43.333 | 1.000 | 1.333 | 0 |
| dtc_svm_proxy | 94.512 | 1.348 | 19.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 94.320 | 0.282 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 94.368 | 1.069 | 29.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 94.337 | 1.177 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 94.490 | 1.704 | 23.000 | 1.000 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 94.205 | 2.752 | 14.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### reverse
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 62.586 | 1.110 | 19.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 62.590 | 0.855 | 42.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 62.792 | 1.285 | 24.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 62.757 | 1.701 | 30.000 | 1.000 | 0.000 | 0 |
| dtc_svm_proxy | 62.783 | 1.419 | 19.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 62.777 | 0.273 | 78.000 | 1.000 | 1.000 | 0 |
| sensorless_adaptive_foc_proxy | 62.793 | 1.114 | 26.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 62.754 | 1.175 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 62.773 | 1.411 | 24.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 62.649 | 1.991 | 20.667 | 0.533 | 0.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `deadbeat_current_proxy`
- `safe_neural_horizon_pwm_h4_sparse`

### braking
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 44.517 | 1.252 | 22.667 | 0.467 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 44.878 | 0.836 | 40.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 44.481 | 1.262 | 14.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 44.497 | 1.508 | 48.333 | 1.000 | 0.333 | 0 |
| dtc_svm_proxy | 44.575 | 1.357 | 17.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 44.647 | 0.286 | 75.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 44.586 | 1.084 | 23.000 | 0.417 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 44.521 | 1.229 | 27.667 | 0.400 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 44.547 | 1.468 | 27.667 | 0.411 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 44.499 | 1.322 | 43.333 | 0.367 | 1.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### regeneration
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 60.180 | 1.243 | 25.667 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 60.155 | 0.933 | 38.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 60.015 | 1.504 | 22.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 60.174 | 1.487 | 48.000 | 1.000 | 3.000 | 0 |
| dtc_svm_proxy | 60.166 | 1.451 | 21.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 59.929 | 0.314 | 75.000 | 1.000 | 2.667 | 0 |
| sensorless_adaptive_foc_proxy | 59.727 | 1.167 | 31.000 | 0.417 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 60.179 | 1.366 | 31.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 60.086 | 1.530 | 30.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 59.674 | 2.199 | 22.667 | 0.372 | 9.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### low_speed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 23.604 | 1.312 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 23.987 | 0.957 | 37.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 23.537 | 1.307 | 19.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 23.478 | 1.811 | 39.667 | 1.000 | 1.000 | 0 |
| dtc_svm_proxy | 23.579 | 1.436 | 21.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 23.618 | 0.272 | 78.000 | 1.000 | 1.000 | 0 |
| sensorless_adaptive_foc_proxy | 23.573 | 1.106 | 29.667 | 0.133 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 23.564 | 1.366 | 32.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 23.541 | 1.553 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 23.515 | 1.693 | 34.000 | 0.083 | 0.333 | 0 |

Pareto front:
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### zero_speed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 0.032 | 0.903 | 29.000 | 0.200 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 0.032 | 0.395 | 66.667 | 1.000 | 3.000 | 0 |
| foc_svm_key_baseline | 0.134 | 1.124 | 10.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 0.040 | 0.959 | 57.000 | 1.000 | 2.667 | 0 |
| dtc_svm_proxy | 0.043 | 0.789 | 50.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 0.060 | 0.272 | 78.000 | 1.000 | 1.000 | 0 |
| sensorless_adaptive_foc_proxy | 0.404 | 0.705 | 36.000 | 0.133 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 0.102 | 0.789 | 26.333 | 0.100 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 0.063 | 0.659 | 38.000 | 0.100 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 0.036 | 0.763 | 34.000 | 0.083 | 0.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### field_weakening
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 180.772 | 1.273 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 180.689 | 0.942 | 38.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 180.604 | 1.392 | 24.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 181.525 | 1.429 | 49.667 | 1.000 | 1.000 | 0 |
| dtc_svm_proxy | 180.666 | 1.442 | 18.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 180.702 | 0.271 | 75.000 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 180.668 | 1.016 | 25.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 180.733 | 1.208 | 32.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 179.153 | 1.938 | 22.667 | 1.000 | 3.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 180.559 | 2.218 | 16.333 | 1.000 | 3.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### overload
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.583 | 1.269 | 29.000 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 87.110 | 0.858 | 36.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.469 | 1.222 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.477 | 1.783 | 39.333 | 1.000 | 0.667 | 0 |
| dtc_svm_proxy | 86.996 | 1.363 | 20.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 87.814 | 0.292 | 75.000 | 1.000 | 2.667 | 0 |
| sensorless_adaptive_foc_proxy | 86.569 | 1.034 | 30.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.501 | 1.421 | 30.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.620 | 1.876 | 19.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.331 | 2.676 | 12.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### dc_sag
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.492 | 1.140 | 17.667 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 86.728 | 0.713 | 37.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.355 | 1.436 | 23.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.410 | 1.747 | 31.000 | 1.000 | 0.333 | 0 |
| dtc_svm_proxy | 91.788 | 1.194 | 18.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 86.461 | 0.249 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.456 | 1.016 | 27.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 88.605 | 1.172 | 30.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.419 | 1.432 | 26.667 | 1.000 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 85.555 | 2.281 | 15.000 | 1.000 | 7.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### motor_heating
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.609 | 1.145 | 24.000 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 86.808 | 0.959 | 36.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.427 | 1.435 | 24.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.480 | 1.636 | 42.000 | 1.000 | 1.667 | 0 |
| dtc_svm_proxy | 86.554 | 1.585 | 18.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 86.726 | 0.293 | 75.667 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.471 | 1.060 | 28.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.520 | 1.353 | 26.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.524 | 1.707 | 21.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.344 | 2.714 | 12.000 | 1.000 | 15.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### inverter_heating
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.744 | 1.176 | 27.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 86.853 | 1.030 | 35.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.423 | 1.472 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.434 | 1.739 | 35.333 | 1.000 | 1.000 | 0 |
| dtc_svm_proxy | 86.482 | 1.287 | 19.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 86.570 | 0.285 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.459 | 1.084 | 28.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.533 | 1.156 | 28.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.462 | 1.512 | 24.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.449 | 2.234 | 22.333 | 1.000 | 3.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### rs_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.297 | 1.133 | 18.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.273 | 0.771 | 39.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.038 | 1.187 | 23.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.070 | 1.723 | 40.000 | 1.000 | 1.000 | 0 |
| dtc_svm_proxy | 77.088 | 1.550 | 18.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.211 | 0.310 | 74.000 | 1.000 | 4.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.161 | 0.968 | 29.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.461 | 1.261 | 31.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.148 | 1.216 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.994 | 1.919 | 22.000 | 0.867 | 4.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### rr_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.393 | 1.113 | 18.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.106 | 0.719 | 37.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.049 | 1.997 | 25.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.141 | 1.459 | 53.333 | 1.000 | 4.000 | 0 |
| dtc_svm_proxy | 78.718 | 1.223 | 18.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.114 | 0.278 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.355 | 1.017 | 28.667 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.155 | 1.156 | 21.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.106 | 1.770 | 26.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.021 | 2.439 | 21.000 | 0.867 | 9.333 | 1 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### lm_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.112 | 1.129 | 22.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.210 | 0.690 | 37.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.014 | 1.387 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.589 | 1.460 | 52.333 | 1.000 | 5.000 | 0 |
| dtc_svm_proxy | 77.059 | 1.793 | 18.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.231 | 0.281 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.175 | 1.142 | 28.333 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.077 | 1.231 | 26.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.158 | 1.520 | 29.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 74.393 | 2.518 | 16.667 | 0.867 | 13.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### j_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.060 | 1.235 | 27.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.088 | 0.976 | 35.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.025 | 1.654 | 21.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.059 | 1.546 | 49.667 | 1.000 | 1.667 | 0 |
| dtc_svm_proxy | 77.065 | 1.357 | 21.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.104 | 0.308 | 73.333 | 1.000 | 4.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.191 | 1.048 | 28.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.069 | 1.600 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.047 | 1.480 | 29.333 | 0.983 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.578 | 2.643 | 13.333 | 0.867 | 14.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### random_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.543 | 1.180 | 22.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 86.682 | 0.933 | 40.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.391 | 1.360 | 23.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.424 | 1.629 | 43.333 | 1.000 | 1.667 | 0 |
| dtc_svm_proxy | 86.411 | 1.717 | 18.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 86.662 | 0.295 | 74.667 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.533 | 1.045 | 26.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.485 | 1.312 | 27.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.457 | 1.146 | 34.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.215 | 2.732 | 12.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### periodic_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.506 | 1.160 | 26.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 86.459 | 0.864 | 35.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.372 | 1.494 | 21.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.371 | 1.840 | 35.333 | 1.000 | 1.333 | 0 |
| dtc_svm_proxy | 86.452 | 1.367 | 20.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 87.184 | 0.293 | 75.000 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.532 | 1.039 | 30.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.469 | 1.208 | 31.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.450 | 1.159 | 33.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.415 | 2.218 | 21.000 | 1.000 | 6.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### shock_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 86.490 | 1.225 | 25.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 88.358 | 0.908 | 33.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.332 | 1.387 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 86.402 | 1.627 | 43.333 | 1.000 | 1.667 | 0 |
| dtc_svm_proxy | 86.386 | 1.798 | 17.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 86.586 | 0.283 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 86.447 | 1.039 | 30.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.416 | 1.406 | 34.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.429 | 1.133 | 34.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.271 | 2.353 | 21.000 | 1.000 | 6.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### two_mass_proxy
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 78.595 | 1.164 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 78.749 | 0.889 | 35.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 78.536 | 1.640 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 78.602 | 1.685 | 41.333 | 1.000 | 4.333 | 0 |
| dtc_svm_proxy | 78.615 | 1.327 | 21.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 80.039 | 0.283 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 78.649 | 1.073 | 29.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 78.609 | 1.337 | 30.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 78.644 | 1.538 | 26.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 78.524 | 2.455 | 16.667 | 1.000 | 7.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### current_sensor_noise
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.155 | 1.266 | 24.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.277 | 0.981 | 38.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.027 | 1.210 | 20.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.069 | 1.550 | 48.000 | 1.000 | 5.333 | 0 |
| dtc_svm_proxy | 77.091 | 1.601 | 17.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.130 | 0.294 | 74.667 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.188 | 1.266 | 29.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.076 | 1.249 | 30.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.146 | 1.451 | 27.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.944 | 2.698 | 15.333 | 0.867 | 15.000 | 1 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### speed_sensor_noise
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.213 | 1.152 | 21.333 | 0.989 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.102 | 0.877 | 34.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.032 | 1.329 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.060 | 1.522 | 51.000 | 1.000 | 3.000 | 0 |
| dtc_svm_proxy | 77.183 | 1.302 | 18.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.088 | 0.294 | 75.000 | 1.000 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.462 | 1.066 | 30.333 | 0.939 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.128 | 1.279 | 30.667 | 0.989 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.102 | 1.609 | 27.333 | 0.994 | 2.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.959 | 2.741 | 16.333 | 0.867 | 25.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### sensor_delay
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.001 | 1.654 | 21.333 | 0.267 | 3.333 | 0 |
| fcs_mpc_one_step_proxy | 77.738 | 0.963 | 37.667 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 76.773 | 2.164 | 14.667 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_proxy | 76.993 | 2.331 | 32.000 | 0.383 | 4.333 | 0 |
| dtc_svm_proxy | 76.876 | 2.566 | 15.667 | 0.383 | 7.000 | 0 |
| deadbeat_current_proxy | 77.143 | 0.356 | 75.000 | 0.383 | 5.667 | 0 |
| sensorless_adaptive_foc_proxy | 76.999 | 1.658 | 24.000 | 0.200 | 2.667 | 0 |
| safe_neural_horizon_pwm_h2 | 77.036 | 1.497 | 37.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.665 | 1.406 | 32.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.154 | 2.154 | 19.667 | 0.133 | 19.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### speed_sensor_failure
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.116 | 1.180 | 24.667 | 0.267 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 78.547 | 0.733 | 37.333 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 77.030 | 1.280 | 21.333 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.073 | 1.681 | 42.667 | 0.383 | 1.667 | 0 |
| dtc_svm_proxy | 77.529 | 1.548 | 18.000 | 0.383 | 0.333 | 0 |
| deadbeat_current_proxy | 77.337 | 0.295 | 74.667 | 0.383 | 3.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.116 | 1.171 | 27.000 | 0.200 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.139 | 1.239 | 31.000 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.076 | 1.264 | 32.000 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.671 | 2.703 | 11.000 | 0.133 | 25.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### current_sensor_failure
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.359 | 1.259 | 21.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.333 | 1.063 | 37.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.016 | 1.393 | 20.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.070 | 1.524 | 52.333 | 1.000 | 3.000 | 0 |
| dtc_svm_proxy | 77.499 | 1.795 | 17.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.128 | 0.282 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.101 | 1.089 | 31.667 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.290 | 1.276 | 24.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.080 | 1.475 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.990 | 2.619 | 17.667 | 0.867 | 12.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### ood
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.094 | 1.149 | 17.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.144 | 0.636 | 42.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.066 | 1.963 | 26.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.081 | 1.389 | 47.333 | 1.000 | 2.333 | 0 |
| dtc_svm_proxy | 77.069 | 1.164 | 18.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_proxy | 77.104 | 0.281 | 76.000 | 1.000 | 2.000 | 0 |
| sensorless_adaptive_foc_proxy | 77.056 | 0.964 | 23.667 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.109 | 1.083 | 23.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.100 | 1.199 | 34.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.068 | 1.461 | 35.333 | 0.867 | 0.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### fault_injection_runtime
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.101 | 0.957 | 12.000 | 0.983 | 30.000 | 3 |
| fcs_mpc_one_step_proxy | 78.179 | 0.475 | 21.333 | 1.000 | 30.000 | 3 |
| foc_svm_key_baseline | 77.035 | 1.107 | 12.667 | 1.000 | 30.000 | 3 |
| dtc_hysteresis_proxy | 77.113 | 1.033 | 24.667 | 1.000 | 30.333 | 3 |
| dtc_svm_proxy | 77.069 | 0.960 | 14.667 | 1.000 | 30.000 | 3 |
| deadbeat_current_proxy | 77.332 | 0.231 | 38.000 | 1.000 | 31.000 | 3 |
| sensorless_adaptive_foc_proxy | 77.098 | 0.714 | 15.333 | 0.917 | 30.000 | 3 |
| safe_neural_horizon_pwm_h2 | 77.109 | 0.867 | 14.667 | 0.983 | 30.000 | 3 |
| safe_neural_horizon_pwm_h3_thermal | 77.180 | 0.937 | 16.667 | 0.983 | 30.000 | 3 |
| safe_neural_horizon_pwm_h4_sparse | 77.035 | 1.519 | 14.000 | 0.867 | 30.000 | 3 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### sensor_dropout
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_proxy | 77.101 | 1.138 | 21.667 | 0.267 | 0.000 | 0 |
| fcs_mpc_one_step_proxy | 77.119 | 0.771 | 39.000 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 77.022 | 1.334 | 22.333 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_proxy | 77.059 | 1.515 | 49.667 | 0.383 | 1.000 | 0 |
| dtc_svm_proxy | 77.196 | 1.331 | 19.667 | 0.383 | 0.000 | 0 |
| deadbeat_current_proxy | 77.338 | 0.297 | 74.667 | 0.383 | 3.333 | 0 |
| sensorless_adaptive_foc_proxy | 77.106 | 0.973 | 29.333 | 0.200 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.162 | 1.238 | 33.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.078 | 1.213 | 32.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 74.413 | 2.739 | 15.000 | 0.133 | 21.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_proxy`
- `fcs_mpc_one_step_proxy`
- `foc_svm_key_baseline`
- `dtc_hysteresis_proxy`
- `dtc_svm_proxy`
- `deadbeat_current_proxy`
- `sensorless_adaptive_foc_proxy`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

## Ablation
| variant | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| ablation_h1_no_horizon | 83.252 | 1.431 | 33.000 | 0.983 | 0.000 | 0 |
| ablation_h2_dense_feedback | 84.025 | 1.343 | 32.667 | 1.000 | 0.000 | 0 |
| ablation_h2_sparse_feedback | 83.996 | 1.395 | 33.000 | 0.833 | 0.000 | 0 |
| ablation_h2_low_switching | 83.823 | 3.083 | 5.000 | 0.983 | 18.667 | 0 |
| ablation_h2_low_current | 84.072 | 0.302 | 73.000 | 0.983 | 2.333 | 0 |

Ablation Pareto front:
- `ablation_h1_no_horizon`
- `ablation_h2_dense_feedback`
- `ablation_h2_sparse_feedback`
- `ablation_h2_low_switching`
- `ablation_h2_low_current`

## Fault Injection

- all_gateway_cases_no_shoot_through: `True`
- raw_shoot_through_detector_triggered: `True`
| case | accepted | pwm_enabled | fault_flags | latched | shoot_through |
|---|---|---|---|---|---|
| invalid_vector | False | False | 128 | True | False |
| too_short_pulse | False | True | 512 | False | False |
| overcurrent | False | False | 1 | True | False |
| overtemperature | False | False | 8 | True | False |
| undervoltage | False | False | 32 | True | False |
| uvlo_like_undervoltage | False | False | 32 | True | False |
| desat_like_overcurrent | False | False | 1 | True | False |
| low_confidence | False | True | 1024 | False | False |
| watchdog | False | False | 64 | True | False |
| raw_shoot_through_request_emulation | False | False | 0 | True | True |
| no_deadtime_transition_emulation | False | False | 0 | True | False |

## Honest Status

- Shown: host-level vector safety, scenario smoke, ablation smoke, Pareto extraction.
- Not shown: real FOC-SVM/DTC-SVM strength, trained neural twin, MCU timing, HIL, or bench safety.
