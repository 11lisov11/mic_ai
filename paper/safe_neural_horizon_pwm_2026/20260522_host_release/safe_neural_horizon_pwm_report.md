# Safe Neural Horizon PWM Host Research Report

- status: `host_simulation_matrix_only`
- hardware_claim: `False`
- mc_trials: `3`
- steps_per_trial: `60`
- seed: `7`

## Scope

This is a host-level simulation report. It is not MCU, HIL, or bench evidence.
The comparison matrix uses named host baselines; none of those rows is hardware, HIL, or publication-tuned evidence.

## Scenario Matrix

### start_no_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 83.957 | 1.148 | 19.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 83.889 | 2.057 | 11.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 83.701 | 1.460 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 84.037 | 0.297 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 84.038 | 0.781 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 83.999 | 0.476 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 84.038 | 1.213 | 10.000 | 0.933 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.010 | 1.356 | 27.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.016 | 1.172 | 32.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 83.876 | 2.444 | 20.333 | 0.883 | 7.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### start_with_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 84.110 | 1.189 | 25.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 84.010 | 2.207 | 9.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 84.037 | 1.535 | 21.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 84.141 | 0.298 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 84.103 | 0.897 | 9.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 84.146 | 0.457 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 85.033 | 1.166 | 10.000 | 0.933 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.216 | 1.202 | 27.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.085 | 1.324 | 30.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 84.010 | 2.489 | 16.333 | 0.883 | 10.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### ramp_to_rated
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 92.078 | 1.262 | 26.333 | 0.967 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 92.012 | 1.897 | 13.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 92.004 | 1.369 | 18.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 92.234 | 0.302 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 92.188 | 1.100 | 10.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 92.186 | 0.395 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 92.348 | 1.049 | 10.000 | 0.850 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 92.067 | 1.347 | 33.333 | 0.967 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 94.569 | 1.118 | 31.000 | 0.967 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 91.269 | 2.740 | 12.000 | 0.717 | 16.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### load_step
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 84.028 | 1.286 | 25.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 83.903 | 2.291 | 9.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 83.962 | 1.536 | 23.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 84.435 | 0.298 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 84.349 | 0.703 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 84.082 | 0.448 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.391 | 1.355 | 10.667 | 0.933 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.022 | 1.221 | 31.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 84.039 | 1.487 | 27.667 | 0.983 | 0.667 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 83.937 | 2.616 | 15.333 | 0.883 | 12.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### load_shed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 94.437 | 1.284 | 20.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 94.153 | 2.538 | 7.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 94.247 | 1.335 | 22.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 94.405 | 0.298 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 94.644 | 0.770 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 94.316 | 0.481 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 94.434 | 1.331 | 10.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 94.337 | 1.177 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 94.490 | 1.704 | 23.000 | 1.000 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 94.205 | 2.752 | 14.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`

### reverse
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 62.586 | 1.110 | 19.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 62.656 | 1.968 | 16.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 62.792 | 1.285 | 24.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 62.773 | 0.302 | 72.667 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 62.783 | 0.894 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 62.772 | 0.422 | 6.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 62.797 | 1.219 | 11.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 62.754 | 1.175 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 62.773 | 1.411 | 24.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 62.649 | 1.991 | 20.667 | 0.533 | 0.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `safe_neural_horizon_pwm_h4_sparse`

### braking
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 44.517 | 1.252 | 22.667 | 0.467 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 43.976 | 1.690 | 22.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 44.481 | 1.262 | 14.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 44.554 | 0.300 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 44.728 | 0.788 | 8.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 44.643 | 0.248 | 6.667 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 44.621 | 1.441 | 13.333 | 0.417 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 44.521 | 1.229 | 27.667 | 0.400 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 44.547 | 1.468 | 27.667 | 0.411 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 44.499 | 1.322 | 43.333 | 0.367 | 1.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### regeneration
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 60.180 | 1.243 | 25.667 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 59.963 | 1.883 | 19.667 | 1.000 | 0.333 | 0 |
| foc_svm_key_baseline | 60.015 | 1.504 | 22.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 60.203 | 0.380 | 68.667 | 1.000 | 4.667 | 0 |
| dtc_svm_baseline | 60.201 | 0.955 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 59.593 | 0.529 | 6.667 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 60.085 | 1.120 | 10.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 60.179 | 1.366 | 31.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 60.086 | 1.530 | 30.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 59.674 | 2.199 | 22.667 | 0.372 | 9.333 | 0 |

Pareto front:
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h4_sparse`

### low_speed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 23.604 | 1.312 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 22.550 | 1.987 | 24.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 23.537 | 1.307 | 19.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 23.680 | 0.303 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 23.609 | 0.933 | 10.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 23.618 | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 23.664 | 1.033 | 8.667 | 0.133 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 23.564 | 1.366 | 32.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 23.541 | 1.553 | 29.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 23.515 | 1.693 | 34.000 | 0.083 | 0.333 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### zero_speed
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 0.032 | 0.903 | 29.000 | 0.200 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 0.032 | 0.937 | 26.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 0.134 | 1.124 | 10.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 0.038 | 0.348 | 63.667 | 1.000 | 2.000 | 0 |
| dtc_svm_baseline | 0.048 | 0.797 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 0.060 | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 0.423 | 1.205 | 11.333 | 0.133 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 0.102 | 0.789 | 26.333 | 0.100 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 0.063 | 0.659 | 38.000 | 0.100 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 0.036 | 0.763 | 34.000 | 0.083 | 0.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### field_weakening
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 180.772 | 1.273 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 180.521 | 2.431 | 7.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 180.604 | 1.392 | 24.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 182.288 | 0.305 | 66.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 180.686 | 1.057 | 14.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 180.691 | 0.501 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 180.683 | 1.409 | 9.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 180.733 | 1.208 | 32.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 179.153 | 1.938 | 22.667 | 1.000 | 3.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 180.559 | 2.218 | 16.333 | 1.000 | 3.333 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### overload
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.583 | 1.269 | 29.000 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 86.189 | 2.258 | 10.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.469 | 1.222 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.560 | 0.301 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 87.129 | 0.902 | 9.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 87.771 | 0.466 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.598 | 1.431 | 12.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.501 | 1.421 | 30.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.620 | 1.876 | 19.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.331 | 2.676 | 12.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`

### dc_sag
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.492 | 1.140 | 17.667 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 86.430 | 2.247 | 10.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.355 | 1.436 | 23.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.486 | 0.771 | 43.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 90.447 | 1.382 | 13.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 86.460 | 0.366 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.474 | 1.395 | 15.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 88.605 | 1.172 | 30.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.419 | 1.432 | 26.667 | 1.000 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 85.555 | 2.281 | 15.000 | 1.000 | 7.000 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### motor_heating
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.609 | 1.145 | 24.000 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 86.144 | 2.301 | 8.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.427 | 1.435 | 24.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.547 | 0.299 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 86.688 | 0.896 | 10.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 86.714 | 0.458 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.488 | 1.441 | 11.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.520 | 1.353 | 26.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.524 | 1.707 | 21.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.344 | 2.714 | 12.000 | 1.000 | 15.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`

### inverter_heating
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.744 | 1.176 | 27.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 85.955 | 2.537 | 7.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.423 | 1.472 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.518 | 0.302 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 86.499 | 0.955 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 86.563 | 0.427 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.475 | 1.377 | 9.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.533 | 1.156 | 28.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.462 | 1.512 | 24.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.449 | 2.234 | 22.333 | 1.000 | 3.333 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### rs_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.297 | 1.133 | 18.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.905 | 2.291 | 9.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.038 | 1.187 | 23.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.173 | 0.300 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.148 | 0.817 | 10.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.202 | 0.379 | 6.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.194 | 1.475 | 17.333 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 84.461 | 1.261 | 31.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.148 | 1.216 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.994 | 1.919 | 22.000 | 0.867 | 4.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### rr_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.393 | 1.113 | 18.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 77.091 | 1.605 | 13.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.049 | 1.997 | 25.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.236 | 0.295 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 78.853 | 1.321 | 10.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.112 | 0.480 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.469 | 1.506 | 12.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.155 | 1.156 | 21.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.106 | 1.770 | 26.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.021 | 2.439 | 21.000 | 0.867 | 9.333 | 1 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### lm_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.112 | 1.129 | 22.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.901 | 2.516 | 8.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.014 | 1.387 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 78.179 | 0.298 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.107 | 0.795 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.226 | 0.457 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.248 | 1.295 | 8.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.077 | 1.231 | 26.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.158 | 1.520 | 29.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 74.393 | 2.518 | 16.667 | 0.867 | 13.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### j_error
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.060 | 1.235 | 27.000 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.999 | 2.149 | 9.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.025 | 1.654 | 21.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.097 | 0.299 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.077 | 0.999 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.101 | 0.398 | 5.667 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.230 | 1.275 | 10.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.069 | 1.600 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.047 | 1.480 | 29.333 | 0.983 | 0.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.578 | 2.643 | 13.333 | 0.867 | 14.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### random_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.543 | 1.180 | 22.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 86.333 | 2.438 | 8.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.391 | 1.360 | 23.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.494 | 0.298 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 86.451 | 0.851 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 86.651 | 0.459 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.563 | 1.524 | 10.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.485 | 1.312 | 27.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.457 | 1.146 | 34.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.215 | 2.732 | 12.000 | 1.000 | 14.667 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `safe_neural_horizon_pwm_h4_sparse`

### periodic_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.506 | 1.160 | 26.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 86.289 | 2.513 | 7.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.372 | 1.494 | 21.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.502 | 0.298 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 86.481 | 0.916 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 87.133 | 0.484 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.592 | 1.512 | 11.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.469 | 1.208 | 31.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.450 | 1.159 | 33.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.415 | 2.218 | 21.000 | 1.000 | 6.333 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### shock_load
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 86.490 | 1.225 | 25.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 85.647 | 2.228 | 9.667 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 86.332 | 1.387 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 86.459 | 0.298 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 86.468 | 0.705 | 8.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 86.574 | 0.455 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 86.471 | 1.466 | 13.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 86.416 | 1.406 | 34.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 86.429 | 1.133 | 34.000 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 86.271 | 2.353 | 21.000 | 1.000 | 6.667 | 0 |

Pareto front:
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`

### two_mass_proxy
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 78.595 | 1.164 | 24.333 | 1.000 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 78.586 | 2.116 | 10.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 78.536 | 1.640 | 22.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 78.691 | 0.299 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 78.645 | 0.964 | 8.667 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 79.963 | 0.446 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 78.683 | 1.335 | 11.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 78.609 | 1.337 | 30.667 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 78.644 | 1.538 | 26.333 | 1.000 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 78.524 | 2.455 | 16.667 | 1.000 | 7.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### current_sensor_noise
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.155 | 1.266 | 24.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.659 | 2.517 | 8.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.027 | 1.210 | 20.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.111 | 0.297 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.161 | 0.809 | 8.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.126 | 0.437 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.324 | 1.041 | 9.333 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.076 | 1.249 | 30.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.146 | 1.451 | 27.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.944 | 2.698 | 15.333 | 0.867 | 15.000 | 1 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### speed_sensor_noise
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.213 | 1.152 | 21.333 | 0.989 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 77.026 | 2.081 | 11.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.032 | 1.329 | 22.333 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.087 | 0.296 | 72.000 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.232 | 0.980 | 10.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.086 | 0.414 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.720 | 1.191 | 10.667 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.128 | 1.279 | 30.667 | 0.989 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.102 | 1.609 | 27.333 | 0.994 | 2.333 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.959 | 2.741 | 16.333 | 0.867 | 25.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### sensor_delay
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.001 | 1.654 | 21.333 | 0.267 | 3.333 | 0 |
| fcs_mpc_one_step_baseline | 75.752 | 2.988 | 10.000 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 76.773 | 2.164 | 14.667 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.104 | 0.299 | 72.000 | 0.383 | 0.000 | 0 |
| dtc_svm_baseline | 77.228 | 1.249 | 2.000 | 0.383 | 0.000 | 0 |
| deadbeat_current_baseline | 77.131 | 0.817 | 12.667 | 0.383 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.393 | 1.505 | 8.000 | 0.200 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.036 | 1.497 | 37.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.665 | 1.406 | 32.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.154 | 2.154 | 19.667 | 0.133 | 19.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### speed_sensor_failure
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.116 | 1.180 | 24.667 | 0.267 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 75.767 | 2.255 | 9.333 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 77.030 | 1.280 | 21.333 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.234 | 0.301 | 72.000 | 0.383 | 0.000 | 0 |
| dtc_svm_baseline | 77.896 | 0.742 | 8.000 | 0.383 | 0.000 | 0 |
| deadbeat_current_baseline | 77.319 | 0.436 | 5.667 | 0.383 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.158 | 1.331 | 10.667 | 0.200 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.139 | 1.239 | 31.000 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.076 | 1.264 | 32.000 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.671 | 2.703 | 11.000 | 0.133 | 25.000 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### current_sensor_failure
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.359 | 1.259 | 21.333 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.364 | 2.503 | 7.000 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.016 | 1.393 | 20.667 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.112 | 0.298 | 72.333 | 1.000 | 0.000 | 0 |
| dtc_svm_baseline | 77.766 | 0.729 | 8.000 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.125 | 0.429 | 5.000 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.136 | 1.262 | 10.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.290 | 1.276 | 24.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.080 | 1.475 | 28.333 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 76.990 | 2.619 | 17.667 | 0.867 | 12.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h3_thermal`
- `safe_neural_horizon_pwm_h4_sparse`

### ood
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.094 | 1.149 | 17.667 | 0.983 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 77.130 | 1.625 | 13.333 | 1.000 | 0.000 | 0 |
| foc_svm_key_baseline | 77.066 | 1.963 | 26.000 | 1.000 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.105 | 0.302 | 72.667 | 1.000 | 0.333 | 0 |
| dtc_svm_baseline | 77.072 | 1.313 | 18.333 | 1.000 | 0.000 | 0 |
| deadbeat_current_baseline | 77.103 | 0.471 | 5.333 | 1.000 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.058 | 1.989 | 26.000 | 0.917 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.109 | 1.083 | 23.667 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.100 | 1.199 | 34.000 | 0.983 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 77.068 | 1.461 | 35.333 | 0.867 | 0.333 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### fault_injection_runtime
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.101 | 0.957 | 12.000 | 0.983 | 30.000 | 3 |
| fcs_mpc_one_step_baseline | 76.806 | 1.592 | 8.000 | 1.000 | 30.000 | 3 |
| foc_svm_key_baseline | 77.035 | 1.107 | 12.667 | 1.000 | 30.000 | 3 |
| dtc_hysteresis_baseline | 77.194 | 0.238 | 38.000 | 1.000 | 30.000 | 3 |
| dtc_svm_baseline | 77.098 | 0.668 | 6.000 | 1.000 | 30.000 | 3 |
| deadbeat_current_baseline | 77.332 | 0.000 | 0.000 | 1.000 | 30.000 | 3 |
| sensorless_adaptive_foc_baseline | 77.120 | 1.231 | 7.333 | 0.917 | 30.000 | 3 |
| safe_neural_horizon_pwm_h2 | 77.109 | 0.867 | 14.667 | 0.983 | 30.000 | 3 |
| safe_neural_horizon_pwm_h3_thermal | 77.180 | 0.937 | 16.667 | 0.983 | 30.000 | 3 |
| safe_neural_horizon_pwm_h4_sparse | 77.035 | 1.519 | 14.000 | 0.867 | 30.000 | 3 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
- `safe_neural_horizon_pwm_h2`
- `safe_neural_horizon_pwm_h4_sparse`

### sensor_dropout
| controller | speed_err | current | switches | feedback | fallback | failures |
|---|---|---|---|---|---|---|
| protected_ai_pwm_h1_baseline | 77.101 | 1.138 | 21.667 | 0.267 | 0.000 | 0 |
| fcs_mpc_one_step_baseline | 76.954 | 2.293 | 9.000 | 0.383 | 0.000 | 0 |
| foc_svm_key_baseline | 77.022 | 1.334 | 22.333 | 0.383 | 0.000 | 0 |
| dtc_hysteresis_baseline | 77.106 | 0.298 | 72.000 | 0.383 | 0.000 | 0 |
| dtc_svm_baseline | 77.281 | 0.783 | 8.667 | 0.383 | 0.000 | 0 |
| deadbeat_current_baseline | 77.326 | 0.422 | 5.000 | 0.383 | 0.000 | 0 |
| sensorless_adaptive_foc_baseline | 77.132 | 1.342 | 10.667 | 0.200 | 0.000 | 0 |
| safe_neural_horizon_pwm_h2 | 77.162 | 1.238 | 33.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h3_thermal | 77.078 | 1.213 | 32.667 | 0.250 | 0.000 | 0 |
| safe_neural_horizon_pwm_h4_sparse | 74.413 | 2.739 | 15.000 | 0.133 | 21.667 | 0 |

Pareto front:
- `protected_ai_pwm_h1_baseline`
- `fcs_mpc_one_step_baseline`
- `foc_svm_key_baseline`
- `dtc_hysteresis_baseline`
- `dtc_svm_baseline`
- `deadbeat_current_baseline`
- `sensorless_adaptive_foc_baseline`
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
- Not shown: publication-tuned FOC-SVM/DTC-SVM/deadbeat strength, trained neural twin, MCU timing, HIL, or bench safety.
