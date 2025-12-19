# Scoring Metrics (Stage 10)

This document defines the metrics and score formula used in the Digital Twin Bench.

## Metrics per case
- iae: integral of absolute speed error, sum(|omega_ref - omega_m|) * dt
- overshoot: max(max(0, |omega_m| - |omega_ref|))
- settling_time: time from last violation of a 5% band around the final reference
- i_rms: RMS of three-phase current
- smoothness: RMS of d/dt for v_d and v_q (control activity)
- energy_proxy: mean absolute electrical power, mean(|p_in|)
- ripple: RMS of current magnitude ripple around its mean
- safety_trip: 1 if any safety flag trips, else 0

## Derived terms
```
p_in = 1.5 * (v_d * i_d + v_q * i_q)
i_mag = sqrt(i_d^2 + i_q^2)
ripple = RMS(i_mag - mean(i_mag))
```

## Normalization
```
iae_norm = iae / (ref_scale * duration)
overshoot_norm = overshoot / ref_scale
settling_norm = settling_time / duration
i_rms_norm = i_rms / current_scale
smooth_norm = smoothness / (voltage_scale / dt)
energy_norm = energy_proxy / power_scale
ripple_norm = ripple / current_scale
power_scale = 1.5 * voltage_scale * current_scale
```

## Score formula
```
cost = w_iae * iae_norm
     + w_overshoot * overshoot_norm
     + w_settling * settling_norm
     + w_i_rms * i_rms_norm
     + w_smooth * smooth_norm
     + w_energy * energy_norm
     + w_ripple * ripple_norm

score = base_score / (1 + cost)
```

Weights (default):
```
w_iae = 1.0
w_overshoot = 0.5
w_settling = 0.2
w_i_rms = 0.5
w_smooth = 0.01
w_energy = 0.2
w_ripple = 0.2
```

## Safety rule (DQ)
If any case has safety_trip == 1, the policy is disqualified and the suite score is set to 0.0.

## Conditions for comparability
Scores are comparable only under identical experiment conditions (testsuite definition, dt, limits, noise, identification).
