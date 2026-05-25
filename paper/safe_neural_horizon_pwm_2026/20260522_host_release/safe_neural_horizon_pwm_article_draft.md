# Safe Neural Horizon PWM with Event-Triggered Twin Feedback

## Abstract

This draft describes a host-simulated induction-motor control variant that combines a neural cost shaper, short-horizon inverter-vector search, an event-triggered neural twin, and a protected AI-PWM Safety Gateway. The method is evaluated only in software simulation in this release. No MCU, HIL, or bench claim is made.

## Contribution

- Alpha-beta induction-motor model with parameter randomization hooks.
- Two-level inverter model with legal vector set, dead-time proxy, loss proxy, and common-mode proxy.
- Safety Gateway that prevents direct AI access to raw high/low gate commands.
- Host-tested no-shoot-through and no-direct-HIGH-to-LOW timing-path invariants for vector transitions.
- Horizon AI-PWM controller with neural cost shaping and event-triggered feedback policy.
- Scenario matrix, ablation smoke, Pareto extraction, and fault-injection summary.
- Machine-checkable release, novelty, and theory-completion audits.

## Novelty Claim Scope

The host-level novelty claim is architectural, not a hardware or universal-superiority claim: SNH-PWM combines event-triggered twin feedback, neural cost shaping, finite-horizon inverter-vector search, and a protected AI-PWM Safety Gateway into one control law.

Compared with classical FOC-SVM, the controller does not synthesize continuous voltage references and then apply SVM; it searches legal inverter vectors directly under feedback/switching/risk costs. Compared with one-step FCS-MPC, it adds neural cost shaping, event-triggered feedback economy, and a mandatory gate-safety layer. Compared with the prior protected AI-PWM H1 model, it adds horizon search, twin uncertainty, and explicit feedback-usage optimization.

The tracked release therefore supports only this claim: a distinct host-simulated control architecture exists and is machine-checked against the current host evidence.

The companion theory-completion audit separates `host_theory_scaffold_ready = true` from `publication_theory_complete = false`. This is intentional: the host scaffold is ready for continued research, but publication-grade superiority and hardware readiness are not claimed.

## Method

The AI layer requests only `vector_id in {0..7}`. The gateway maps accepted vectors to gate states and inserts BOTH_OFF dead-time states on changing legs. Unsafe requests are rejected, held, or latched depending on fault severity.

The optimization cost includes speed error, torque error, current stress, flux building, torque-ripple proxy, switching events, loss proxy, thermal proxy, feedback usage, confidence/risk, and common-mode proxy.

## Evaluation

- Status: `host_simulation_matrix_only`
- Hardware claim: `False`
- MC trials: `3`
- Steps per trial: `60`
- Scenarios: `31`

Scenario list:
- `start_no_load`
- `start_with_load`
- `ramp_to_rated`
- `load_step`
- `load_shed`
- `reverse`
- `braking`
- `regeneration`
- `low_speed`
- `zero_speed`
- `field_weakening`
- `overload`
- `dc_sag`
- `motor_heating`
- `inverter_heating`
- `rs_error`
- `rr_error`
- `lm_error`
- `j_error`
- `random_load`
- `periodic_load`
- `shock_load`
- `two_mass_proxy`
- `current_sensor_noise`
- `speed_sensor_noise`
- `sensor_delay`
- `speed_sensor_failure`
- `current_sensor_failure`
- `ood`
- `fault_injection_runtime`
- `sensor_dropout`

Fault-injection result:
- all_gateway_cases_no_shoot_through: `True`
- raw_shoot_through_detector_triggered: `True`
- deadtime_transition_detector_triggered: `True`

## Preliminary Findings

- H2 is the safer current research candidate than the sparse H4 variant in the short host matrix.
- Sparse H4 can reduce feedback and switching, but current stress and fallback events increase in several scenarios.
- One-step FCS proxy tends to keep current lower but uses dense feedback and more switching.
- Current proxy baselines are useful for smoke testing but are not publication-grade strong baselines.

## Limitations

- Host simulation only.
- Proxy baselines, not final tuned FOC-SVM/FCS-MPC/DTC-SVM implementations.
- No trained domain-randomized neural twin yet.
- First MC=100 smoke exists, but no MC=500..1000 publication-scale run yet.
- No long-run trace package with FFT/THD torque-current evidence yet.
- No fixed-point/WCET analysis.
- No MCU, HIL, oscilloscope, inverter, or motor-bench validation.

## Required Next Work

- Replace proxy baselines with tuned strong baselines.
- Run publication-scale MC after baseline replacement.
- Add publication-grade plots and FFT/THD metrics.
- Port the safety gateway and timing checks to the target MCU/HIL path.
- Validate gate timing and current trips on real hardware before any hardware-ready claim.
