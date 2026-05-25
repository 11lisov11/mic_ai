# Safe Neural Horizon PWM with Event-Triggered Twin Feedback

## Abstract

This draft describes a host-simulated induction-motor control variant that combines a neural cost shaper, short-horizon inverter-vector search, an event-triggered neural twin, and a protected AI-PWM Safety Gateway. The method is evaluated only in software simulation in this release. No MCU, HIL, or bench claim is made.

## Contribution

- Alpha-beta induction-motor model with parameter randomization hooks.
- Two-level inverter model with legal vector set, dead-time proxy, loss proxy, and common-mode proxy.
- Safety Gateway that prevents direct AI access to raw high/low gate commands.
- Host-tested no-shoot-through and no-direct-HIGH-to-LOW timing-path invariants for vector transitions.
- Horizon AI-PWM controller with neural cost shaping and event-triggered feedback policy.
- Domain-randomized theta-conditioned twin identification evidence with multi-step rollout losses.
- Bounded parameter-sweep tuning evidence for all named host comparison baselines.
- Scenario matrix, ablation smoke, Pareto extraction, fault-injection summary, and host trace/FFT evidence package.
- Machine-checkable release, algorithm-identity, novelty, and theory-completion audits.

## Novelty Claim Scope

The host-level novelty claim is architectural, not a hardware or universal-superiority claim: SNH-PWM combines event-triggered twin feedback, neural cost shaping, finite-horizon inverter-vector search, and a protected AI-PWM Safety Gateway into one control law.

Compared with classical FOC-SVM, the controller does not synthesize continuous voltage references and then apply SVM; it searches legal inverter vectors directly under feedback/switching/risk costs. Compared with one-step FCS-MPC, it adds neural cost shaping, event-triggered feedback economy, and a mandatory gate-safety layer. Compared with the prior protected AI-PWM H1 model, it adds horizon search, twin uncertainty, and explicit feedback-usage optimization.

The tracked release therefore supports only this claim: a distinct host-simulated control architecture exists and is machine-checked against the current host evidence.

The companion theory-completion audit separates host/software evidence from hardware evidence. When `publication_theory_complete = true`, it means the current host-theory evidence package passes the configured software gates; it still does not claim MCU, HIL, bench, or universal-superiority readiness.

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
- The FCS-MPC comparison is now a separate one-step current/torque/flux predictive baseline.
- The prior protected AI-PWM H1, FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current, and sensorless/adaptive FOC comparisons are now separate host baselines.
- The new FOC-SVM/FCS-MPC/DTC/DTC-SVM/deadbeat/sensorless baselines are competitive, so SNH-PWM cannot claim classical-control superiority yet.

## Limitations

- Host simulation only.
- FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC have bounded host tuning evidence, but are still not hardware or vendor-grade certified controllers.
- Domain-randomized theta-conditioned twin evidence exists, but it is host-only and assumes theta/passport or identification context.
- Host MC=500 publication-scale smoke exists, but final MC must be repeated after strong-baseline tuning.
- Host trace package with time-series CSV plus FFT/THD-like torque-current evidence exists, but it is still simulation-only and not hardware THD.
- No fixed-point/WCET analysis.
- No MCU, HIL, oscilloscope, inverter, or motor-bench validation.

## Required Next Work

- Expand the bounded tuning sweep into larger publication sweeps after any model/controller change.
- Re-run publication-scale MC after baseline replacement/tuning.
- Expand the host trace/FFT package after baseline tuning and validate it against HIL/bench traces.
- Replace the theta-conditioned host twin with a production online identifier before MCU/HIL/bench claims.
- Port the safety gateway and timing checks to the target MCU/HIL path.
- Validate gate timing and current trips on real hardware before any hardware-ready claim.
