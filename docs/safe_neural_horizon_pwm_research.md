# Safe Neural Horizon PWM Research Track

Date: `2026-05-22`
Status: `HOST_SIMULATION_ONLY`
Hardware claim: `false`

## Name

Working title:

`Safe Neural Horizon PWM with Event-Triggered Twin Feedback`

Short name:

`SNH-PWM`

Russian description:

`нейропредиктивный ИИ-ШИМ с горизонтом, нейродвойником, событийной обратной связью и защищенным Safety Gateway`.

## Hypothesis

The new variant should not claim that AI is universally better than FOC or MPC.

The testable hypothesis is narrower:

- A neural cost shaper plus a short horizon vector search can improve the trade-off between speed error, switching activity, current stress, and feedback usage under parameter uncertainty.
- The AI layer must never command raw low-side gates. It may only request one of eight legal inverter vectors.
- A Safety Gateway must be able to reject unsafe vector requests, enforce dead-time/min-pulse/current/confidence limits, and keep the no-shoot-through invariant true for every generated gate waveform.
- Full no-feedback control is not claimed; the target is event-triggered/almost-feedbackless control with fallback when uncertainty grows.

## Architecture

```text
omega_ref / torque target
        |
        v
event-triggered feedback policy
        |
        v
neural twin: physics alpha-beta model + residual envelope
        |
        v
neural cost shaping + horizon vector search H=1..4
        |
        v
requested switch vector in {000..111}
        |
        v
Safety Gateway: vector, dead-time, min-pulse, current, Vdc, Tj, confidence, OOD
        |
        v
dead-time gate waveform: HIGH -> BOTH_OFF -> LOW
        |
        v
two-level inverter + induction motor alpha-beta model
```

## Mathematical Motor Model

State:

```text
x = [psi_s_alpha, psi_s_beta, psi_r_alpha, psi_r_beta, omega_m]
```

Flux-current equations per alpha/beta axis:

```text
[psi_s] = [Ls  Lm] [i_s]
[psi_r]   [Lm  Lr] [i_r]

Ls = Lls + Lm
Lr = Llr + Lm
```

The implementation solves this 2x2 system for alpha and beta axes. The stationary-frame dynamics are:

```text
d psi_s_alpha / dt = v_alpha - Rs(Ts) * i_s_alpha
d psi_s_beta  / dt = v_beta  - Rs(Ts) * i_s_beta

d psi_r_alpha / dt = -Rr(Tr) * i_r_alpha - p * omega_m * psi_r_beta
d psi_r_beta  / dt = -Rr(Tr) * i_r_beta  + p * omega_m * psi_r_alpha

T_e = 1.5 * p * (psi_s_alpha * i_s_beta - psi_s_beta * i_s_alpha)
d omega_m / dt = (T_e - T_L - B * omega_m) / J
```

Implemented in:

- `models/induction_motor_alpha_beta.py`

Current included model effects:

- alpha-beta flux state
- Rs/Rr temperature scaling hooks
- Lm saturation hook
- parameter domain randomization for Rs, Rr, Lm, J, B

Still not complete:

- steel-loss state model
- two-mass shaft model
- sensor delay/ADC quantization in the main loop
- validated real motor parameter identification

## Inverter Model

The two-level inverter uses:

```text
s = [Sa, Sb, Sc], Sa/Sb/Sc in {0, 1}
U = {000, 001, 010, 011, 100, 101, 110, 111}
```

The motor phase voltages are computed after common-mode removal:

```text
v_a = (Sa - mean(S)) * Vdc
v_b = (Sb - mean(S)) * Vdc
v_c = (Sc - mean(S)) * Vdc
```

Then Clarke transform gives `v_alpha, v_beta`.

Implemented in:

- `models/two_level_inverter.py`

Included nonideal proxies:

- dead-time voltage error
- device voltage drop
- on-resistance conduction loss
- switching-event loss proxy
- common-mode voltage proxy

Still not complete:

- diode reverse recovery
- detailed IGBT/MOSFET temperature-dependent loss maps
- DC-link ripple model
- validated EMI model

## Safety Gateway

Implemented in:

- `safety/ai_pwm_gateway.py`

The AI request contains:

```text
vector_id
dwell_s
confidence
predicted_i_abs
measured_i_abs
vdc
tj_c
predicted_risk
watchdog_ok
```

The gateway checks:

- valid vector id
- dwell >= min_pulse
- predicted current < i_soft
- measured current < i_trip
- Vdc inside limits
- junction temperature inside limits
- confidence above minimum
- risk below maximum
- switching budget
- watchdog
- fault latch

The AI never receives direct access to:

```text
AH, AL, BH, BL, CH, CL
```

For every vector transition the generated timing path is:

```text
old vector -> changed legs BOTH_OFF for dead-time ticks -> new vector
```

The tested invariant is:

```text
not (AH and AL)
not (BH and BL)
not (CH and CL)
```

Host-level test:

```bash
python -m pytest -q tests/test_safe_neural_horizon_pwm.py
```

Current result:

```text
13 passed
```

The test file includes:

- 8x8 no-shoot-through transition waveform check
- invalid-vector fault injection
- min-pulse fault injection
- AI-confidence fault injection
- overcurrent fault injection
- undervoltage fault injection
- overtemperature fault injection
- watchdog fault injection
- H=4 bounded sequence-selection smoke
- matrix, Pareto, fault-summary, and markdown-report builder smoke

## AI Controller And AI-PWM

Implemented in:

- `control/safe_neural_horizon_pwm.py`

The first research implementation is deliberately lightweight:

- no heavy RL training
- no claim of learned optimality
- deterministic neural cost-shaping MLP
- short-horizon enumeration of safe inverter vectors
- horizon supported by code: `H=1..4`
- default smoke horizon: `H=2`

Cost terms:

```text
J = J_speed
  + J_torque
  + J_current
  + J_flux_building
  + J_torque_ripple
  + J_switching
  + J_loss
  + J_thermal
  + J_feedback_usage
  + J_risk
  + J_common_mode_proxy
```

Important bug found and fixed during this implementation:

- First version returned pre-step currents from the alpha-beta model, so the smoke showed zero current and zero switching. This made the comparison meaningless.
- First version had no flux-building objective; at zero initial flux all torque candidates looked similar and the optimizer selected zero vector. This is a real induction-motor control pitfall.
- The current version adds a flux objective and zero-vector penalty during low-flux startup.

## Neural Twin And Event Feedback

Implemented:

- physics alpha-beta twin
- residual envelope
- uncertainty scalar
- confidence estimate
- event-triggered measurement request
- sparse feedback usage metric

Current feedback policy:

```text
sample if periodic interval expires
or speed error is high
or uncertainty is high
or residual envelope is high
```

Not claimed:

- full no-feedback operation
- sensorless production mode
- conformal uncertainty
- trained ensemble twin

## Host-Level Monte Carlo Smoke

Command run:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 100 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc100.json
```

Scope:

- `N = 100`
- short host simulation only
- parameter randomization: Rs, Rr, Lm, J, B
- compared proxies:
  - `protected_ai_pwm_h1_proxy`
  - `fcs_mpc_one_step_proxy`
  - `safe_neural_horizon_pwm_h2`

The run is a smoke/diagnostic study, not final control-performance evidence.
The simulated time is intentionally short to keep weak-hardware iterations cheap.
Non-quick mode also includes H=3 and H=4 smoke variants:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --mc 3 --steps 40 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_h4_smoke.json
```

Matrix mode adds scenario, ablation, Pareto, and fault-injection summaries:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 5 --steps 80 --out-json .tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.json
python tools/build_safe_neural_horizon_pwm_report.py --input-json .tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.json --out-md .tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.md
```

| Controller | Mean speed error | Mean current | Max current mean | Switch events mean | Feedback usage | Fallback mean | Fault latch mean | Safety violations | Failure count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protected_ai_pwm_h1_proxy | 84.043 | 2.007 | 3.704 | 30.99 | 0.983 | 9.56 | 0.53 | 0 | 6 |
| fcs_mpc_one_step_proxy | 83.909 | 1.664 | 2.595 | 47.96 | 1.000 | 3.02 | 0.00 | 0 | 0 |
| safe_neural_horizon_pwm_h2 | 83.825 | 1.861 | 3.269 | 42.69 | 0.983 | 8.12 | 0.00 | 0 | 0 |

Preliminary reading:

- `safe_neural_horizon_pwm_h2` has the best speed-error proxy in this short smoke.
- It uses less feedback than the one-step FCS proxy.
- It switches less than the one-step FCS proxy.
- It has higher current stress than the one-step FCS proxy.
- It does not yet prove superiority over FOC-SVM, DTC-SVM, or a tuned production FCS-MPC.
- Safety waveform violations were zero in this host-level test.

## Host-Level Scenario Matrix Smoke

Command run:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 5 --steps 80 --out-json .tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.json
```

Scope:

- `N = 5` per scenario/controller pair
- scenarios:
  - `start_no_load`
  - `start_with_load`
  - `load_step`
  - `load_shed`
  - `reverse`
  - `low_speed`
  - `dc_sag`
  - `sensor_dropout`
- proxy baselines:
  - `foc_svm_key_proxy`
  - `fcs_mpc_one_step_proxy`
  - `dtc_hysteresis_proxy`
  - `dtc_svm_proxy`
  - `deadbeat_current_proxy`
  - `sensorless_adaptive_foc_proxy`

Important limitation:

- These are host-level proxies used to expose trade-offs and bugs. They are not yet the final strong baselines required for a paper claim.

Observed pattern in the `MC=5` matrix:

- `safe_neural_horizon_pwm_h4_sparse` often reduces feedback and switching, but it can increase current stress and fallback/fault events. This is useful, not a failure of the study: sparse/horizon control must be current-constrained harder before it can be promoted.
- `fcs_mpc_one_step_proxy` generally keeps current low but switches more frequently and uses dense feedback.
- `foc_svm_key_proxy` is a useful conservative proxy with lower switching, but it is not a full tuned FOC-SVM implementation.
- `safe_neural_horizon_pwm_h2` is safer than the current H4 sparse variant in this short matrix; it avoids the H4 current/fallback issue but does not dominate every metric.
- Fault-injection summary reports `all_cases_no_shoot_through = true`.

Report builder:

- `tools/build_safe_neural_horizon_pwm_report.py`
- output used during this audit: `.tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.md`

## Baselines Still Needed

The current comparison is not enough for publication.

Required next baselines:

- replace `foc_svm_key_proxy` with a tuned key-level FOC-SVM with the same inverter/dead-time/min-pulse/current limits
- replace `fcs_mpc_one_step_proxy` with tuned FCS-MPC current/torque/flux baseline
- replace `dtc_hysteresis_proxy` with tuned DTC hysteresis
- replace `dtc_svm_proxy` with tuned DTC-SVM
- replace `deadbeat_current_proxy` with tuned deadbeat predictive current control
- replace `sensorless_adaptive_foc_proxy` with MRAS/EKF/adaptive FOC
- current protected AI-PWM release model

## Required Robust Tests Still Open

Not finished:

- long nominal scenarios
- start with load
- load throw/shed
- reverse
- braking/regeneration
- low speed and zero speed
- field weakening
- DC-link sag
- motor/inverter heating
- sensor noise/delay/failure
- one-current-sensor fault
- OOD/fault injection matrix
- H=3/H=4 full comparison
- thermal and spectral ablations
- Pareto front generation
- publication-grade plots

## MCU/HIL/Bench Status

Current status:

- MCU port: `not done`
- PWM timer binding: `not done`
- ADC binding: `not done`
- gate driver binding: `not done`
- HIL: `not done`
- bench: `not done`

Required before any hardware claim:

- fixed-point or bounded floating-point implementation
- WCET measurement
- PWM timer/dead-time validation by oscilloscope
- ADC timing validation
- hardware current trip validation
- watchdog/fault latch validation
- HIL fault-injection run
- low-voltage current-limited bench run
- real inverter/motor/load-machine A/B run

## Honest Conclusion

This track is now a real host-level research scaffold, not just text.

What is shown:

- alpha-beta induction motor model exists
- two-level inverter vector model exists
- Safety Gateway exists
- no-shoot-through waveform invariant is host-tested
- horizon AI-PWM controller exists
- neural twin and event feedback scaffold exists
- MC=100 smoke runs without safety waveform violations for the new H=2 variant

What is not shown:

- no proof of hardware readiness
- no proof of HIL safety
- no proof of superiority over strong FOC-SVM/DTC-SVM baselines
- no trained domain-randomized neural twin yet
- no final paper-grade Monte Carlo/Pareto/ablation package yet
