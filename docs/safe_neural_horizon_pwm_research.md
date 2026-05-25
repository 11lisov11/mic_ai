# Safe Neural Horizon PWM Research Track

Date: `2026-05-25`
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

## Novelty Claim

The current host-level novelty claim is precise:

`SNH-PWM` is a distinct control architecture that combines:

- event-triggered neural-twin feedback;
- deterministic neural cost shaping;
- finite-horizon legal inverter-vector search;
- feedback/switching/risk costs inside the vector-selection objective;
- a Safety Gateway that enforces both no-shoot-through and no-direct-HIGH-to-LOW transition paths.

This is not just a neural replacement for a PI loop:

- compared with FOC-SVM, SNH-PWM does not first synthesize continuous `v_d/v_q` references and then run SVM; it searches legal inverter vectors directly;
- compared with one-step FCS-MPC, SNH-PWM adds neural cost shaping, event-triggered feedback economy, and a mandatory protected AI-PWM gateway;
- compared with the earlier protected AI-PWM H1 variant, SNH-PWM adds horizon search, twin uncertainty, ablation variants, and explicit feedback-usage optimization.

Allowed claim from the tracked evidence:

- a new host-simulated control architecture exists and is machine-checkable.

Not allowed yet:

- publication-grade superiority over tuned FOC-SVM/FCS-MPC/DTC baselines;
- MCU/HIL/bench readiness;
- full no-feedback control;
- trained neural-twin optimality.

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

The tested timing-path invariant is:

```text
HIGH_ON -> BOTH_OFF -> LOW_ON
LOW_ON  -> BOTH_OFF -> HIGH_ON
```

Direct adjacent transitions are rejected by the host detector:

```text
HIGH_ON -> LOW_ON
LOW_ON  -> HIGH_ON
```

Host-level test:

```bash
python -m pytest -q tests/test_safe_neural_horizon_pwm.py
```

Current result:

```text
20 passed
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
- host-release packager smoke
- release-novelty audit smoke

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
- The Safety Gateway originally checked only shoot-through states, not direct HIGH-to-LOW timing-path violations. The current version adds `has_direct_leg_transition()` and `DEADTIME_FAULT` checks.
- Controller thermal/loss metrics originally used the planned sequence before Safety Gateway fallback. The current version reports applied losses/switching after the gateway decision.
- The release checker originally trusted whatever files were listed in the manifest. The current version requires all essential release artifacts and rejects unsafe manifest paths.

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
- compared rows:
  - `protected_ai_pwm_h1_proxy`
  - `fcs_mpc_one_step_baseline`
  - `foc_svm_key_baseline`
  - `dtc_hysteresis_baseline`
  - `dtc_svm_baseline`
  - `safe_neural_horizon_pwm_h2`

The run is a smoke/diagnostic study, not final control-performance evidence.
The simulated time is intentionally short to keep weak-hardware iterations cheap.
Non-quick mode also includes H=3 and H=4 smoke variants:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --mc 3 --steps 40 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_h4_smoke.json
```

Matrix mode adds scenario, ablation, Pareto, and fault-injection summaries.
The tracked host release uses the full host matrix:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 3 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json
python tools/package_safe_neural_horizon_pwm_release.py --input-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json --out-dir paper/safe_neural_horizon_pwm_2026/20260522_host_release --tag 20260522_safe_neural_horizon_pwm_host_release
```

| Controller | Mean speed error | Mean current | Max current mean | Switch events mean | Feedback usage | Fallback mean | Fault latch mean | Safety violations | Failure count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protected_ai_pwm_h1_proxy | 84.322 | 1.532 | 2.576 | 42.11 | 0.983 | 3.40 | 0.00 | 0 | 0 |
| fcs_mpc_one_step_baseline | 82.544 | 2.547 | 3.355 | 22.36 | 1.000 | 0.01 | 0.00 | 0 | 0 |
| foc_svm_key_baseline | 83.844 | 1.011 | 2.568 | 37.38 | 1.000 | 0.00 | 0.00 | 0 | 0 |
| dtc_hysteresis_baseline | 84.791 | 0.309 | 0.683 | 142.76 | 1.000 | 0.45 | 0.00 | 0 | 0 |
| dtc_svm_baseline | 84.695 | 0.578 | 2.008 | 17.98 | 1.000 | 0.00 | 0.00 | 0 | 0 |
| safe_neural_horizon_pwm_h2 | 84.235 | 1.757 | 3.122 | 45.22 | 0.983 | 5.37 | 0.00 | 0 | 0 |

Preliminary reading:

- The new `fcs_mpc_one_step_baseline` is strongest in this short MC=100 smoke for speed error and switching, but it uses higher current than FOC-SVM and SNH-H2.
- The new `foc_svm_key_baseline` is strongest for mean current and fallback count in this smoke.
- The new `dtc_hysteresis_baseline` is safe after current-penalty tuning, but it pays with very high switching and worse speed error in this smoke.
- The new `dtc_svm_baseline` is safe in MC=100 and cuts switching strongly relative to DTC hysteresis, but it is still not publication tuned and has worse speed error than FCS-MPC/FOC-SVM in this short smoke.
- `safe_neural_horizon_pwm_h2` still uses slightly less feedback than dense FOC-SVM/FCS rows, but it does not dominate the new FOC-SVM/FCS-MPC baselines in this smoke.
- It must now be judged against a real one-step FCS-MPC baseline, not the older weight-tuned proxy.
- It does not yet prove superiority over tuned production-grade FOC-SVM, DTC-SVM, or FCS-MPC; this is exactly why the baseline tuning phase still matters.
- Safety waveform violations were zero in this host-level test.

## Host-Level Scenario Matrix Release

Command run:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 3 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json
```

Scope:

- `N = 3` per scenario/controller pair
- scenarios:
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
- comparison baselines:
  - `foc_svm_key_baseline` (separate host key-level PI/current/SVM baseline; not final publication tuned)
  - `fcs_mpc_one_step_baseline` (separate host one-step current/torque/flux FCS-MPC baseline; not final publication tuned)
  - `dtc_hysteresis_baseline` (separate host torque/flux hysteresis DTC baseline; not final publication tuned)
  - `dtc_svm_baseline` (separate host torque/flux voltage-reference DTC-SVM baseline; not final publication tuned)
  - `deadbeat_current_proxy`
  - `sensorless_adaptive_foc_proxy`

Important limitation:

- FOC-SVM, one-step FCS-MPC, DTC hysteresis, and DTC-SVM are now separate host baselines, but not yet tuned publication-grade baselines.
- Deadbeat, sensorless/adaptive FOC, and protected H1 comparison rows are still host-level proxies used to expose trade-offs and bugs.

Tracked release package:

- [20260522_host_release](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release)
- [HOST_RELEASE_MANIFEST.json](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/HOST_RELEASE_MANIFEST.json)
- [HOST_ACCEPTANCE_SUMMARY.json](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/HOST_ACCEPTANCE_SUMMARY.json)
- [safe_neural_horizon_pwm_article_draft.md](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/safe_neural_horizon_pwm_article_draft.md)
- [WHAT_IS_NOT_DONE.md](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/WHAT_IS_NOT_DONE.md)
- aggregate figures:
  - [fig_speed_error_vs_current.svg](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/figures/fig_speed_error_vs_current.svg)
  - [fig_feedback_vs_switching.svg](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/figures/fig_feedback_vs_switching.svg)
  - [fig_h2_scenario_speed_error.svg](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/figures/fig_h2_scenario_speed_error.svg)

Host-release gate:

```bash
python tools/check_safe_neural_horizon_pwm_release.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_novelty.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_theory.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
```

Current host-release result:

```text
host_release_ready = true
hardware_ready = false
strong_baselines_ready = false
host_novelty_claim_supported = true
host_theory_scaffold_ready = true
publication_theory_complete = false
```

The tracked release also contains:

- [safe_neural_horizon_pwm_theory_completion_audit.json](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/safe_neural_horizon_pwm_theory_completion_audit.json)
- [safe_neural_horizon_pwm_mc100_smoke.json](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release/safe_neural_horizon_pwm_mc100_smoke.json)

Interpretation:

- `host_theory_scaffold_ready = true` means the new architecture has a reproducible host model, safety gate, horizon controller, scenario matrix, first MC=100 smoke, report artifacts, and honest claim boundaries.
- `publication_theory_complete = false` remains intentional until the FOC-SVM key baseline is tuned, remaining proxy baselines are replaced, the neural twin is trained/identified, MC reaches publication scale, and long-run FFT/THD/trace evidence exists.

Observed pattern in the host matrix:

- `safe_neural_horizon_pwm_h4_sparse` often reduces feedback and switching, but it can increase current stress and fallback/fault events. This is useful, not a failure of the study: sparse/horizon control must be current-constrained harder before it can be promoted.
- `fcs_mpc_one_step_baseline` is now a separate host FCS-MPC baseline: it predicts one step for each legal vector and minimizes current/torque/flux/switching/loss cost through the same Safety Gateway.
- `foc_svm_key_baseline` is now a separate host key-level FOC-SVM baseline with speed PI, dq current PI, nearest legal vector/SVM selection, and the same Safety Gateway. It is stronger than the old proxy, but still not a final tuned publication baseline.
- `dtc_hysteresis_baseline` is now a separate host DTC hysteresis baseline: torque and flux hysteresis comparators request increase/decrease/hold, then a legal vector is selected through the same Safety Gateway.
- `dtc_svm_baseline` is now a separate host DTC-SVM baseline: torque and flux PI loops synthesize a stator-flux-frame voltage reference, then nearest legal-vector SVM selection is passed through the same Safety Gateway.
- `safe_neural_horizon_pwm_h2` is safer than the current H4 sparse variant in this short matrix; it avoids the H4 current/fallback issue but does not dominate every metric.
- Fault-injection summary reports `all_gateway_cases_no_shoot_through = true`; the raw shoot-through detector triggers on deliberately illegal raw gate emulation; the dead-time path detector distinguishes direct HIGH/LOW transitions from valid BOTH_OFF paths.

Report builder:

- `tools/build_safe_neural_horizon_pwm_report.py`
- output used during this audit: `.tmp_pytest/safe_neural_horizon_pwm_matrix_mc5.md`

## Baselines Still Needed

The current comparison is not enough for publication.

Required next baselines:

- tune and stress-test `foc_svm_key_baseline` into a publication-grade key-level FOC-SVM baseline with the same inverter/dead-time/min-pulse/current limits
- tune and stress-test `fcs_mpc_one_step_baseline` into a publication-grade FCS-MPC current/torque/flux baseline
- tune and stress-test `dtc_hysteresis_baseline` into a publication-grade DTC hysteresis baseline
- tune and stress-test `dtc_svm_baseline` into a publication-grade DTC-SVM baseline
- replace `deadbeat_current_proxy` with tuned deadbeat predictive current control
- replace `sensorless_adaptive_foc_proxy` with MRAS/EKF/adaptive FOC
- current protected AI-PWM release model

## Robust Tests Status

Host-level matrix now covers the full 30-scenario TZ set plus one explicit `sensor_dropout` stress case:

- start/no-load, start/load, ramp, load throw/shed, reverse, braking, regeneration;
- low speed, zero speed, field weakening, overload, DC-link sag;
- motor heating, inverter heating, Rs/Rr/Lm/J errors;
- random, periodic, and shock load;
- two-mass proxy;
- current/speed sensor noise, sensor delay, speed sensor failure, current sensor failure;
- OOD, runtime fault injection, and sensor dropout.

Still not publication-grade:

- long-duration traces with FFT/THD;
- tuned strong classical baselines;
- publication-scale MC `N=500..1000`;
- full thermal/spectral ablations;
- final Pareto fronts after tuning FOC-SVM/FCS-MPC and replacing the remaining proxy baselines;
- MCU/HIL/bench timing evidence.

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
- no-direct-HIGH-to-LOW transition-path invariant is host-tested
- horizon AI-PWM controller exists
- neural twin and event feedback scaffold exists
- MC=100 smoke runs without safety waveform violations for the new H=2 variant
- tracked novelty audit supports only the host-level distinct-architecture claim
- tracked theory-completion audit supports `host_theory_scaffold_ready = true`, not publication-grade completion

What is not shown:

- no proof of hardware readiness
- no proof of HIL safety
- no proof of superiority over strong FOC-SVM/DTC-SVM baselines
- no trained domain-randomized neural twin yet
- no final paper-grade Monte Carlo/Pareto/ablation package yet
