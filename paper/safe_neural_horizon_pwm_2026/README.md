# Safe Neural Horizon PWM 2026

This folder contains host-level research evidence for:

`Safe Neural Horizon PWM with Event-Triggered Twin Feedback`

Current tracked package:

- [20260522_host_release](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release)

Status:

- `HOST_SIMULATION_ONLY`
- `hardware_claim = false`
- `host_release_ready = true`
- `host_novelty_claim_supported = true`
- `host_theory_scaffold_ready = true`
- `trace_fft_thd_evidence_ready = true`
- `publication_plots_fft_thd_ready = true`
- `trained_domain_randomized_twin_ready = true`
- `publication_theory_complete = false`
- `hardware_ready = false`
- `strong_baselines_ready = false`

Do not use this package as MCU, HIL, or bench evidence. The package is a reproducible
software/theory release for the new AI-PWM research branch.

Release gate:

```bash
python tools/check_safe_neural_horizon_pwm_release.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_novelty.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_theory.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
```
