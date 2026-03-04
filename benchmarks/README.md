# Benchmarks

This folder stores baseline `summary.json` files for regression checks.

Current baselines:
- `baseline_summary_ci.json` (CI smoke run)
- `baseline_summary_physical_motor1.json` (physical motor1, t_end=1.2, dt=0.001, scenarios=speed_step,ramp,load_step,start_stop)
- `baseline_summary_physical_motor2.json` (physical motor2, t_end=1.2, dt=0.001, scenarios=speed_step,ramp,load_step,start_stop)
- `step28_ieee_summary_baseline_20260303_ai_config_locked_nodrift.json` (frozen step28 MIC regression baseline for drift guard)

Legacy baselines:
- `legacy/baseline_summary.json` (archived multi-scenario snapshot)

Update workflow:
- Run a trusted benchmark (e.g., `mic_ai.tools.run_benchmark`).
- Review the new `summary.json`.
- Replace the relevant baseline (CI or physical) once the new numbers are accepted.
