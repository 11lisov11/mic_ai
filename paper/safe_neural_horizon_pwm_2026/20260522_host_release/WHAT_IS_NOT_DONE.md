# Safe Neural Horizon PWM Open Items

- Tune the host key-level FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat, and sensorless/adaptive FOC baselines to publication-grade strength.
- Expand the host trace/FFT/THD-like package after baseline tuning; current evidence is simulation-only and not hardware power-analyzer THD.
- Run MC=500..1000 after strong baselines are ready.
- Train or identify the neural twin with domain randomization and multi-step losses.
- Keep `publication_theory_complete=false` until strong baselines, trained twin, and MC=500..1000 are present; host trace evidence alone is not enough.
- Add fixed-point or bounded floating-point MCU implementation plus WCET.
- Add HIL, oscilloscope gate timing, current trip, watchdog, and bench validation.
- Do not claim hardware-ready status until real MCU/HIL/bench evidence exists.
