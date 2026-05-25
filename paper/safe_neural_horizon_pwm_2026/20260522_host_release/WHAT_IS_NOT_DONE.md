# Safe Neural Horizon PWM Open Items

- Tune the host key-level FOC-SVM, FCS-MPC, and DTC hysteresis baselines and replace remaining host proxy baselines with tuned DTC-SVM, deadbeat, and sensorless/adaptive FOC.
- Add publication-grade long-run metrics: THD, FFT torque, switching loss, conduction loss, thermal imbalance, EMI/common-mode proxy.
- Run MC=500..1000 after strong baselines are ready.
- Train or identify the neural twin with domain randomization and multi-step losses.
- Keep `publication_theory_complete=false` until strong baselines, trained twin, MC=500..1000, and FFT/THD trace evidence are present.
- Add fixed-point or bounded floating-point MCU implementation plus WCET.
- Add HIL, oscilloscope gate timing, current trip, watchdog, and bench validation.
- Do not claim hardware-ready status until real MCU/HIL/bench evidence exists.
