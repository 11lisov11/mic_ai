# Safe Neural Horizon PWM Open Items

- Expand bounded baseline tuning to wider MC/scenario sweeps before any journal superiority claim.
- Expand the host trace/FFT/THD-like package after future model/controller changes; current evidence is simulation-only and not hardware power-analyzer THD.
- Re-run/scale MC=500..1000 after any controller, model, or tuning-grid change; current MC500 is valid for this host release.
- Replace the theta-conditioned host twin evidence with a production online parameter identifier before MCU/HIL/bench claims.
- Keep hardware readiness false: host theory completion does not replace MCU/HIL/bench validation.
- Add fixed-point or bounded floating-point MCU implementation plus WCET.
- Add HIL, oscilloscope gate timing, current trip, watchdog, and bench validation.
- Do not claim hardware-ready status until real MCU/HIL/bench evidence exists.
