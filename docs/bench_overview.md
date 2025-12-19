# MIC_AI Digital Twin Bench Overview

## Scope and goals
The bench provides a reproducible, PC-first environment for comparing induction motor control strategies under the same scenarios, sensors, and safety constraints. It targets a 1:1 path to embedded firmware and keeps experiment artifacts for auditability.

## Architecture summary
- Plant: induction motor + inverter + servo load modeled in dq with abc/dq transforms.
- Sensors: configurable noise, quantization, and delay.
- Safety: current, voltage, and speed limits with trip flags.
- Identification: Rs estimation with drift/aging hooks.
- Evaluation: fixed testsuite E1..E7 plus scoring and leaderboard.
- Runner: JSON-defined candidates for baselines and research policies.

## Stages 1-9 (progressive scaffold)
Stage 1: load model sanity check.
Stage 2: safety supervision.
Stage 3: orchestrator + logging.
Stage 4: fixed testsuite E1..E7.
Stage 5: scoring and leaderboard aggregation.
Stage 6: JSON candidate runner.
Stage 7: identification integration and drift response.
Stage 8: full suite with identification + scoring.
Stage 9: baseline policy variants (PID, MIC, etc).

## Testsuite E1..E7
The testsuite is a fixed set of scenarios covering speed steps, load disturbances, and sensor effects. It is designed for fair comparisons: all policies see the same references, disturbances, and safety limits.

## Identification and aging
Identification runs estimate Rs and allow optional aging factors to influence controller gains. The MIC rule policy can apply gain scheduling using the aging signal, while other controllers keep fixed gains. This makes adaptation explicit and testable.

## Scoring and leaderboard
Each case produces metrics such as IAE, overshoot, settling time, RMS current, smoothness, and safety trips. Scores are aggregated into a suite score and written to leaderboard.json along with per-case details.

## Reproducibility artifacts
Every run logs timeseries as NPZ and metadata/metrics as JSON. Test runs can be replayed by pointing to the logged config and seeds, enabling apples-to-apples comparisons across policies and environments.
