# Stage 11 Report (Random Search)

This report summarizes the random-search results for PIDSpeedPolicy and MICRulePolicy.

## PIDSpeedPolicy (pid_speed)
- base candidate: `candidates/examples/pid_default.json`
- seeds: 0, 1, 2 (replicates=3)
- iterations: 20
- baseline mean score: 45.6314
- best mean score: 47.6886
- improvement: +2.0572 (+4.51%)
- best params:
  - kp: 4.0446363524
  - ki: 1.0973986077
  - integrator_limit: 0.5491459506

Artifact: `search_runs/random_search_pid_speed_20251220_011547/search_summary.json`
Persisted baseline: `candidates/examples/pid_tuned_stage11.json`

## MICRulePolicy (mic_rule)
- base candidate: `candidates/examples/mic_default.json`
- seeds: 0, 1, 2 (replicates=3)
- iterations: 60
- baseline mean score: 45.7545
- best mean score: 47.8783
- improvement: +2.1238 (+4.64%)
- best params:
  - kp: 2.2129830207
  - ki: 0.7170407640
  - integrator_limit: 0.7235028213
  - vq_max: 223.6294733162
  - dvq_max: 1629.0294817172
  - tau_action: 0.0451505354
  - k_age: 0.2020041498

Artifact: `search_runs/random_search_mic_rule_20251220_011722/search_summary.json`
Persisted baseline: `candidates/examples/mic_tuned_stage11.json`
