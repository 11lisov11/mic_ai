# Step27 Extended Reproducibility Report

## Runs
- tag=baseline, perturb_level=0.000, out_dir=`C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_run_step27_extended_repro0\extended_repro\runs\baseline`
- tag=perturb_0p2, perturb_level=0.200, out_dir=`C:\mic_theory\.tmp_pytest\pytest-of-11lis\pytest-0\test_run_step27_extended_repro0\extended_repro\runs\perturb_0p2`

## MIC stats (mean/std/min/max/worst)
| run_tag | motor | perturb_level | power_mean | power_std | power_worst | eta_mean | eta_std | eta_worst | err_worst | speed_err_worst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | air56 | 0.000 | -63.300 | 0.000 | -63.300 | -33.807 | 0.000 | -33.807 | 1.00 | 0.632 |
| perturb_0p2 | air56 | 0.200 | -19.924 | 0.000 | -19.924 | -9.834 | 0.000 | -9.834 | 1.00 | 0.323 |

## Stress sweep summary
| perturb_level | motor | power_mean_mean | power_worst_min | eta_mean_mean | eta_worst_min | err_worst_max | speed_err_worst_max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.000 | air56 | -63.300 | -63.300 | -33.807 | -33.807 | 1.00 | 0.632 |
| 0.200 | air56 | -19.924 | -19.924 | -9.834 | -9.834 | 1.00 | 0.323 |
