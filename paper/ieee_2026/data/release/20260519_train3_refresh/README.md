# 20260519 Train3 Refresh Release

- research_refresh_complete: `true`
- hardware_deploy_complete: `false`
- summary_source: `outputs/train3_fullprog_20260519/final_selected_strict_recheck_20260519/selected_recheck_summary.json`

| Motor | Decision | Power saving | Eta gain | Start-stop | Err | Envelope fails |
|---|---|---:|---:|---:|---:|---:|
| air56 | keep_canonical_baseline | 1.072% | 0.112% | 2.140% | 0 | 0 |
| al31 | promote_training_checkpoint | 3.455% | 0.003% | 13.736% | 0 | 0 |
| ao2 | keep_canonical_baseline | 0.512% | 1.724% | 0.000% | 0 | 0 |

## Checkpoint Policy

AIR56 and AO2 remain on the accepted canonical baselines. AL31 is promoted from the 2026-05-19 fine-tune run.

Large checkpoint binaries are intentionally not committed; use the manifest hashes plus reproduce commands to regenerate them.
