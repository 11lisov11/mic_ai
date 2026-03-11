# IEEE Figures Directory

This directory stores promoted figure assets used by `paper/ieee_2026/manuscript.md`.

## Promotion workflow

1. Build frozen package:
```bash
python tools/reproduce_ieee_step28.py --package-tag <tag>
```
2. Promote into IEEE figure set:
```bash
python tools/promote_ieee_release.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --pgups-fig-dir paper/pgups_2026/fig \
  --tag <tag>
```

Promotion manifest is written to:
- `paper/ieee_2026/data/release/<tag>/promotion_manifest.json`

## Current canonical set (`20260303_ai_config_locked_nodrift`)

1. `fig1_mic_methodology.png`  
   MIC block diagram.
2. `fig2_pi_foc_mic_power.(png|pdf|svg)`  
   PI vs FOC vs MIC across 3 motors (mode1/mode2).
3. `fig3_air56_working_characteristics.(png|pdf|svg)`  
   AIR56 detailed FOC vs MIC characteristics.
4. `fig4_cross_motor_robustness.(png|pdf|svg)`  
   Cross-motor scenario robustness heatmap.
5. `fig5_training_to_foc.(png|pdf|svg)`  
   Training-to-FOC convergence view.

## Format policy

Keep:
1. `PDF` for submission.
2. `SVG` for vector editing.
3. `PNG` for quick review/preprint.
