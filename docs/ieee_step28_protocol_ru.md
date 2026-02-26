# IEEE Protocol Step28 (замороженный)

Дата фиксации протокола: 2026-02-26.

## Цель

Единый, воспроизводимый протокол сравнения `PI vs FOC vs MIC` для 3 моторов
с расчетом `mean/std/min/max` и worst-case.

## Моторы

- `air56`: `config/env_research_air56_025kw.py`
- `al31`: `config/env_research_al31_4_06kw.py`
- `ao2`: `config/env_research_ao2_32_4_3kw.py`

## Seeds

- `101,202,303,404,505`

## Сценарии

- `speed_step,ramp,load_step,start_stop`

## Метрики (обязательные)

- `avg_power_saving_pct`
- `avg_eta_gain_pct`
- `err_failures`
- `start_stop_power_saving_pct`
- `worst_current_peak_ratio`
- `worst_current_mean_ratio`
- `avg_controller_speed_err`

## Режимы сравнения (оба обязательны)

1. `FOC(encoder)` vs `MIC(sensorless)`.
2. `FOC(sensorless)` vs `MIC(sensorless)`.

## Seed-variation (для честной статистики)

Во всех прогонах step28 включается `--seed-perturbation` в `tools/step27_pipeline.py`.
Это добавляет детерминированные (по seed) вариации параметров модели и шума
измерений, одинаково воспроизводимые между повторами.

## Команда запуска

Использовать только:

- `scripts/run_step28_ieee_protocol.ps1`
- `scripts/run_step28_ieee_protocol.sh`

## Критерии AIR56 (режим 1)

- `avg_power_saving_pct > 0.5%`
- `avg_eta_gain_pct >= 0`
- `err_failures <= 2`
- `start_stop_power_saving_pct >= -0.5%`

## Артефакты

Минимальный набор:

- `step27_per_seed_metrics.csv`
- `step27_stats_motor_controller.csv`
- `step27_final_pi_vs_foc_vs_mic.csv`
- `step27_air56_acceptance.json`
- `step27_reproducibility.json`
- `step27_report.md`

Для step28 дополнительно:

- `step28_ieee_summary.csv`
- `step28_ieee_summary.md`

