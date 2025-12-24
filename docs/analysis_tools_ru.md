# Аналитические инструменты для характеристик привода

Этот раздел описывает два основных скрипта для сравнения FOC и MIC AI:
статические характеристики (по нагрузке/скорости) и временные ряды при
переменной нагрузке.

## 1) Нагрузочные и рабочие характеристики

Скрипт: `mic_ai/tools/drive_characteristics_ai.py`.

Пример (FOC vs MIC AI по контрольным точкам нагрузки):

```
python -m mic_ai.tools.drive_characteristics_ai \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-pu 0.8 \
  --load-values 0.0,0.03,0.06,0.09,0.12 \
  --t-end 1.2 \
  --dt 0.001 \
  --window-frac 0.25 \
  --out-dir outputs/drive_characteristics
```

Выходные файлы:
- `outputs/drive_characteristics/load_characteristics.*`
- `outputs/drive_characteristics/working_characteristics.*`
- `outputs/drive_characteristics/*_filtered.csv` (после фильтра по допуску скорости)

## 2) Временные ряды при переменной нагрузке

Скрипт: `mic_ai/tools/timeseries_compare.py`.

Пример (ступенчатый профиль нагрузки):

```
python -m mic_ai.tools.timeseries_compare \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-pu 0.8 \
  --t-end 1.2 \
  --dt 0.001 \
  --load-profile step \
  --load-steps "0:0.0,0.4:0.06,0.8:0.12" \
  --out-dir outputs/timeseries_compare
```

Выходные файлы:
- `outputs/timeseries_compare/timeseries_compare.{png,pdf,svg}`
- `outputs/timeseries_compare/timeseries_foc.csv`
- `outputs/timeseries_compare/timeseries_mic_ai.csv`

## 3) Один номинальный режим (скорость + момент)

Скрипт: `mic_ai/tools/nominal_case.py`.

Пример (номинальная скорость и момент на валу):

```
python -m mic_ai.tools.nominal_case \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-rpm 1450 \
  --load-torque 1.65 \
  --t-end 6.0 \
  --dt 0.001 \
  --out-dir outputs/nominal_case
```

Выходные файлы:
- `outputs/nominal_case/timeseries_compare.{png,pdf,svg}`
- `outputs/nominal_case/summary.csv`

## Принятые метрики

- Активная электрическая мощность:
  `P_эл(t) = v_a i_a + v_b i_b + v_c i_c`
- RMS-ток статора:
  `I_rms(t) = sqrt((i_a^2 + i_b^2 + i_c^2) / 3)`
- Механическая мощность:
  `P_мех(t) = omega(t) * M_эл(t)`
