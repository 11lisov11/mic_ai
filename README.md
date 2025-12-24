# MIC_AI (асинхронный электропривод)

Этот репозиторий содержит цифровой двойник асинхронного двигателя, базовый FOC и модуль AI‑оптимизации.
Цель: запуск симуляции и построение корректных, читабельных графиков по токам, моменту и мощности.

## Быстрый старт

1) Установить зависимости:

```
python -m pip install -r requirements.txt
```

2) Графики для фиксированных нагрузок (холостой ход и 5 Н·м):

```
python -m mic_ai.tools.drive_characteristics_ai \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-pu 0.8 \
  --load-values 0,5 \
  --t-end 1.2 \
  --dt 0.001 \
  --window-frac 0.25 \
  --speed-tol 0.05 \
  --out-dir outputs/drive_characteristics
```

Выход:
- `outputs/drive_characteristics/load_characteristics.*`
- `outputs/drive_characteristics/working_characteristics.*`

3) Временные ряды для постоянной нагрузки 5 Н·м:

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
  --load-steps "0:5" \
  --out-dir outputs/timeseries_load_5
```

4) Временные ряды для холостого хода:

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
  --load-steps "0:0" \
  --out-dir outputs/timeseries_load_0
```

Дополнительные инструкции: `docs/analysis_tools_ru.md`.

## Один номинальный режим (скорость + момент)

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

## Принятые метрики

- Активная электрическая мощность:
  `P_эл(t) = v_a i_a + v_b i_b + v_c i_c`
- RMS-ток статора:
  `I_rms(t) = sqrt((i_a^2 + i_b^2 + i_c^2) / 3)`
- Механическая мощность:
  `P_мех(t) = omega(t) * M_эл(t)`
