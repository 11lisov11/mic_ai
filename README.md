# MIC_AI - цифровой двойник асинхронного электропривода

Репозиторий содержит модель асинхронного двигателя, базовый FOC и алгоритмы MIC (AI) для снижения энергопотребления при сохранении точности управления. Основная цель - воспроизводимые симуляции и корректные графики для научной публикации.

## Что внутри

- Цифровой двойник АД и инвертора
- Базовый регулятор FOC
- MIC-регулирование на базе AI (в том числе rule-based режим)
- Инструменты для сравнения FOC vs MIC и построения графиков

## Быстрый старт

1) Установить зависимости:

```
python -m pip install -r requirements.txt
```

2) Сравнение FOC vs MIC без RL-чекпойнтов (rule-based MIC):

```
python -m mic_ai.tools.drive_characteristics_ai \
  --env-config config/env_demo_true_motor1_nominal.py \
  --mic-id-ref-low 1.0 \
  --mic-id-ref-high 1.4 \
  --mic-id-ref-speed-tol-rel 0.05 \
  --mic-id-ref-omega-min 0.1 \
  --omega-ref-pu 0.8 \
  --load-points 6 \
  --t-end 2.0 \
  --dt 0.001 \
  --window-frac 0.25 \
  --speed-tol 0.05 \
  --out-dir outputs/drive_characteristics_nominal_rule
```

Выход:
- `outputs/drive_characteristics_nominal_rule/load_characteristics.*`
- `outputs/drive_characteristics_nominal_rule/working_characteristics.*`

3) Сравнение с RL (опционально, если есть чекпойнт):

```
python -m mic_ai.tools.drive_characteristics_ai \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint path/to/best_actor.pth \
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

Дополнительные инструкции: `docs/analysis_tools_ru.md`.

## Один номинальный режим (скорость + момент)

Требуется RL-чекпойнт.

```
python -m mic_ai.tools.nominal_case \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint path/to/best_actor.pth \
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

## Структура репозитория

- `config/` - конфигурации среды и параметров двигателя
- `mic_ai/` - основной пакет (AI, метрики, инструменты)
- `simulation/` - окружение симуляции
- `outputs/` - результаты вычислений (игнорируются в git)

## Примечания

- RL-чекпойнты в репозиторий не включены.
