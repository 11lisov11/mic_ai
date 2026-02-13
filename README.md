# MIC_AI - цифровой двойник асинхронного электропривода

Репозиторий содержит модель асинхронного двигателя, базовый FOC и алгоритмы MIC (AI) для снижения энергопотребления при сохранении точности управления. Основная цель - воспроизводимые симуляции и корректные графики для научной публикации.

Материалы статьи (Markdown, DOCX, рисунки): `paper/pgups_2026/`.

## Что внутри

- Цифровой двойник АД и инвертора
- Базовый регулятор FOC
- MIC-регулирование на базе AI (в том числе rule-based режим)
- Инструменты для сравнения FOC vs MIC и построения графиков

## Быстрый старт

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Сравнение FOC vs MIC без RL-чекпойнтов (rule-based MIC):

```bash
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

Сравнение с RL (опционально, если есть чекпойнт):

```bash
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

## Физические конфиги (v2)

Для более реалистичной модели (потери инвертора/железа, dead-time, насыщение) доступны:

- `config/env_demo_true_motor1_physical.py`
- `config/env_demo_true_motor2_physical.py`

Тренировочные скрипты (`mic_ai/ai/train_ai_id_ref.py`, `mic_ai/ai/train_ai_voltage.py`) берут диапазоны рандомизации из конфига, если заданы `ai_omega_ref_pu_range` / `ai_load_mult_range`.

## Принятые метрики

- Активная электрическая мощность: `P_эл(t) = v_a i_a + v_b i_b + v_c i_c`
- RMS-ток статора: `I_rms(t) = sqrt((i_a^2 + i_b^2 + i_c^2) / 3)`
- Механическая мощность: `P_мех(t) = omega(t) * M_эл(t)`

## Публикация (PGUPS)

- Исходник статьи: `paper/pgups_2026/article_mic_ieee_vak_pgups.md`
- Рисунки: `paper/pgups_2026/fig/`
- Готовый DOCX: `paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx`

Пересборка DOCX из Markdown (требуются доп. зависимости):

```bash
python -m pip install -r requirements-paper.txt
python tools/build_publication_from_markdown.py --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md --out-docx paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx
```

## Структура репозитория

- `config/` - конфигурации среды и параметров двигателя
- `mic_ai/` - основной пакет (AI, метрики, инструменты)
- `models/` - математическая модель двигателя/инвертора
- `control/` - алгоритмы управления (FOC, V/f, варианты MIC)
- `simulation/` - окружение симуляции
- `tests/` - тесты (используются в CI)
- `paper/` - материалы публикации (статья, рисунки)
- `outputs/` - вычислительные артефакты/результаты прогонов (игнорируются в git)
- `results_run/` - локальные прогоны обучения (игнорируются в git)
- `archive/` - локальный архив/легаси (игнорируется в git)

## Примечания

- RL-чекпойнты в репозиторий не включены.
