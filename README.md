# MIC_AI

Motor Intelligence Control with Artificial Intelligence (MIC_AI) — исследовательско-инженерный проект по самообучающемуся управлению асинхронным двигателем (dq-модель) с прицелом на перенос на MCU.

Сравнение проводится между:
- `AI (RL)`: прямое управление напряжениями `v_d/v_q` (без PI/FOC в контуре управления).
- `FOC`: классическое векторное управление с PI-регуляторами.

Цель визуализаций и таблиц — строгое воспроизводимое сравнение при одинаковых условиях эксперимента (профили `ω_ref`, нагрузки, длительность эпизодов, ограничения по току/напряжению).

## Что внутри

- Digital Twin: dq-двигатель + инвертор + преобразования `abc <-> dq` + механика нагрузки.
- Базовые алгоритмы: V/f и FOC (baseline/fallback).
- AI-контур `ai_voltage`: агент управляет `v_d/v_q` напрямую.
- Инструменты для отчётов, сравнения и paper-графиков (PNG+PDF, русские подписи).

## Установка (Windows)

Если виртуальное окружение уже создано (`.venv/`), достаточно:

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Если окружения нет:

`python -m venv .venv`

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

## Обучение `ai_voltage`

`.\.venv\Scripts\python.exe mic_ai/ai/train_ai_voltage.py config/env_demo_true_motor1.py --episodes 600 --episode-steps 200`

Артефакты:
- `outputs/demo_ai/episode_logs/` — JSON-логи по эпизодам
- `outputs/demo_ai/plots/` — графики обучения
- `outputs/demo_ai/checkpoints/` — `best_actor.pth`, `last_actor.pth`

## Научные графики (AI vs FOC, IEEE-стиль)

Строгое сравнение (одинаковые `ω_ref`, нагрузка, длительность эпизода, ограничения):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_plots_ai_vs_foc --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --window 5 --out-dir outputs/paper_figures_motor1_strict_v7 --voltage-scale 1.25 --disable-noise`

Вывод: `outputs/paper_figures_motor1_strict_v7/` (PNG+PDF+JSON+`captions_ru.txt`).

## Пакет для статьи: многократные прогоны (seeds), таблицы и 95% ДИ

Запуск без шумов измерений (AI: среднее и 95% доверительный интервал по сид-прогонам):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_eval_suite --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --disable-noise --out-dir outputs/paper_eval_suite_motor1_s10`

Режим с шумами измерений (шум применяется к AI-наблюдениям и входам FOC-контроллера в симуляции):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_eval_suite --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --sigma-omega 0.05 --sigma-i 0.03 --out-dir outputs/paper_eval_suite_motor1_s10_noise_fair`

Артефакты в `outputs/paper_eval_suite_motor1_s10/`:
- `summary_overall.csv` — сводная таблица метрик (AI mean/std/ci95 vs FOC)
- `summary_by_stage.csv` — сводка по стадиям
- `fig_stage_Irms.*`, `fig_stage_Pin_pos.*`, `fig_stage_speed_error.*` — графики по стадиям (AI: 95% ДИ)
- `run_meta.json` — параметры запуска (для воспроизводимости)

## Экспорт таблиц (LaTeX/Markdown)

Генерация таблиц для вставки в статью (LaTeX) и для предпросмотра (Markdown):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_tables --summary-overall outputs/paper_eval_suite_motor1_s10/summary_overall.csv --summary-by-stage outputs/paper_eval_suite_motor1_s10/summary_by_stage.csv --summary-cases outputs/paper_param_robustness_motor1_s5_no_noise/summary_cases.csv --out-dir outputs/paper_tables --prefix motor1`

## Робастность: вариации параметров двигателя/нагрузки

Прогон набора сценариев с вариацией параметров (пример: `Rs +10%`, `Lm -10%`, `J +20%`, `T_load +20%`, комбинированный worst-case):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_param_robustness --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 5 --seed0 0 --disable-noise --out-dir outputs/paper_param_robustness_motor1_s5_no_noise`

Вывод: `outputs/paper_param_robustness_motor1_s5_no_noise/`

- `summary_cases.csv` — сравнение AI vs FOC по всем кейсам (mean + 95% ДИ для AI)
- `case_*/summary_overall.csv`, `case_*/summary_by_stage.csv` — таблицы по каждому кейсу
- `case_*/run_meta.json` — параметры и масштабирование кейса

## Manifest эксперимента (воспроизводимость)

Экспорт “паспортных” данных запуска (конфиг, ограничения, профиль `ω_ref`, сиды, версии, хэш чекпойнта):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_experiment_manifest --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --disable-noise --out-dir outputs/paper_manifest_motor1`

Вывод: `outputs/paper_manifest_motor1/manifest_motor1.json` и `outputs/paper_manifest_motor1/manifest_motor1.md`.

## Структура репозитория (кратко)

- `models/` — dq-модель, инвертор, преобразования
- `control/` — V/f, FOC
- `simulation/` — окружение и симуляция
- `mic_ai/ai/` — окружения и агенты
- `mic_ai/tools/` — отчёты, сравнения, paper-графики
- `outputs/` — результаты экспериментов
