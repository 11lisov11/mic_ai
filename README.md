# MIC_AI

Motor Intelligence Control with Artificial Intelligence (MIC_AI) - экспериментально-исследовательский симуляционный стенд для управления асинхронным двигателем, ориентированный на сравнение AI-управления и классического FOC в dq-представлении и дальнейший перенос на MCU.

В репозитории две тесно связанные части:
- Part A. AI-based Voltage Control (ai_voltage, RL): обучение RL-политики, пайплайн для paper-графиков и сравнение AI vs FOC.
- Part B. MIC_AI Digital Twin Bench (Reproducible Experiments): воспроизводимый стенд, testsuite E1..E7, scoring + leaderboard, JSON runner.

Ключевой вклад: воспроизводимый digital twin bench для управления асинхронным двигателем с фиксированным testsuite и лидербордом, сравнение классических каскадных регуляторов, rule-based MIC и прямых AI-политик по напряжению, учет идентификации/старения и 1:1 путь к MCU.

## Part A. AI-based Voltage Control (ai_voltage, RL)

Поддерживаются два режима управления:
- `AI (RL)`: обучаемая политика, напрямую задающая `v_d/v_q` (вместо PI/FOC и каскадов регуляторов).
- `FOC`: классическое векторное управление с PI-регуляторами.

Есть полноценная симуляция с инвертором, датчиками и нагрузкой для сравнения стратегий в одинаковых условиях (профили `omega_ref`, возмущения, шумы, изменения параметров, ограничения по напряжению/току).

### Ключевые особенности

- Digital Twin: dq-модель двигателя + инвертор + преобразования `abc <-> dq` + модель датчиков/нагрузки.
- Классические контроллеры: V/f и FOC (baseline/fallback).
- AI-агент `ai_voltage`: политика прямого задания `v_d/v_q` напряжений.
- Набор скриптов для генерации графиков и таблиц под paper-результаты (PNG+PDF, русские подписи).

### Установка (Windows)

Если виртуальное окружение уже создано (`.venv/`), достаточно:

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Иначе создайте виртуальное окружение:

`python -m venv .venv`

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

### Обучение `ai_voltage`

`.\.venv\Scripts\python.exe mic_ai/ai/train_ai_voltage.py config/env_demo_true_motor1.py --episodes 600 --episode-steps 200`

Выходы:
- `outputs/demo_ai/episode_logs/`: JSON-логи по эпизодам
- `outputs/demo_ai/plots/`: графики и сводки
- `outputs/demo_ai/checkpoints/`: `best_actor.pth`, `last_actor.pth`

### Графики для статьи (AI vs FOC, IEEE-стиль)

Генерация графиков по сценарию (профили `omega_ref`, возмущения, шумы, ограничения):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_plots_ai_vs_foc --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --window 5 --out-dir outputs/paper_figures_motor1_strict_v7 --voltage-scale 1.25 --disable-noise`

Результат: `outputs/paper_figures_motor1_strict_v7/` (PNG+PDF+JSON+`captions_ru.txt`).

### Пакетная оценка: множественные прогоны (seeds), доверительный интервал 95%

Оценка на эталонном наборе сцен (AI: среднее и 95% ДИ по метрикам):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_eval_suite --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --disable-noise --out-dir outputs/paper_eval_suite_motor1_s10`

Оценка на наборе с шумами (сравнение AI и FOC в одинаковых условиях):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_eval_suite --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --sigma-omega 0.05 --sigma-i 0.03 --out-dir outputs/paper_eval_suite_motor1_s10_noise_fair`

Результаты в `outputs/paper_eval_suite_motor1_s10/`:
- `summary_overall.csv`: сводные метрики (AI mean/std/ci95 vs FOC)
- `summary_by_stage.csv`: метрики по стадиям
- `fig_stage_Irms.*`, `fig_stage_Pin_pos.*`, `fig_stage_speed_error.*`: графики по стадиям (AI: 95% ДИ)
- `run_meta.json`: метаданные прогона (сценарии и параметры)

### Таблицы для статьи (LaTeX/Markdown)

Сборка таблиц по метрикам для вставки в статью (LaTeX) или в Markdown:

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_tables --summary-overall outputs/paper_eval_suite_motor1_s10/summary_overall.csv --summary-by-stage outputs/paper_eval_suite_motor1_s10/summary_by_stage.csv --summary-cases outputs/paper_param_robustness_motor1_s5_no_noise/summary_cases.csv --out-dir outputs/paper_tables --prefix motor1`

### Параметрическая робастность: вариации параметров двигателя/нагрузки

Проверка чувствительности по параметрам и нагрузке (например: `Rs +10%`, `Lm -10%`, `J +20%`, `T_load +20%`, а также worst-case):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_param_robustness --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 5 --seed0 0 --disable-noise --out-dir outputs/paper_param_robustness_motor1_s5_no_noise`

Результат: `outputs/paper_param_robustness_motor1_s5_no_noise/`

- `summary_cases.csv`: сводные сравнения AI vs FOC по кейсам (mean + 95% ДИ для AI)
- `case_*/summary_overall.csv`, `case_*/summary_by_stage.csv`: метрики по кейсам
- `case_*/run_meta.json`: метаданные по каждому кейсу

### Manifest экспериментов (для воспроизводимости)

Автоматическая выгрузка описания экспериментов (настройки, параметры, профили `omega_ref`, шумы, метрики, затраты вычислений):

`.\.venv\Scripts\python.exe -m mic_ai.tools.paper_experiment_manifest --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --voltage-scale 1.25 --seeds 10 --seed0 0 --disable-noise --out-dir outputs/paper_manifest_motor1`

Результат: `outputs/paper_manifest_motor1/manifest_motor1.json` и `outputs/paper_manifest_motor1/manifest_motor1.md`.

### Структура репозитория (Part A)

- `models/`: dq-модель двигателя, инвертор, преобразования
- `control/`: V/f, FOC
- `simulation/`: окружение и сценарии
- `mic_ai/ai/`: обучение и агенты
- `mic_ai/tools/`: утилиты, графики, paper-скрипты
- `outputs/`: результаты прогонов

## Part B. MIC_AI Digital Twin Bench (Reproducible Experiments)

The Digital Twin Bench is a reproducible experiment suite for induction motor control with identification, safety supervision, scoring, and a JSON candidate runner. A short methodology-style overview lives in `docs/bench_overview.md`.

### Who controls what

| Strategy | Controls | Uses model | Uses aging | Learning |
| --- | --- | --- | --- | --- |
| V/f | voltage magnitude + frequency | no | no | no |
| PID (speed) | speed loop (PI via FOC) | yes | no | no |
| FOC | speed + current loops | yes | no | no |
| MIC (rule) | speed logic + gain scheduling | yes | yes | no |
| AI (RL) | `v_d/v_q` directly | no (model-free) | optional | yes |

All classical controllers (PID/FOC/MIC) share the same inner current regulation layer; differences are in the outer-loop logic. The AI voltage policy bypasses the inner current loop by design.

Stub policies such as `nn_stub` and `quick_stub` intentionally emit constant actions to smoke-test the pipeline; identical leaderboard scores are expected.

Scores are comparable only within identical experiment conditions (testsuite, dt, limits, noise, identification).

### Digital Twin Bench (WIP)

New bench scaffold (in progress) for reproducible digital twin experiments:
- `sim/` (servo load model added in Stage 1)
- `bench/`, `candidates/`, `docs/`, `logs/` (bench core, candidates, docs, logs)

### Stages 1-9: quick checks and demos

Stage 1:
- quick check: `python -m pytest tests/test_load_model.py`
- example run: `python sim/load_servo.py`

Stage 2:
- quick check: `python -m pytest tests/test_safety.py`

Stage 3:
- demo (bench orchestrator + logging): `python main.py --demo one_experiment`
- tests: `python -m pytest tests/test_orchestrator_logging.py`

Stage 4:
- demo (testsuite E1..E7): `python main.py --demo testsuite`
- tests: `python -m pytest tests/test_testsuite.py`

Stage 5:
- demo (score + leaderboard): `python main.py --demo score_testsuite --policy-id foc_baseline`
- tests: `python -m pytest tests/test_scoring_leaderboard.py`

Stage 6:
- demo (candidate runner): `python candidates/runner.py --candidate candidates/examples/vf_default.json`
- batch demo: `python candidates/runner.py --batch-dir candidates/examples`
- tests: `python -m pytest tests/test_candidate_runner.py`

Stage 7:
- demo (identify only): `python main.py --demo identify_only --t-end 2.0 --dt 0.001`
- demo (testsuite + identification): `python main.py --demo testsuite --with-identification --t-end 0.2 --dt 0.01`
- demo (candidate runner + identification): `python candidates/runner.py --candidate candidates/examples/foc_default.json --with-identification`
- tests: `python -m pytest tests/test_identification_basic.py tests/test_identification_drift_response.py`

Stage 8:
- demo (full testsuite + identification + scoring): `python main.py --demo score_testsuite_full --t-end 0.2 --dt 0.001`
- tests: `python -m pytest tests/test_full_suite_smoke.py`

Stage 9:
- demo (PID baseline): `python candidates/runner.py --candidate candidates/examples/pid_default.json`
- demo (MIC baseline + identification): `python candidates/runner.py --candidate candidates/examples/mic_default.json --with-identification`
- demo (batch baselines + identification): `python candidates/runner.py --batch-dir candidates/examples --with-identification`
- tests: `python -m pytest tests/test_baseline_policies.py`

### Stage 11: Auto search (random)

PID search:
`python search/random_search.py --policy pid_speed --iters 20 --base-candidate candidates/examples/pid_default.json --seed 0 --with-identification`

MIC search:
`python search/random_search.py --policy mic_rule --iters 20 --base-candidate candidates/examples/mic_default.json --seed 0 --with-identification`

Each run writes `search_summary.json` with baseline score, best score, percent improvement, and best parameters.
Stage 11 results summary: `docs/stage11_report.md`.

### Tuned baselines (Stage 11)

Persisted tuned baselines for downstream stages and papers:
- `candidates/examples/pid_tuned_stage11.json` (PIDSpeedPolicy)
- `candidates/examples/mic_tuned_stage11.json` (MICRulePolicy)

These are derived from Stage 11 random search under the reference conditions hash and should be used as the new baseline references for Stage 13+ comparisons.

### Project summary (MIC_AI Digital Twin Bench)

MIC_AI = Motor Intelligence Control with Artificial Intelligence. This repository hosts a reproducible digital twin bench
for an induction motor + inverter with servo load, sensors, and safety supervision. The bench adds identification
(Rs estimate + aging), a fixed testsuite E1..E7, scoring + leaderboard, and a JSON candidate runner for baseline and
research policies. The goal is PC-first simulation with a 1:1 path to embedded hardware. Logs are stored as NPZ
(timeseries) + JSON (metadata and metrics).
