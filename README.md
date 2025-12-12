# MIC_AI — Motor‑Independent Cognitive AI

Проект моделирует асинхронный двигатель и управление им (V/f и FOC), а также содержит подсистему идентификации параметров и AI‑агентов, которые учатся улучшать управление после идентификации.

Старый файл `README` оставлен как legacy; актуальная инструкция здесь.

## Быстрый старт

1. Установить зависимости:
```bash
python -m pip install -r requirements.txt
```

2. Запустить симуляцию:
```bash
python main.py --mode foc --scenario speed_step --t-end 0.5 --dt 0.0001 --save-prefix run --no-plot
```
Результаты сохраняются в `outputs/results/data_N.npz`.

3. Построить графики по сохранённому прогону:
```bash
python -c "from outputs.plots import plot_run; plot_run('outputs/results/data_1.npz')"
```

## Идентификация параметров двигателя

Идентификация работает в среде прямого задания dq‑напряжений:

```bash
python mic_ai/tools/identify_motor.py --env-config config/env_demo_true.py --output out/ident_simple.json
python mic_ai/tools/full_identify_motor.py --env-config config/env_demo_true.py --output out/ident_full.json
```

Результат содержит оценённые `Rs, Rr, Ls, Lr, Lm, J, B`.

## Обучение AI после идентификации

Главная идея: сначала идентифицируем мотор, затем **применяем оценённые параметры к конфигу** и уже на нём обучаем агента.

Пример (скоростной агент поверх FOC):

```bash
python mic_ai/ai/train_agent.py --env-config config/env_demo_true_motor1.py --ident out/ident_full.json --episodes 80 --control-mode foc_assist --enable-id-control
```

Пример PPO‑агента для прямого управления dq‑напряжениями:

```bash
python mic_ai/ai/train_ai_voltage.py config/env_demo_true_motor1.py --ident out/ident_full.json --episodes 600
```

## Структура

- `models/` — dq‑модель двигателя, инвертор, преобразования.
- `control/` — V/f и FOC контроллеры.
- `simulation/` — Gym‑совместимая среда и сценарии.
- `mic_ai/ident/` — тесты и оценка параметров.
- `mic_ai/ai/` — обёртка‑среда и агенты/тренировка.
- `config/` — демо‑конфиги двигателя и AI‑настроек.

## Тесты

```bash
python -m pytest -q
```

