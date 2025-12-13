# MIC_AI

Motor Intelligence Control with Artificial Intelligence  
Self‑Learning Control of Induction Motor (dq‑модель)

## 1. Что это за проект и зачем он существует

MIC_AI — исследовательско‑инженерный проект по созданию самообучающегося управления асинхронным двигателем, ориентированного на реальный электропривод (ограничения, безопасность, воспроизводимость), а не на демонстрационные reinforcement learning.

Проект отвечает на вопрос:
можно ли управлять асинхронным двигателем без классического FOC и PI‑регуляторов, используя прямое управление напряжениями в dq‑координатах, и получить сопоставимую или лучшую точность и энергоэффективность с перспективой переноса на микроконтроллер.

## 2. Ключевая идея

Сначала физика и безопасность → потом обучение → потом компактность → потом адаптация.

RL здесь — инструмент синтеза регулятора. Финальные выводы делаются по физическим метрикам и детерминированной оценке, а не по reward.

## 3. Отличия от типичных RL‑проектов

| Типичный RL | MIC_AI |
|---|---|
| Агент «играет» | Агент — регулятор |
| Нет реальных ограничений | Жёсткие ограничения по току/напряжению |
| «Сломать эпизод» допустимо | Hard‑termination отслеживается и не является целью |
| Reward = критерий качества | Reward — обучающий сигнал |
| Деплой не обязателен | Цель — перенос на MCU |

## 4. Архитектура управления

### 4.1 Режимы управления

- Baseline и safety fallback: V/f и FOC (PI по скорости, PI по токам).
- AI‑управление (`ai_voltage`): агент напрямую управляет `v_d / v_q` (без PI и FOC в контуре).

### 4.2 Контур `ai_voltage`

`action ∈ [-1, 1]^2` → масштабирование `v_d/v_q` → safety envelope (по току) → сглаживание → `dq→abc` → инвертор → dq‑модель АД

## 5. Digital Twin (модель привода)

- dq‑модель асинхронного двигателя
- модель трёхфазного инвертора
- преобразования `abc ↔ dq`
- расчёт момента и механики (`J`, `B`)
- сценарии скорости и нагрузки

## 6. Идентификация параметров двигателя

Команды (симуляция/демо‑конфиг):
```bash
python mic_ai/tools/identify_motor.py --env-config config/env_demo_true_motor1.py --output out/ident_simple.json
python mic_ai/tools/full_identify_motor.py --env-config config/env_demo_true_motor1.py --output out/ident_full.json
```

## 7. RL (PPO)

- Алгоритм: PPO (on‑policy), PyTorch
- Состояние (пример): нормированная скорость, `i_d/i_q`, slip, предыдущие действия
- Reward используется как обучающий сигнал; качество сравнивается по физическим метрикам на детерминированном eval

## 8. Методика сравнения AI vs FOC (строго и воспроизводимо)

Для корректного сравнения используются одинаковые:
- профили `ω_ref(t)` (в т.ч. piecewise по эпизоду)
- нагрузка `T_load(t)`
- длительность эпизода (`episode_steps`)
- ограничения по току и напряжению

Оцениваемые метрики (усреднение по эпизоду):
- `I_rms` статора, A
- `P_in^+` — положительная составляющая входной мощности, W
- `|ω_ref − ω|`, рад/с

## 9. Установка и быстрый старт

Установка зависимостей:
```bash
python -m pip install -r requirements.txt
```

Запуск симуляции:
```bash
python main.py --mode foc --scenario speed_step --t-end 0.5 --dt 0.0001 --no-plot
```
Результаты: `outputs/results/data_N.npz`

Графики по сохранённому `npz`:
```bash
python -c "from outputs.plots import plot_run; plot_run('outputs/results/data_1.npz')"
```
Результаты: `outputs/figures/`

## 10. Обучение `ai_voltage`

Рекомендуемый режим для воспроизводимых сравнений: `--episode-steps 200`.

```bash
python mic_ai/ai/train_ai_voltage.py config/env_demo_true_motor1.py --episodes 600 --episode-steps 200
```

Артефакты:
- `outputs/demo_ai/episode_logs/` — JSON‑логи эпизодов
- `outputs/demo_ai/plots/` — `npz` с кривыми обучения
- `outputs/demo_ai/checkpoints/` — чекпойнты (`best_actor.pth`, `last_actor.pth`)
- `results_run/<timestamp>_<config>/` — метрики/плоты запуска

## 11. Детерминированный eval и сравнение с FOC

Eval чекпойнта (политика без стохастики действий):
```bash
python -m mic_ai.tools.eval_ai_voltage_checkpoint --env-config config/env_demo_true_motor1.py --checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episodes 30 --episode-steps 200 --output outputs/demo_ai/episode_logs/ai_voltage_eval_motor1.json
```

Сравнение и базовые графики:
```bash
python -m mic_ai.tools.compare_ai_foc --env-config config/env_demo_true_motor1.py --ai-episodes outputs/demo_ai/episode_logs/ai_voltage_eval_motor1.json --output-dir outputs/compare_ai_foc_eval_200 --episode-steps 200 --foc-eval-episodes 30
```

## 12. Графики для статьи (IEEE‑стиль, русские подписи)

Скрипт генерирует 4 фигуры (Irms, Pin+, |ω_ref−ω|, Pareto) в PNG и PDF и сохраняет исходные JSON‑данные для воспроизводимости.

```bash
python -m mic_ai.tools.paper_plots_ai_vs_foc --env-config config/env_demo_true_motor1.py --ai-checkpoint outputs/demo_ai/checkpoints/motor1/last_actor.pth --episode-steps 200 --episodes-per-stage 25 --window 5 --out-dir outputs/paper_figures_motor1_strict --voltage-scale 1.0 --disable-noise
```

Результаты: `outputs/paper_figures_motor1_strict/`

## 13. Структура репозитория

```
mic_ai/
  ai/            обучение и окружение AI
  ident/         идентификация параметров
  tools/         отчёты, графики, анализ
control/         V/f, FOC
models/          dq‑двигатель, инвертор, преобразования
simulation/      окружение и сценарии
outputs/         результаты и артефакты
results_run/     артефакты запусков (метрики/плоты/веса)
```

## 14. Тесты

```bash
python -m pytest -q
```

## 15. Примечания

- `gym` выводит предупреждение на NumPy 2.x; это не блокирует запуск. Переход на `gymnasium` запланирован.
- Инструменты в `mic_ai/tools/` предпочтительно запускать как модуль: `python -m mic_ai.tools.<script> ...`
