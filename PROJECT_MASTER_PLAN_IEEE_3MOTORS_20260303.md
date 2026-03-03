# MASTER PLAN: MIC/FOC/PI для 3 двигателей (AIR56, AL31, AO2)

Дата фиксации плана: 2026-03-03  
Репозиторий: `c:\mic_theory`  
Цель: довести проект до воспроизводимого состояния для инженерной эксплуатации и публикации уровня IEEE.
Последнее обновление статуса: 2026-03-04

### Статус-срез на 2026-03-03
- Инфраструктура step27/step28: закрыта (`DONE`), есть smoke в CI.
- Теория/физика: закрыта (`DONE`), есть валидатор + тесты + авто-репорты.
- Тестовый контур: закрыт для unit/integration/regression (`DONE`, 94 passed локально).
- IEEE контур: закрыт технически (`DONE`) — one-command reproducibility, package, derived figures/tables, auto-checklist, submission lock, strict verify, submission dossier.
- Обучение по 3 моторам: закрыто (`DONE`) в текущем frozen-пакете с acceptance-отчетами.
- Публикационный блок закрыт (`DONE`): manuscript sync + promoted figure/table release snapshot.
- Для канонического frozen-тега `20260303_ai_config_locked_nodrift` strict verify пройден (`verification_ok=True`).
- Итог: план выполнения доведен до `100%` по инфраструктурным/публикационным задачам; остаточные риски вынесены в научные `warnings`.

---

## 1) Текущее состояние (факт по аудиту кода)

### 1.1 Что уже есть
- Базовая модель АД, FOC и MIC/AI контур в репозитории.
- Набор исследовательских конфигов для 3 двигателей:
  - `config/env_research_air56_025kw.py`
  - `config/env_research_al31_4_06kw.py`
  - `config/env_research_ao2_32_4_3kw.py`
- Инструменты сравнения и отчётов:
  - `mic_ai/tools/scenario_compare.py`
  - `tools/multi_motor_study_report.py`
  - `tools/validate_pgups_study.py`
  - `tools/step27_pipeline.py`
  - `tools/build_step28_ieee_summary.py`
- Новые инструменты reproducibility и публикации IEEE:
  - `tools/reproduce_ieee_step28.py`
  - `tools/build_ieee_figures_tables.py`
  - `tools/build_ieee_final_checklist.py`
- CI и тесты есть (`.github/workflows/ci.yml`, 89 тестов проходят локально).
- Валидация PGUPS-таблиц проходит:
  - `python tools/validate_pgups_study.py` -> OK.

### 1.2 Подтверждённые критические проблемы (блокеры)
- Критичных блокеров в инфраструктуре/валидации нет: step27/step28 и publication-пайплайн воспроизводимы.
- Критичных блокеров по обучению нет: контур `10.3` закрыт (`DONE`) и подтверждён frozen-пакетом.
- Остаются научные риски (не блокеры инфраструктуры):
  - AO2 имеет ограниченный запас по экономии;
  - в passport-контуре AO2 MIC-строка невалидна на номинале и помечается как warning (экспортируется FOC proxy).

### 1.3 Что не доведено по результатам 3-моторного сравнения
- По текущему frozen-пакету `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift`:
  - AIR56: mean/worst acceptance по step27 проходит.
  - AL31: положительная экономия и неотрицательный gain по η.
  - AO2: экономия положительная, но запас по power-saving малый (зона повышенного риска регрессий).
- Следствие: инфраструктурно блокеров нет; для научной устойчивости рекомендован следующий retrain-цикл AO2 с фиксацией нового frozen-тега.

---

## 2) Целевое состояние (Definition of Done)

Проект считается доведённым, если одновременно выполнены пункты:

1. Step27/Step28 запускаются одной командой без ручных правок кода/конфигов.
2. Есть фиксированный протокол:
   - моторы: `air56, al31, ao2`
   - seeds: `101,202,303,404,505`
   - сценарии: `speed_step,ramp,load_step,start_stop`
   - контроллеры: `PI, FOC, MIC`
3. Есть стабильные итоговые артефакты:
   - `step27_per_seed_metrics.csv`
   - `step27_stats_motor_controller.csv`
   - `step27_final_pi_vs_foc_vs_mic.csv`
   - `step27_air56_acceptance.json`
   - `step27_reproducibility.json`
   - `step27_report.md`
   - `step28_ieee_summary.csv`
   - `step28_ieee_summary.md`
4. Физическая верификация проходит:
   - `0 <= cosφ <= 1`, физичная форма;
   - `0 <= η <= 1` (с малым численным допуском);
   - `P2 <= P1_total + eps`;
   - адекватные формы `M2(P2), I1(P2), n2(P2), η(P2), cosφ(P2)`.
5. Тестовый контур покрывает критические пайплайны (unit + integration + regression).
6. Подготовлен reproducible пакет под IEEE (данные, графики, таблицы, текст, протокол воспроизведения).

---

## 3) Приоритизация (P0/P1/P2)

## P0: восстановить работоспособность протокола

### P0.1 Починить `step27_pipeline.py` (API drift)
Проблема:
- `step27_pipeline.py` импортирует несуществующие функции из `scenario_compare.py`.

Сделать:
- Синхронизировать интерфейс:
  - либо вернуть в `scenario_compare.py` требуемые helper-функции;
  - либо обновить `step27_pipeline.py` под текущий API.
- Проверить совместимость `_simulate_ai`, `_summarize`, feature-key resolver, supervisor hooks.

Артефакты:
- Рабочий запуск:
  - `python tools/step27_pipeline.py --motors air56 --seeds 101 --scenarios speed_step --skip-air56-tune --out-dir outputs/_smoke_step27`

Проверка:
- Скрипт завершается без ImportError/TypeError.
- Генерирует `step27_*` артефакты в `out-dir`.

---

### P0.2 Добавить обязательные `ai_eval_*` параметры в research-конфиги
Проблема:
- Step27 требует `ai_eval_checkpoint_path`, но в research-конфигах этого нет.

Сделать:
- Для каждого из 3 конфигов добавить:
  - `ai_eval_checkpoint_path`
  - `ai_eval_id_ref_alpha`
  - `ai_eval_delta_id_max`
  - `ai_eval_id_ref_gate_speed_tol_rel`
  - `ai_eval_supervisor_enabled`
  - `ai_eval_sup_*` параметры
- Определить fallback-логику:
  - если checkpoint отсутствует, явная ошибка с понятным сообщением;
  - отдельный режим `--mic-rule-only` для диагностики без RL.

Артефакты:
- Коммит с обновлёнными конфигами.
- Документация по required fields в `docs/ieee_step28_protocol_ru.md`.

Проверка:
- `step27_pipeline.py` не падает на этапе загрузки env/agent.

---

### P0.3 Минимальный smoke тест для Step27 в CI
Проблема:
- CI не проверяет step27/step28, поэтому API drift не ловится заранее.

Сделать:
- Добавить в CI отдельный smoke job:
  - 1 мотор, 1 seed, 1 сценарий, короткий `t_end`.
- Тест должен запускать реальный `tools/step27_pipeline.py` и проверять наличие обязательных файлов.

Артефакты:
- Обновлённый `.github/workflows/ci.yml`.

Проверка:
- PR без совместимости step27 больше не проходит CI.

---

## P1: доведение качества MIC на 3 двигателях

### P1.1 Единый pipeline обучения для 3 двигателей
Проблема:
- Обучение по 3 моторам не оформлено как единый reproducible workflow.

Сделать:
- Ввести `tools/train_3motors_pipeline.py` (новый):
  - вход: список конфигов, seeds, эпизоды, curriculum;
  - выход: checkpoints + run manifests + eval snapshots.
- Режимы:
  - `separate-per-motor` (3 отдельных политики),
  - `joint-domain-randomized` (обучение с переключением мотора между эпизодами),
  - `fine_tune_per_motor` (дообучение после joint).

Артефакты:
- `results_run/<timestamp>_3motors_joint/...`
- `results_run/<timestamp>_3motors_finetune/...`
- `training_manifest_3motors.json`

Проверка:
- Повторный запуск с тем же seed даёт сопоставимые итоговые метрики.

---

### P1.2 AIR56: целевой тюнинг `start_stop` (без деградации остальных)
Проблема:
- Это целевой узкий сценарий из ТЗ.

Сделать:
- Зафиксировать acceptance:
  - `avg_power_saving_pct > 0.5%`
  - `avg_eta_gain_pct >= 0`
  - `err_failures <= 2`
  - `start_stop_power_saving_pct >= -0.5%`
- Реализовать target-tuning цикл:
  - stage1: быстрый поиск по `start_stop`,
  - stage2: проверка на полном наборе сценариев и seeds.
- Не допускать регрессии `ramp/load_step`.

Артефакты:
- `air56_tuning/stage1_rank.csv`
- `air56_tuning/stage2_rank.csv`
- `air56_tuning/tuning_summary.json`

Проверка:
- Проходит mean и worst-case acceptance одновременно.

---

### P1.3 AL31 и AO2: план исправления деградации
Проблема:
- AO2 в текущем состоянии даёт отрицательную экономию.
- AL31 в steady-части около нуля/ниже нуля.

Сделать:
- Для каждого мотора:
  - sweep по `id_ref_alpha`, `delta_id_max`, gating;
  - отдельный анализ по сценариям (`start_stop`, `load_step`);
  - калибровка потерь (`loss_inv_r`, `loss_core_k`) при необходимости.
- Добавить per-motor constraints:
  - MAE ratio <= 1.05;
  - load-work ratio >= 0.99;
  - mean power saving >= 0 (минимальная фаза стабилизации), потом целевой порог > 0.

Артефакты:
- `motor_<key>_tuning_report.md`
- `motor_<key>_search_rank.csv`

Проверка:
- AO2 выходит из отрицательной зоны по среднему power-saving.

---

## P1: физическая верификация и согласование с классической теорией АД

### P1.4 Теоретический валидатор характеристик (новый обязательный скрипт)
Проблема:
- Сейчас физичность графиков проверяется вручную, что нестабильно.

Сделать:
- Добавить `tools/validate_theory_working_characteristics.py`:
  - вход: CSV характеристик FOC/MIC;
  - проверки:
    - bounds: `eta in [0,1.02]`, `cosphi in [0,1]`;
    - `P2 <= P1_total + eps`;
    - тренды и shape-тесты:
      - `M2` растёт с нагрузкой;
      - `I1` растёт;
      - `n2` не растёт существенно (или убывает);
      - `η` имеет максимум/плато около номинала;
      - `cosφ` низкий на малой нагрузке, растёт к номиналу, без немотивированных пиков.
- Выход:
  - `theory_validation_report.json`
  - `theory_validation_report.md`

Артефакты:
- Новый скрипт + тесты + CI-hook.

Проверка:
- Валидатор запускается автоматически в пайплайне публикации.

---

### P1.5 Проверка cosφ методики и контроль line-vs-phase гипотезы
Проблема:
- Исторически были ошибки из-за путаницы фазных/линейных напряжений.

Сделать:
- Формально зафиксировать метод в `docs/analysis_tools_ru.md`.
- В `calc_cos_phi` добавить:
  - диагностические флаги, если обе гипотезы спорные;
  - персистентный лог warning по trace/scenario;
  - опцию жесткой фиксации метода (для экспериментальных повторов).

Артефакты:
- Обновление `mic_ai/analysis/metrics.py` + unit tests.

Проверка:
- Отдельный regression-test на синтетических 3-фазных сигналах:
  - unity PF, lag PF, distorted PF.

---

## P1: шаг к IEEE-ready результатам

### P1.6 Закрыть контур PI vs FOC vs MIC по 3 моторам (mean/std/min/max + worst)
Проблема:
- Протокол описан, но технически не закрыт end-to-end.

Сделать:
- После фикса P0 выполнить:
  - mode1: `FOC(encoder)` vs `MIC(sensorless)`,
  - mode2: `FOC(sensorless)` vs `MIC(sensorless)`.
- Для каждого mode:
  - собрать `step27_*`,
  - построить `step28_ieee_summary.*`.
- Ввести хранилище frozen-результатов:
  - `paper/ieee_2026/data/step28/<date>/...` (без тяжелых raw outputs).

Артефакты:
- Финальные CSV/MD сводки под статью.

Проверка:
- `stable_vs_previous` + hash-фиксация таблиц.

---

### P1.7 Убрать несогласованность сценариев в публикационных контурах
Проблема:
- В разных ветках отчётов используются `load_profile` и `load_step`.

Сделать:
- Для IEEE унифицировать сценарии: `speed_step,ramp,load_step,start_stop`.
- Для PGUPS legacy оставить backward-compatible генерацию, но пометить как legacy.
- Добавить явный mapping в отчётах.

Артефакты:
- Обновлённые `tools/multi_motor_study_report.py` / docs.

Проверка:
- Нет смешения режимов в итоговых таблицах.

---

## 4) Исторические gaps обучения (закрыты в текущем контуре)

### 4.1 Нет единой стратегии обучения «любой мотор -> эффективное управление»
Статус: `DONE` (joint-domain-randomized + fine_tune_per_motor реализованы в `tools/train_3motors_pipeline.py`).
- Сейчас обучение в основном запускается на одном конфиге за раз.
- Нет канонического meta-пайплайна, где мотор меняется по эпизодам в рамках одной политики.

Нужно:
- Joint training на распределении моторов + domain randomization.
- Затем тонкая адаптация на конкретный мотор (few-shot fine-tune).
- Сравнение:
  - zero-shot на новом моторе,
  - few-shot после N эпизодов.

---

### 4.2 Не зафиксированы единые training acceptance-критерии для всех 3 моторов
Статус: `DONE` (acceptance matrix + auto pass/fail в training manifests).
Нужно:
- Ввести training acceptance matrix:
  - per-motor/per-scenario:
    - power saving >= target,
    - eta gain >= target,
    - mae_ratio <= target,
    - current peak ratio <= target.
- Автоматическое «pass/fail» после каждой training-сессии.

---

### 4.3 Нет воспроизводимого registry checkpoint-ов
Статус: `DONE` (`config/checkpoint_registry.json` + checksum/hash связка в reproducibility контуре).
Проблема:
- Нет зафиксированных checkpoint-путей в версии кода.

Нужно:
- Добавить lightweight registry:
  - `checkpoints_registry.json` (локально/или в артефактном хранилище),
  - checksum каждого checkpoint,
  - связка с config hash и train manifest.

---

### 4.4 Недостаточно мониторинга деградаций во время обучения
Статус: `DONE` (eval snapshots + acceptance reports + CI checks добавлены).
Нужно:
- Ввести обязательный eval interval и лог:
  - `eval/ep_xxx/summary.json` для всех моторов/сценариев.
- Автостоп при деградации:
  - если `start_stop` сильно уходит в минус;
  - если MAE ratio превышает порог.

---

## 5) Тестовый план (исторический список; статус закрыт)
Статус: `DONE` (текущий объем: 89 passed).

### 5.1 Unit tests (обязательные новые)
- `tests/test_metrics_power_factor.py`
  - `calc_v_rms`, `calc_i_rms`, `calc_cos_phi`:
    - фазный вход;
    - линейный вход;
    - синтетические сигналы с известным PF.
- `tests/test_drive_characteristics_metrics.py`
  - корректность расчета `P1/P2/eta/cosphi` по окну.
- `tests/test_theory_validator.py`
  - shape checks и boundary checks.

### 5.2 Integration tests (обязательные новые)
- `tests/test_step27_pipeline_smoke.py`
  - мини-запуск step27 и проверка файлов.
- `tests/test_step28_summary_smoke.py`
  - сборка итогового summary из mode1/mode2.
- `tests/test_reproducibility_hash.py`
  - стабильность `table_sha256` при повторе.

### 5.3 Regression tests
- Зафиксировать эталонные мини-таблицы для 3 моторов (1 seed, короткий горизонт).
- Сравнение допусков по ключевым метрикам.

### 5.4 CI updates
- Добавить jobs:
  - `step27-smoke`
  - `theory-validation-smoke`
  - `step28-summary-smoke`

---

## 6) План верификации против реальной теории АД (обязательный контрольный контур)

## 6.1 Физические инварианты
- `P_in >= 0` (или корректная обработка рекуперации в signed-режиме).
- `P2 <= P1_total + eps`.
- `eta = P2/P1_total` не выходит за физический диапазон.
- `cosφ` в `[0,1]`.

## 6.2 Форма характеристик (качественные критерии)
- `M2(P2)`: монотонный рост в рабочей области.
- `I1(P2)`: рост с нагрузкой.
- `n2(P2)`: почти постоянная с небольшим проседанием под нагрузкой.
- `η(P2)`: рост от малых нагрузок, максимум/плато около номинала.
- `cosφ(P2)`: рост от малых нагрузок к номиналу, затем насыщение/слабое изменение.

## 6.3 Количественные критерии shape-контроля
- Монотонность с допусками на шум (например через isotonic envelope).
- Ограничения на вторую разность (anti-spike rule).
- Запрет «физически невозможных» скачков между соседними точками.

## 6.4 Проверка against-passport
- Для каждой машины:
  - сопоставить номинальную точку `Pn, In, cosφn, ηn, n_rated`.
- Ввести таблицу отклонений от паспортных значений.

## 6.5 Отдельный раздел по `start_stop`
- Проверка, что эффект экономии не достигается за счёт недовыполненной механической работы.
- Обязательный `load_work_ratio >= 0.99` (full + steady).

---

## 7) Публикационный контур IEEE (статус и остаток)

### 7.1 Структура IEEE-пакета (новая)
Статус: `DONE` (каркас создан).
- `paper/ieee_2026/manuscript.md`
- `paper/ieee_2026/fig/README.md`

Состав:
- `paper/ieee_2026/`
  - `manuscript.md` (рабочий англ. каркас),
  - `fig/` (правила и целевые подписи),
  - `data/` (итоговые CSV/JSON),
  - `reproduce.sh/.ps1` (одна команда).

### 7.2 Конверсия материалов VAK -> IEEE
Статус: `DONE`.
- Перенести численные результаты из PGUPS-формата в IEEE narrative.
- Убрать локальные path-зависимости на `outputs/research20260212/...`.

Факт закрытия:
- Обновлён `paper/ieee_2026/manuscript.md` с числовыми результатами frozen-тега `20260303_ai_config_locked_nodrift`.
- Добавлен `tools/promote_ieee_release.py` и выполнен промоушен в:
  - `paper/ieee_2026/fig/fig1..fig5*`
  - `paper/ieee_2026/data/release/20260303_ai_config_locked_nodrift/*`

### 7.3 Необходимые IEEE-таблицы/рисунки
Статус: `DONE`.
- PI vs FOC vs MIC (3 мотора): mean/std/min/max + worst-case.
- AIR56 детальная механическая характеристика FOC vs MIC.
- Cross-motor robustness figure.
- Training-to-performance curve (по 3 моторам).

Факт закрытия:
- Канонический набор фигур размещён в `paper/ieee_2026/fig/`.
- Канонический набор таблиц размещён в `paper/ieee_2026/data/release/20260303_ai_config_locked_nodrift/tables/`.

### 7.4 Репродуцируемость публикации
Статус: `DONE`.
- Скрипт одной кнопкой:
  - генерирует все таблицы и фигуры;
  - проверяет теорию;
  - пишет manifest + hashes.
- Добавлен `submission_candidate_lock.json` (SHA lockfile) и проверка lock в `FINAL_CHECKLIST_AUTO.md`.

---

## 8) Технический долг, который нужно закрыть

### 8.1 Устаревшие пути и legacy-сценарии в tools
- Есть скрипты с жёсткой привязкой к локальным `outputs/research20260212/...` и `results_run/...`.

Действия:
- Вынести все источники данных в явные CLI-аргументы.
- Добавить `--paper-data-dir` по умолчанию на committed `paper/pgups_2026/data`.
- Legacy скрипты пометить как `deprecated`.

Статус на 2026-03-03:
- `DONE` для `tools/plot_publication_figures_ru.py`, `tools/build_publication_from_markdown.py`, `tools/build_time_to_foc_summary_ru.py` (legacy default paths убраны; добавлены явные CLI-аргументы/предупреждения).
- `DONE` mapping `load_profile -> load_step` для publication report scripts при сохранении backward compatibility.

### 8.2 Консолидация инструментов AIR56-графиков
- Сейчас есть несколько перекрывающихся скриптов:
  - `tools/build_air56_mech_journal_from_traces.py`
  - `tools/build_air56_mech_only_compare.py`
  - `tools/build_air56_mech_only_from_sweep.py`
  - `tools/build_air56_working_characteristics_article.py`

Действия:
- Оставить 1 основной production script + 1 validation script.
- Остальные либо удалить, либо пометить deprecated.

Статус на 2026-03-03:
- `DONE`: production = `tools/build_air56_mech_journal_from_traces.py`.
- `DONE`: validation = `tools/validate_theory_working_characteristics.py`.
- `DONE`: `tools/build_air56_mech_only_compare.py`, `tools/build_air56_mech_only_from_sweep.py`, `tools/build_air56_working_characteristics_article.py` помечены как deprecated.

### 8.3 Кодировка/локаль
- Часть файлов читается с артефактами кодировки (cp1251/utf-8 mix).

Действия:
- Зафиксировать UTF-8 policy.
- Прогнать нормализацию markdown/docs.

Статус на 2026-03-03:
- `DONE` policy: `docs/UTF8_POLICY.md`.

---

## 9) Фазовый roadmap (практический порядок работ)

## Фаза 0 (1-2 дня): восстановить работоспособность ядра
1. Починить `step27_pipeline.py` API compatibility.
2. Добавить `ai_eval_*` параметры в 3 research-конфига.
3. Сделать smoke-run step27 (1 мотор, 1 seed, 1 scenario).
4. Добавить smoke-test в CI.

Критерий выхода:
- step27 smoke проходит локально и в CI.

## Фаза 1 (2-4 дня): стандартизация метрик и физическая верификация
1. Добавить theory-validator скрипт.
2. Добавить unit tests по metrics/cosphi.
3. Привязать validator к generation pipeline характеристик.

Критерий выхода:
- Любая новая фигура/таблица проходит auto-validation.

## Фаза 2 (4-7 дней): обучение и тюнинг на 3 моторах
1. Запустить отдельные baseline training/eval по 3 моторам.
2. Запустить joint-domain-randomized training.
3. Сфокусированный тюнинг AIR56 start_stop.
4. AO2 remediation до неотрицательной экономии.

Критерий выхода:
- Положительный/приемлемый баланс метрик по всем 3 моторам.

## Фаза 3 (2-3 дня): step28 режимы + итоговые таблицы
1. Полный прогон mode1/mode2.
2. Сборка `step28_ieee_summary.csv/.md`.
3. Reproducibility hash freeze.

Критерий выхода:
- Есть стабильный финальный step28 пакет.

## Фаза 4 (2-4 дня): IEEE пакет
1. IEEE figures/tables/scripts.
2. Manuscript sync.
3. Final reproducibility checklist.

Критерий выхода:
- Пакет готов к передаче в IEEE workflow.

---

## 10) Матрица контроля качества (для ежедневного прогресса)

Формат статуса: `TODO | IN_PROGRESS | DONE | BLOCKED`

### 10.1 Пайплайн и инфраструктура
- [x] DONE Починить API-совместимость `step27_pipeline.py`.
- [x] DONE Добавить `ai_eval_*` параметры в 3 research-конфига.
- [x] DONE Добавить checkpoint registry.
- [x] DONE Добавить step27 smoke в CI.
- [x] DONE Добавить step28 smoke в CI.

### 10.2 Теория и физика
- [x] DONE Реализовать `validate_theory_working_characteristics.py`.
- [x] DONE Ввести shape-контроль `M2/I1/n2/η/cosφ`.
- [x] DONE Добавить against-passport таблицу для 3 моторов.
- [x] DONE Включить validation report в publication pipeline.

### 10.3 Обучение
- [x] DONE Единый pipeline обучения для 3 моторов.
- [x] DONE Joint training across motors.
- [x] DONE AIR56 start_stop target-tuning.
- [x] DONE AL31 стабилизация steady-режимов.
- [x] DONE AO2 вывод из отрицательной экономии.

### 10.4 Тесты
- [x] DONE Unit tests для метрик cosφ/η/P2.
- [x] DONE Integration tests для step27/step28.
- [x] DONE Regression тесты на frozen mini-baselines.
- [x] DONE Обновление CI gates.
- [x] DONE CI gate `ieee-frozen-verify` (strict verify frozen submission-candidate).
- [x] DONE Manuscript template gate (`tools/check_ieee_manuscript_template.py`) встроен в verify/reproduce.
- [x] DONE Submission bundle builder (`tools/build_ieee_submission_bundle.py`) встроен в release contour.
- [x] DONE Release git planner (`tools/prepare_ieee_release_commit.py`) добавлен; генерирует `RELEASE_GIT_PLAN.{json,md}`.
- [x] DONE IEEE handoff note builder (`tools/build_ieee_submission_handoff.py`) добавлен; генерирует `IEEE_SUBMISSION_HANDOFF.{json,md}`.

### 10.5 IEEE
- [x] DONE Создать `paper/ieee_2026/`.
- [x] DONE Сформировать IEEE figures/tables.
- [x] DONE Подготовить reproducibility script one-command.
- [x] DONE Финальный checklist перед отправкой.
- [x] DONE Cross-motor guardrails (AIR56/AL31/AO2 saving thresholds) встроены в `FINAL_CHECKLIST_AUTO`.
- [x] DONE Submission candidate note (`SUBMISSION_CANDIDATE.md/.json`) формируется автоматически.
- [x] DONE Пороговые guardrails вынесены в версионируемую policy-конфигурацию (`paper/ieee_2026/guardrails_policy.json`).
- [x] DONE Добавлен schema-test для guardrails policy (защита от silent drift).
- [x] DONE Добавлены release-wrapper скрипты strict submission-candidate (PS1/SH).
- [x] DONE Добавлен immutable `RELEASE_COMMIT_MANIFEST` (hash + git metadata) в one-command pipeline.
- [x] DONE Добавлен `verify`-контур для готового пакета (`tools/verify_ieee_submission_candidate.py`).
- [x] DONE Добавлены wrapper-скрипты verify (`scripts/verify_ieee_submission_candidate.ps1/.sh`).
- [x] DONE Добавлен `IEEE_SUBMISSION_DOSSIER` (единый digest статуса/метрик/hash для frozen-пакета).
- [x] DONE Добавлен manuscript consistency gate (`tools/check_ieee_manuscript_consistency.py`) в verify/CI.

---

## 11) Риски и меры

### Риск R1: нестабильные метрики из-за seed/model drift
Меры:
- фиксированные seeds;
- hash таблиц;
- deterministic режимы где возможно;
- strict protocol freeze.

### Риск R2: «улучшение» за счёт ухудшения механической работы
Меры:
- обязательный load-work sanity check;
- отчёт `load_work_ratio` в каждой таблице.

### Риск R3: несовместимость скриптов после локальных правок
Меры:
- smoke tests на step27/step28 в CI;
- запрет merge без прохождения новых smoke checks.

### Риск R4: AO2 остаётся отрицательным
Меры:
- отдельная ветка параметров/гейтинга для AO2;
- staged acceptance (сначала >=0, затем целевой +margin).

---

## 12) Конкретный ближайший спринт (следующие действия по порядку)

1. [x] Технически подготовлен релизный commit-план с frozen-артефактами (`RELEASE_GIT_PLAN.{json,md}` + команды `git add/commit/tag/push`).
2. [ ] Запушить релизный commit/tag в origin (операция требует фактического выполнения в git remote).
3. [x] Подготовить IEEE submission bundle (архив + hash-manifest + ссылки на reproducibility scripts).
4. [x] Провести финальную вычитку `paper/ieee_2026/manuscript.md` под целевой IEEE template через авто-проверки (consistency + template gate).
5. [x] Подготовлен IEEE handoff package (`IEEE_SUBMISSION_HANDOFF.{json,md}` + submission bundle archives).

---

## 13) Команды для контрольного запуска (после фикса P0)

### 13.1 Step27 (базовый прогон)
```bash
python tools/step27_pipeline.py \
  --motors air56,al31,ao2 \
  --seeds 101,202,303,404,505 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --out-dir outputs/progress_step27_pipeline \
  --foc-feedback-mode encoder \
  --mic-feedback-mode sensorless \
  --seed-perturbation \
  --seed-perturb-level 0.2
```

### 13.2 Step28 (оба режима)
```bash
scripts/run_step28_ieee_protocol.ps1
```
или
```bash
scripts/run_step28_ieee_protocol.sh
```

### 13.3 One-command IEEE reproducibility (текущий основной запуск)
```bash
python tools/reproduce_ieee_step28.py \
  --out-root outputs/progress_step28_ieee_repro \
  --package-root paper/ieee_2026/data/step28 \
  --package-tag 20260303_repro \
  --mic-mode rule
```

### 13.4 Promoted IEEE release (figures + release snapshot)
```bash
python tools/promote_ieee_release.py \
  --step28-dir paper/ieee_2026/data/step28/20260303_repro \
  --ieee-root paper/ieee_2026 \
  --pgups-fig-dir paper/pgups_2026/fig \
  --tag 20260303_repro
```

### 13.5 Валидация теории (точечный запуск)
```bash
python tools/validate_theory_working_characteristics.py \
  --csv paper/pgups_2026/fig/working_characteristics_air56_foc_mic_table.csv \
  --out-json outputs/theory_validation_manual/report.json \
  --out-md outputs/theory_validation_manual/report.md
```

---

## 14) Результат, который должен быть получен на выходе проекта

- Рабочий и воспроизводимый контур:
  - обучение,
  - сравнение PI/FOC/MIC,
  - физическая верификация,
  - публикационные артефакты.
- Подтверждённое обобщение MIC на 3 двигателя.
- Прозрачный журнал проверки теории и качества.
- IEEE-ready пакет без ручной доработки чисел/графиков.
