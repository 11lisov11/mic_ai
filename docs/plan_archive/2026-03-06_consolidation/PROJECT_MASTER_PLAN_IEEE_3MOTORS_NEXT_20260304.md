# GLOBAL MASTER PLAN: MIC/FOC/PI для 3 двигателей (AIR56, AL31, AO2)

Дата обновления: 2026-03-04  
База: `PROJECT_MASTER_PLAN_IEEE_3MOTORS_20260303.md` закрыт на `100%` как инфраструктурный этап.  
Назначение этого файла: единый практический roadmap до инженерно зрелого и публикационно устойчивого состояния.

## 0) Цель цикла
- Зафиксировать воспроизводимый baseline для 3 моторов и двух сравнительных режимов (`mode1`, `mode2`).
- Довести MIC до устойчивого преимущества относительно FOC/PI без нарушения физики АД.
- Закрыть публикационный контур IEEE (submission-ready + rebuttal-ready) для стабильного frozen тега.

## 1) Опорные требования (Definition of Done)
- [x] Единый step27/step28 pipeline работает без ручных правок.
- [x] Есть strict-проверки пакета (`verification_ok`, `camera_ready_ok`, `strict_ready`).
- [x] Есть regression guard на summary-таблицы.
- [ ] По всем 3 моторам подтвержден устойчивый запас MIC (mean и worst-case) на фиксированном seed-протоколе.
- [x] AO2 выведен из зоны малого запаса: целевой `avg_power_saving_pct_mean >= 0.5%` без деградации `avg_eta_gain_pct`.
- [x] Подготовлен единый training/eval контур для 3 моторов (без лишнего retrain в прод-сценариях).
- [x] Полный тестовый контур включает физические sanity-checks и визуальные guardrails для графиков.

## 2) Текущий факт-срез (что уже закрыто)
- [x] Robust hardening tool внедрен: `tools/robust_motor_hardening.py`.
- [x] Для AL31 применен устойчивый профиль (`rand_009`) с улучшением perturbation worst-case.
- [x] Собран строгий пакет: `paper/ieee_2026/data/step28/20260304_al31_robust_rand009_nodrift_v3`.
- [x] Выпущен submission bundle: `paper/ieee_2026/submission_bundle/20260304_al31_robust_rand009_nodrift_v3`.
- [x] Passport/rebuttal/release контуры сформированы и связаны с frozen тегом.

## 3) Глобальный backlog (по потокам работ)

### 3.1 Stream A: Научная валидность и физика модели
- [x] A1. Зафиксировать единый physical config policy для AIR56/AL31/AO2 (loss/thermal/friction assumptions в одном документе).  
  Факт: `tools/build_physical_config_policy.py`, `docs/physical_config_policy_3motors.{md,json}`.
- [x] A2. Добавить автоматическую проверку формы кривых `M2(P2), I1(P2), n2(P2), η(P2), cosφ(P2)` без ручного просмотра.  
  Факт: `tools/validate_theory_working_characteristics.py` + тесты `tests/test_theory_validator.py`.
- [x] A3. Добавить отдельный детектор нефизичных изломов/скачков (`n2`, `cosφ`, `η`) на уровне CSV.  
  Факт: локальный spike-detector в `validate_theory_working_characteristics.py` (`n2_spike_detector`, `eta_spike_detector`, `cosphi_spike_detector`).
- [x] A4. Ввести визуальный regression-тест для ключевых фигур (допуски на shape и axis consistency).  
  Факт: `tools/check_working_characteristics_visual_regression.py` + `tests/test_check_working_characteristics_visual_regression_smoke.py`, отчет: `paper/pgups_2026/fig/working_characteristics_air56_foc_mic_visual_regression.json`.
- [x] A5. Выпустить теоретический verification report по каждому мотору (AIR56/AL31/AO2) в `paper/ieee_2026/data/theory_validation/`.  
  Факт: `tools/build_theory_validation_reports.py`, отчеты в `paper/ieee_2026/data/theory_validation/20260304_al31_robust_rand009_nodrift_v3/` (текущий `all_passed=false`, требуется дальнейшая калибровка/сглаживание).

### 3.2 Stream B: Устойчивость алгоритма и качество управления
- [x] B1. Завершить AO2 hardening с целевым запасом `>= 0.5%` по `avg_power_saving_pct_mean`.  
  Факт: `outputs/ao2_hardening_v11_20260304_b1_relaxed/ao2_tuning_summary.json`, выбран кандидат `rand_002`: `avg_power_saving_pct=+2.102%`, `avg_eta_gain_pct=+10.502%`, `err=2.0`.  
  Примечание: stricter safety/envelope критерии (`start_stop`, current ratios) остаются открыты и закрываются отдельно в `Phase 1`.
- [x] B2. Проверить AO2/AL31 на расширенном perturbation sweep (`0.0/0.1/0.2/0.3/0.4`) с отчетом worst-case.  
  Факт: `outputs/step27_extended_repro_v5_20260304_al31_ao2_p014/step27_extended_stress_sweep.csv` + `step27_extended_report.md`.
- [x] B3. Ввести dual-criteria selection policy: robust-score + baseline safety guard для всех моторов.  
  Факт: реализовано в `tools/robust_motor_hardening.py` (`selection_policy=safe_baseline_guard`).
- [x] B4. Зафиксировать per-motor acceptance envelopes для `start_stop`, `ramp`, `load_step`, `speed_step`.  
  Факт: `config/acceptance_envelopes_3motors.json` + `tools/check_motor_acceptance_envelopes.py`; пример отчета: `outputs/step27_extended_repro_v5_20260304_al31_ao2_p014/runs/baseline/acceptance_envelopes/acceptance_envelope_summary.json`.
- [x] B5. Выпустить consolidated robust ranking table для 3 моторов в одном CSV/MD.  
  Факт: `tools/build_robust_hardening_consolidated.py` и артефакты в `outputs/robust_hardening_consolidated_20260304/`.

### 3.3 Stream C: Обучение и обобщение на 3 моторах
- [x] C1. Спроектировать `train_3motors_pipeline` (single entrypoint, manifest, seed protocol, resume).  
  Факт: `tools/train_3motors_pipeline.py` поддерживает `--resume-manifest`, `--eval-first`, фиксирует `training_protocol_3motors.json`.
- [x] C2. Разделить режимы `joint-domain-randomized` и `fine_tune_per_motor` с воспроизводимыми конфигами.  
  Факт: режимы зафиксированы в едином entrypoint + protocol hash и SHA конфигов моторов в `training_protocol_3motors.json`.
- [x] C3. Добавить cross-motor generalization evaluation (train on subset, eval on held-out motor domains).  
  Факт: `tools/eval_cross_motor_generalization.py` + `tests/test_eval_cross_motor_generalization_smoke.py`; пример реального запуска: `outputs/cross_motor_generalization_20260304_small/`.
- [x] C4. Добавить ограничение "no unnecessary retrain": eval-first policy в CI/tools.  
  Факт: `tools/train_3motors_pipeline.py --eval-first --resume-manifest ...` переиспользует accepted прогоны без retrain.
- [x] C5. Зафиксировать минимальный reproducible training package (manifest + checkpoints + metrics hash).  
  Факт: `training_repro_package_3motors.json` (артефакты + SHA256 + checkpoint inventory).

### 3.4 Stream D: Тестирование, рефакторинг, инженерная зрелость
- [x] D1. Расширить smoke -> integration контур для `step27_pipeline`, `reproduce_ieee_step28`, `robust_motor_hardening`.  
  Факт: `tools/run_integration_pipeline.py` + `tests/test_run_integration_pipeline_smoke.py`, реальные отчеты в `outputs/integration_pipeline/`.
- [x] D2. Добавить unit-тесты на метрики мощности/КПД/cosφ и детекцию line-vs-phase mismatch.  
  Факт: `tests/test_metrics_power_factor.py`.
- [x] D3. Ввести contract-тесты на формат всех итоговых CSV/JSON артефактов.  
  Факт: `tests/test_artifact_contracts.py` (контракты на `step27_per_seed_metrics.csv`, `step27_stats_motor_controller.csv`, `step28_ieee_summary.csv`, `theory_validation_summary.csv`).
- [ ] D4. Провести рефакторинг `tools/` по слоям: `data`, `eval`, `report`, `release`.
- [x] D5. Обновить `docs/` (операционный runbook + troubleshooting + acceptance protocol).  
  Факт: `docs/runbook_3motors_ops.md`.

### 3.5 Stream E: Публикационный контур IEEE
- [ ] E1. Пересобрать step28 для нового финального frozen тега после AO2 hardening.
- [ ] E2. Обновить Figures/Tables package и проверить непротиворечивость manuscript ссылок.
- [ ] E3. Выпустить camera-ready checklist и submission handoff для нового тега.
- [ ] E4. Сформировать rebuttal evidence pack с хэшами и traceability до raw CSV.
- [ ] E5. Провести final editorial pass по результатам 3 моторов (одна сводная narrative-логика).

## 4) План по этапам (очередность выполнения)
- [x] Phase 0: Stabilize baseline artifacts (завершено).
- [ ] Phase 1: AO2 margin hardening + robust sweep closure.
- [x] Phase 2: Theory/shape validators + visual regression guards.
- [x] Phase 3: Training pipeline formalization for 3 motors.
- [ ] Phase 4: Full regression and release dry-run.
- [ ] Phase 5: Final IEEE freeze + camera-ready handoff.

## 5) Метрики управления планом
- [ ] Completion >= 90% по чекбоксам этого плана.
- [ ] Ни одного `BLOCKED` пункта в критическом пути (`B1`, `D2`, `E1`).
- [ ] Все целевые артефакты нового frozen тега существуют и проходят strict checks.
- [ ] Научные риски AO2/AL31 документированы и закрыты количественно (mean/std/worst).

## 6) Критический путь (коротко)
1. `B1 -> B2 -> B4 -> E1`
2. `A2 -> A3 -> D2 -> D3 -> E2`
3. `C1 -> C2 -> C5 -> E5`

## 7) Примечание по статусу
Текущий план намеренно не равен 100%: это рабочий backlog следующего цикла.  
Отчеты прогресса обновляются автоматически через `tools/report_plan_completion.py`.
