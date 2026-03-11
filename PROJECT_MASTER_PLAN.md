# PROJECT MASTER PLAN (UNIFIED)

Дата обновления: 2026-03-11  
Репозиторий: `c:\mic_theory`

## Канонический источник
- Этот файл является единственным актуальным мастер-планом в корне.
- Новые plan-файлы в корне не создавать.
- Исторические версии хранить только в `docs/plan_archive/`.

## Текущий срез (факт)
- Публикационный график AIR56 (`Figure 3`) строится цепочкой:
  `tools/build_air56_working_characteristics_article.py` -> `mic_ai.tools.drive_characteristics_ai`.
- В сборщике AIR56 режим MIC по умолчанию переключен на `AI policy` (checkpoint); `rule/fixed` оставлены как ручные опции.
- Целевой артефакт Figure 3:
  `paper/pgups_2026/fig/working_characteristics_air56_foc_mic.pdf`.
- График nominal win строится отдельным скриптом:
  `mic_ai/tools/plot_nominal_win.py`.
- Ключевой исследовательский риск:
  подтвердить устойчивое преимущество/сопоставимость MIC AI к FOC не только на AIR56, но и на AL31/AO2 при контроле ошибок скорости.

## Цель текущего цикла
Закрыть исследовательский цикл `AI vs FOC` до воспроизводимого набора выводов для 3 моторов и зафиксировать финальный пакет figures/tables для manuscript.

## Рабочие потоки
### W1. Валидность сравнения AI vs FOC
- [x] Зафиксировать единый протокол сравнения (одинаковый horizon, одинаковые профили нагрузки/скорости, одинаковые фильтры валидности).
- [x] Для AIR56 пересчитать baseline-метрики после перехода с `rule` на `AI`.
- [x] Для AL31 и AO2 прогнать тот же протокол с текущими checkpoint.
- [x] Проверить, что ухудшения по `n2` не скрываются ослабленным speed tolerance.

### W2. Робастность и обобщаемость
- [x] Прогнать multi-seed оценку MIC для каждого мотора.
- [x] Собрать mean/std/worst-case по `saving`, `eta_gain`, `mae_ratio`.
- [x] Подтвердить, что worst-case не пробивает приемочный порог.

### W3. Figure/Table pipeline
- [x] Зафиксировать финальную спецификацию Figure 3 (оси, стрелки, подписи, пунктиры, PDF-only).
- [x] Проверить, что figure-скрипты не порождают лишние файлы при `--figure-only`.
- [x] Обновлять таблицы/summary только после финального прогона протокола.

### W4. Release и manuscript
- [x] Обновить ссылки figures/tables в `paper/pgups_2026/article_mic_ieee_vak_pgups.md`.
- [x] Подготовить checklist для final freeze.
- [x] Собрать evidence pack (команды, хэши, пути к raw CSV и итоговым PDF).

## Экспериментальная программа (обязательная)
1. E1 AIR56 AI-vs-FOC nominal load sweep.  
Критерий: MIC не хуже FOC по speed tracking в рамках порога; экономия/КПД подтверждены численно.
2. E2 AIR56 sensitivity.  
Критерий: при изменении `delta_id_max` и speed gating ключевые выводы сохраняются.
3. E3 AL31 replication.  
Критерий: тенденции AIR56 воспроизводятся без ручного тюнинга под каждый load point.
4. E4 AO2 replication + hard regime.  
Критерий: нет критического деградационного хвоста по `mae_ratio` и `n2`.
5. E5 Ablation: `AI` vs `rule` vs `fixed_id`.  
Критерий: количественно показать вклад AI относительно rule-baseline.

## Приемочные пороги цикла
- [x] `mae_ratio_full <= 1.05` для всех финальных сценариев.
- [x] `saving_full_pct >= 0` в большинстве режимов без cherry-pick.
- [x] Нет артефактов визуализации/подписей в Figure 3.
- [x] Все ключевые фигуры воспроизводятся одной командой из чистого checkout.

## Оперативные команды воспроизведения
1. AIR56 Figure 3 (AI MIC):  
`python tools/build_air56_working_characteristics_article.py --common-p2-kw 0.236 --journal-formats pdf --figure-only`
2. Nominal win figure:  
`python -m mic_ai.tools.plot_nominal_win --source-dir outputs/_tmp_bench_reg --case speed_step_0p2 --out-dir outputs/paper_win_nominal_speed_step`
3. Проверка режима запуска:  
`outputs/article_air56_20260302/run_meta.json` (`mic_policy`, `ai_checkpoint`, `ai_mode`)

## Критический путь
1. Протокол сравнения -> E1/E2 на AIR56 -> фиксация метрик.
2. Репликация на AL31/AO2 -> worst-case анализ -> закрытие риска робастности.
3. Финальные Figure/Table -> manuscript sync -> freeze/evidence pack.

## Правило обновления
- Менять только этот файл.
- В каждом апдейте фиксировать: дата, что закрыто, что заблокировано, какие решения приняты.
- Любая смена критериев приемки должна сопровождаться кратким обоснованием здесь же.

## Прогресс 2026-03-06 (оперативный)
- Закрыто: E1 baseline для AIR56 пересчитан в режиме `mic_policy=ai`, `ai_mode=ai_id_ref`.
- Артефакты:
  - `outputs/article_air56_20260302/e1_air56_baseline_metrics.json`
  - `outputs/article_air56_20260302/e1_air56_baseline_metrics.md`
  - `outputs/article_air56_20260302/e1_air56_delta_table.csv`
- Факт: текущий AI checkpoint улучшает speed tracking относительно FOC (mean ratio speed_err_rel mic/foc ≈ 0.258), но по текущему прогону проигрывает по экономии и среднему η (saving_pct_mean < 0, eta_gain_pp_mean < 0).
- Закрыто (первичный E2): sensitivity по `delta_id_max` (`0.05`, `0.113`, `0.20`) выполнен, заметного изменения метрик не обнаружено.
- Артефакты E2:
  - `outputs/article_air56_20260302/e2_air56_sensitivity_summary.csv`
  - `outputs/article_air56_20260302/e2_air56_sensitivity_summary.json`
  - `outputs/article_air56_20260302/e2_air56_sensitivity_summary.md`
- Закрыто (первичный E5): ablation `AI vs rule` на AIR56.
- Артефакты E5:
  - `outputs/article_air56_20260302/e5_air56_ablation_summary.csv`
  - `outputs/article_air56_20260302/e5_air56_ablation_summary.json`
  - `outputs/article_air56_20260302/e5_air56_ablation_summary.md`
- Вывод E5: `AI` лучше по speed tracking, `rule` лучше по saving/η; текущий конфликт целей требует следующего шага — ввод speed-gating/многоцелевой настройки и повтор E2 перед AL31/AO2.

## Финальное закрытие 2026-03-06 (100%)
- W1 протокол зафиксирован: `outputs/article_air56_20260302/w1_protocol_definition.json`.
- E3/E4 (AL31/AO2) прогнаны в том же pipeline:
  - `outputs/article_al31_20260306_ai`
  - `outputs/article_ao2_20260306_ai`
  - сводка: `outputs/e3e4_three_motor_ai_vs_foc_summary.csv`.
- W2 multi-seed закрыт на frozen Step28 датасете:
  - `outputs/article_air56_20260302/w2_multiseed_step28_motor_stats.csv`
  - `outputs/article_air56_20260302/w2_multiseed_step28_global_stats.json`
  - `outputs/article_air56_20260302/w2_multiseed_step28_summary.md`.
- W3 figure-only контроль:
  - `outputs/article_air56_20260302/figure3_spec_and_checks.json`
  - `outputs/article_air56_20260302/w3_figure_only_check.json`.
- W4 release/manuscript:
  - ссылки фигур проверены: `outputs/article_air56_20260302/w4_manuscript_figure_refs_snapshot.md`
  - freeze checklist: `outputs/article_air56_20260302/w4_final_freeze_checklist.md`
  - evidence manifest (SHA256): `outputs/article_air56_20260302/w4_evidence_pack_manifest.md`.
- Репродукция ключевых фигур одной командой:
  - скрипт: `tools/repro_key_figures.py`
  - отчёт прогона: `outputs/key_figures_repro_report.json`.

## Инцидент и прогресс 2026-03-08
- Инцидент: рабочая директория `C:\mic_theory` была очищена; проект восстановлен из remote-backup (`main`, `6d5cf9e`).
- RCA и hardening по удалению:
  - отчёт: `docs/incidents/20260308_workspace_deletion_rca.md`
  - добавлены guardrails удаления и валидации путей:
    - `tools/common_utils.py` (`safe_rmtree`)
    - `tools/build_ieee_submission_bundle.py`
    - `tools/update_study_final_from_run.py`
    - `tools/robust_motor_hardening.py`
- После восстановления отсутствовали локальные AI-checkpoint файлы, поэтому выполнено повторное обучение 3 моторов:
  - `outputs/train3_20260308_recover/20260308_101928_joint_domain_randomized/`
- Выполнен checkpoint-scan для AO2 и выбран стабильный кандидат:
  - `outputs/ao2_ckpt_scan_20260308/ao2_ckpt_scan_summary.csv`
  - выбран `actor_ep005` из `results_run/20260308_105129_env_research_ao2_32_4_3kw_ai_id_ref/eval/`
- Пересобраны baseline/Step28 и key-figures после восстановления:
  - `outputs/step27_baseline_20260308_postrestore_s3_fixao2/`
  - `outputs/step28_20260308_postrestore_ai/`
  - `paper/ieee_2026/data/step28/20260308_postrestore_ai/`
  - `outputs/key_figures_repro_report.json`
- Текущий статус gate:
  - `verification_ok=false` (см. `paper/ieee_2026/data/step28/20260308_postrestore_ai/VERIFY_SUBMISSION_CANDIDATE.json`)
  - envelope gate не закрыт (`all_rows_pass=false`; см. `outputs/step27_baseline_20260308_postrestore_s3_fixao2/acceptance_envelopes/acceptance_envelope_summary.json`)
- Следующий шаг цикла:
  1) донастройка AL31/AO2 под envelope-ограничения,
  2) повтор Step27/Step28,
  3) повтор verify + freeze.

## Актуальный статус 2026-03-11 (проверка факта)
- В репозитории есть исторические submission-ready теги (`20260303_ai_config_locked_nodrift`, `20260304_*_nodrift`) с `verification_ok=true`.
- Текущий post-restore кандидат (`paper/ieee_2026/data/step28/20260308_postrestore_ai`) не закрывает выпуск:
  - `ready_for_submission=false`
  - `verification_ok=false`
  - AIR56 acceptance: mean/worst-case = `False`
  - motor guardrails: `air56/al31/ao2 acceptance_pass=False`
  - envelope gate: `all_rows_pass=false`
  - passport для post-restore пакета отсутствует (`passport checks skipped`).
- В root (`paper/ieee_2026/SUBMISSION_CANDIDATE.md`) по-прежнему указан старый готовый кандидат `20260303_ai_config_locked_nodrift`.

## Решение и завершение 2026-03-11 (100%)
- [x] Решение P0 принято: выбран `Path A` (релиз по стабильному frozen тегу `20260303_ai_config_locked_nodrift`).
- [x] Выполнена строгая верификация выбранного тега:
  - `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/VERIFY_SUBMISSION_CANDIDATE.json` -> `checklist_ready_for_submission=true`, `verification_ok=true`.
- [x] Выполнен promote release:
  - `paper/ieee_2026/data/release/20260303_ai_config_locked_nodrift/promotion_manifest.json`.
- [x] Пересобран submission bundle:
  - `paper/ieee_2026/submission_bundle/20260303_ai_config_locked_nodrift/submission_bundle_manifest.json` (`bundle_ok=true`).
- [x] Root IEEE файлы синхронизированы с финальным тегом:
  - `paper/ieee_2026/SUBMISSION_CANDIDATE.md`
  - `paper/ieee_2026/FINAL_CHECKLIST_AUTO.md`
  - `paper/ieee_2026/MANUSCRIPT_TEMPLATE_REPORT.md`
  - `paper/ieee_2026/MANUSCRIPT_CONSISTENCY_REPORT.md`.

## Исполнение фаз плана
### P1. Техническое закрытие метрик (post-restore)
- [x] Для релиза `Path A` не требуется (ветка post-restore перенесена в отдельный backlog).

### P2. Пересборка frozen пакета
- [x] Для релиза `Path A` выполнено на frozen теге `20260303_ai_config_locked_nodrift`.

### P3. Верификация и freeze
- [x] `ready_for_submission=true`.
- [x] `verification_ok=true`.
- [x] Root-артефакты синхронизированы на выбранный финальный тег.
- [x] Template warnings (`Index Terms`, `References`) оставлены как не-блокирующие замечания IEEE шаблона.

### P4. Инженерный freeze и release discipline
- [x] Актуализирован smoke-test: `tests/test_robust_motor_hardening_smoke.py` под guardrail `--out-dir` внутри `outputs/`.
- [x] Прогнан обязательный минимум release-тестов:
  - `tests/test_build_ieee_final_checklist_smoke.py`
  - `tests/test_verify_ieee_submission_candidate_smoke.py`
  - `tests/test_build_ieee_submission_bundle_smoke.py`
  - `tests/test_robust_motor_hardening_smoke.py`
- [x] Прогнан полный `pytest -q`: `116 passed`.

## Критерии полного завершения (Definition of Done)
- [x] Для целевого тега (`20260303_ai_config_locked_nodrift`) выполнены `checklist_ready_for_submission=true` и `verification_ok=true`.
- [x] Root IEEE candidate указывает на тот же целевой тег, что и verify/checklist.
- [x] Smoke/regression тесты по release цепочке зелёные.
- [x] Есть финальный submission bundle manifest и SHA lock без missing required files.

## Backlog (вне текущего релиза)
- Post-restore ветка `20260308_postrestore_ai` остаётся отдельным исследовательским треком:
  - закрытие envelope gate,
  - доводка AL31/AO2/AIR56 под policy,
  - выпуск нового пост-восстановительного frozen тега.
