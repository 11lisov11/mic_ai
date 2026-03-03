# MASTER PLAN NEXT: MIC/FOC/PI для 3 двигателей (Post-IEEE v2)

Дата фиксации: 2026-03-04  
База: `PROJECT_MASTER_PLAN_IEEE_3MOTORS_20260303.md` закрыт на `100%` (см. `PROJECT_MASTER_PLAN_IEEE_3MOTORS_20260303_PROGRESS_20260304.json`).

## 0) Цель следующего цикла
- Повысить научную устойчивость результата после frozen IEEE-пакета:
  - увеличить запас по AO2 (power-saving margin),
  - подтвердить стабильность на расширенном наборе seeds/возмущений,
  - подготовить «camera-ready + rebuttal-ready» контур.

---

## 1) Scope v2 (Definition of Done)

1. AO2 margin hardening завершён:
   - `avg_power_saving_pct_mean >= 0.20%` в mode1/mode2,
   - `avg_eta_gain_pct_mean >= 0`,
   - `err_failures_max <= 2`.
2. Расширенный воспроизводимый прогон завершён:
   - seeds: `101,202,303,404,505,606,707,808`,
   - добавлен stress-набор perturbation sweep.
3. Обновлённый frozen-пакет v2 собран:
   - новый tag в `paper/ieee_2026/data/step28/<tag_v2>/`.
4. Научный контур защищён тестами:
   - regression на drift по ключевым таблицам,
   - plan-completion отчёт генерируется автоматически.
5. Camera-ready handoff обновлён:
   - bundle + dossier + handoff для нового тега.

---

## 2) Рабочая матрица статусов

Формат: `TODO | IN_PROGRESS | DONE | BLOCKED`

### 2.1 AO2 hardening
- [ ] TODO Запустить parameter sweep AO2 по `id_ref_alpha`, `delta_id_max`, gating.
- [ ] TODO Построить rank-таблицу AO2 и выбрать 3 кандидата.
- [ ] TODO Выполнить full protocol для кандидатов и выбрать финальный AO2 профиль.

### 2.2 Расширенная статистика устойчивости
- [ ] TODO Добавить extended-seed режим в reproducibility контур без retrain-by-default.
- [ ] TODO Сформировать `mean/std/min/max/worst` отчёт для extended seeds.
- [ ] TODO Добавить стресс-отчёт по perturb-level sweep.

### 2.3 Reproducibility и качество
- [ ] TODO Добавить CI smoke для `tools/report_plan_completion.py`.
- [ ] TODO Добавить regression guard на `step28_ieee_summary.csv` (drift threshold).
- [ ] TODO Добавить release note generator для новых frozen-tag версий.

### 2.4 Publication operations
- [ ] TODO Подготовить camera-ready checklist для IEEE template pipeline.
- [ ] TODO Подготовить rebuttal evidence pack (таблицы/фигуры/хэши/логи).
- [ ] TODO Выпустить `submission_bundle` для нового frozen-tag v2.

---

## 3) Ближайший спринт (порядок)

1. AO2 sweep + shortlist.
2. Extended-seed evaluation.
3. Freeze нового тега v2.
4. Bundle + dossier + handoff v2.
5. Финальная проверка против drift guard.

