# Аналитические инструменты для характеристик привода

Этот раздел описывает два основных скрипта для сравнения FOC и MIC AI:
статические характеристики (по нагрузке/скорости) и временные ряды при
переменной нагрузке.

## 1) Нагрузочные и рабочие характеристики

Скрипт: `mic_ai/tools/drive_characteristics_ai.py`.

Пример (FOC vs MIC AI по контрольным точкам нагрузки):

```
python -m mic_ai.tools.drive_characteristics_ai \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-pu 0.8 \
  --load-values 0.0,0.03,0.06,0.09,0.12 \
  --t-end 1.2 \
  --dt 0.001 \
  --window-frac 0.25 \
  --out-dir outputs/drive_characteristics
```

Выходные файлы:
- `outputs/drive_characteristics/load_characteristics.*`
- `outputs/drive_characteristics/working_characteristics.*`
- `outputs/drive_characteristics/*_filtered.csv` (после фильтра по допуску скорости)

## 2) Временные ряды при переменной нагрузке

Скрипт: `mic_ai/tools/timeseries_compare.py`.

Пример (ступенчатый профиль нагрузки):

```
python -m mic_ai.tools.timeseries_compare \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-pu 0.8 \
  --t-end 1.2 \
  --dt 0.001 \
  --load-profile step \
  --load-steps "0:0.0,0.4:0.06,0.8:0.12" \
  --out-dir outputs/timeseries_compare
```

Выходные файлы:
- `outputs/timeseries_compare/timeseries_compare.{png,pdf,svg}`
- `outputs/timeseries_compare/timeseries_foc.csv`
- `outputs/timeseries_compare/timeseries_mic_ai.csv`

## 3) Один номинальный режим (скорость + момент)

Скрипт: `mic_ai/tools/nominal_case.py`.

Пример (номинальная скорость и момент на валу):

```
python -m mic_ai.tools.nominal_case \
  --env-config config/env_demo_true_motor1.py \
  --ai-checkpoint outputs/ai_id_ref/checkpoints/env_demo_true_motor1/best_actor.pth \
  --ai-mode ai_id_ref \
  --ai-id-relative \
  --delta-id-max 0.1 \
  --omega-ref-rpm 1450 \
  --load-torque 1.65 \
  --t-end 6.0 \
  --dt 0.001 \
  --out-dir outputs/nominal_case
```

Выходные файлы:
- `outputs/nominal_case/timeseries_compare.{png,pdf,svg}`
- `outputs/nominal_case/summary.csv`

## Принятые метрики

- Активная электрическая мощность:
  `P_эл(t) = v_a i_a + v_b i_b + v_c i_c`
- RMS-ток статора:
  `I_rms(t) = sqrt((i_a^2 + i_b^2 + i_c^2) / 3)`
- Механическая мощность:
  `P_мех(t) = omega(t) * M_эл(t)`

## 4) Калибровка потерь (loss_inv_r, loss_core_k)

Скрипт: `mic_ai/tools/calibrate_losses.py`.

Назначение: по CSV с временными рядами оценить коэффициенты потерь для
`p_in_total` (инвертор + железо), чтобы модель лучше совпадала с экспериментом.

Пример (на данных scenario_compare):

```
python -m mic_ai.tools.calibrate_losses \
  --dir outputs/scenario_compare_nominal_rule_id1p0 \
  --pattern "*_foc.csv" \
  --omega-col omega \
  --i-rms-col i_rms \
  --p-el-col p_el \
  --p-mech-col p_mech \
  --omega-exp 1.0 \
  --psi-exp 0.0 \
  --clip-negative \
  --write-snippet outputs/loss_snippet.txt
```

Вставьте полученные параметры в конфиг (например `env_demo_true_motor1_physical.py`).

Совет: если есть `Rs/Rr/B` и каналы `i_dr/i_qr`, можно вычесть медные и механические потери:

```
python -m mic_ai.tools.calibrate_losses \
  --csv path/to/log.csv \
  --config config/env_demo_true_motor1_physical.py \
  --i-dr-col i_dr --i-qr-col i_qr \
  --subtract-copper --subtract-mech
```

Подбор показателей степеней (grid search):

```
python -m mic_ai.tools.calibrate_losses \
  --csv path/to/log.csv \
  --psi-col psi_s \
  --omega-exp-range 0.8,2.0 \
  --psi-exp-range 1.5,2.5 \
  --omega-exp-grid 5 \
  --psi-exp-grid 5 \
  --write-report outputs/loss_report.json
```

## 5) Guardrails-проверка (регрессии)

Скрипт: `mic_ai/tools/guardrails_check.py`.

Проверяет, что:
- ошибка скорости не хуже FOC (err_ok)
- экономия мощности выше порога

Пример:

```
python -m mic_ai.tools.guardrails_check \
  --summary outputs/bench_v3_physical/summary.json \
  --min-power-saving-pct 0.0
```

## 6) Бенчмарк 1 командой

Скрипт: `mic_ai/tools/run_benchmark.py`.

Пример (rule-based MIC + V3, потери учтены):

```
python -m mic_ai.tools.run_benchmark \
  --env-config config/env_demo_true_motor1_physical.py \
  --out-dir outputs/bench_v3_physical \
  --mic-id-ref-low 1.0 --mic-id-ref-high 1.4 \
  --include-v3 --use-total-power \
  --min-power-saving-pct 0.0
```

Опционально можно сразу проверить регрессию по baseline:

```
python -m mic_ai.tools.run_benchmark \
  --env-config config/env_demo_true_motor1_physical.py \
  --out-dir outputs/bench_v3_physical \
  --mic-id-ref-low 1.0 --mic-id-ref-high 1.4 \
  --include-v3 --use-total-power \
  --min-power-saving-pct 0.0 \
  --baseline-summary benchmarks/baseline_summary_physical_motor1.json \
  --compare-max-err-rel 0.1 \
  --compare-max-power-rel 0.1
```

## 7) Сравнение summary с базовой линией (регрессии)

Скрипт: `mic_ai/tools/compare_summary.py`.

Назначение: сравнить `summary.json` текущего прогона с эталоном и проверить,
что ошибка/потребление не выросли больше допусков.

Пример:

```
python -m mic_ai.tools.compare_summary \
  --baseline benchmarks/baseline_summary_physical_motor1.json \
  --current outputs/bench_v3_physical/summary.json \
  --max-err-rel 0.1 \
  --max-power-rel 0.1 \
  --no-require-err-ok \
  --report outputs/bench_v3_physical/compare_report.json
```

## 8) Подбор id_ref по сетке (минимум мощности)

Скрипт: `mic_ai/tools/id_ref_sweep.py`.

Пример:

```
python -m mic_ai.tools.id_ref_sweep \
  --env-config config/env_demo_true_motor1_physical.py \
  --scenario speed_step:0.2 \
  --id-ref-min 0.2 --id-ref-max 2.0 --id-ref-steps 12 \
  --t-end 1.2 --dt 0.001 \
  --use-total-power \
  --out-dir outputs/id_ref_sweep_motor1
```

## 9) Таблица id_ref (LUT) по скорости и нагрузке

Скрипт: `mic_ai/tools/id_ref_lut.py`.

Назначение: построить LUT `id_ref(omega_ref, load_torque)` с ограничением
на ошибку скорости относительно базового FOC (`foc.id_ref`). В LUT всегда
добавляется базовый `id_ref`, чтобы можно было вернуться к эталону.

Пример (диапазоны в pu по скорости, базовая нагрузка 0.4 Н*м):

```
python -m mic_ai.tools.id_ref_lut \
  --env-config config/env_demo_true_motor1_physical.py \
  --omega-ref-range 0.3,1.1 --omega-ref-pu --omega-ref-steps 5 \
  --load-range 0.2,0.6 --load-steps 5 \
  --id-ref-min 0.2 --id-ref-max 1.2 --id-ref-steps 10 \
  --use-total-power \
  --error-tol-rel 0.0 \
  --out-dir outputs/id_ref_lut_motor1
```

После генерации добавьте в конфиг:

```
id_ref_lut_path = "outputs/id_ref_lut_motor1/id_ref_lut.json"
```
