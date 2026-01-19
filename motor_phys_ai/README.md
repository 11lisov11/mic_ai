# motor_phys_ai

Отдельный проект для физически осведомленного управления асинхронным двигателем. Содержит два контроллера:

- PhysModController (адаптация Kt + интегратор)
- PIController (классический PI по скорости)

## Быстрый запуск

```
python -m motor_phys_ai.training.eval \
  --env-config config/env_demo_true_motor1_nominal.py \
  --scenarios step,load,drift \
  --run-tag run_phys
```

Результаты будут сохранены в `motor_phys_ai/outputs/runs_phys/run_phys/`.
