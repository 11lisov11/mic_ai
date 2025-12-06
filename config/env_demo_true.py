"""
Демонстрационная конфигурация с известными истинными параметрами для самопроверки идентификации.

Повторяет базовый ENV, но сокращает время моделирования ради быстрых тестов.
"""

from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env, SimulationParams


# Базовое окружение
_base = create_default_env()

# Горизонт моделирования под задачи идентификации
_sim = replace(
    _base.sim,
    t_end=3.0,   # дольше для устойчивости locked-rotor
    dt=1e-3,
    save_prefix="demo_selfcheck",
)

# Финальный ENV, который отдаём make_env_from_config
ENV = replace(_base, sim=_sim)

# Дополнительные настройки идентификации (читаются через getattr в auto_id)
# Даже при «замороженном» EnvConfig читаем через getattr, поэтому храним
# их как отдельные атрибуты модуля для удобного доступа.
ident_u_d_step = 180.0
ident_total_time = 2.0
ident_u_q_step = 280.0
ident_locked_total_time = 2.5
ident_torque_ref = 2.5
ident_runup_time = 1.0
ident_coast_time = 1.0


__all__ = ["ENV"]
