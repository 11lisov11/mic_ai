"""
Demo config with known true parameters for identification self-check.

It mirrors the default ENV but shortens simulation time for faster tests.
"""

from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env, SimulationParams


# Base environment
_base = create_default_env()

# Simulation horizon tuned for identification
_sim = replace(
    _base.sim,
    t_end=3.0,   # дольше для устойчивости locked-rotor
    dt=1e-3,
    save_prefix="demo_selfcheck",
)

# Final ENV exposed for make_env_from_config
ENV = replace(_base, sim=_sim)

# Optional identification tuning (read via getattr in auto_id)
# These will be pulled via getattr even though EnvConfig is frozen,
# because we attach them to the module-level ENV via replace() not allowed;
# so we store them separately in module attributes for easy access.
ident_u_d_step = 180.0
ident_total_time = 2.0
ident_u_q_step = 280.0
ident_locked_total_time = 2.5
ident_torque_ref = 2.5
ident_runup_time = 1.0
ident_coast_time = 1.0


__all__ = ["ENV"]
