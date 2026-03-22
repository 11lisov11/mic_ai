from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.ai_env import compute_energy_reward_gate


def test_compute_energy_reward_gate_hard_mode() -> None:
    assert compute_energy_reward_gate(speed_err_abs=0.5, speed_tol=1.0, mode="hard") == 1.0
    assert compute_energy_reward_gate(speed_err_abs=1.5, speed_tol=1.0, mode="hard") == 0.0


def test_compute_energy_reward_gate_soft_mode_respects_min_scale() -> None:
    gate = compute_energy_reward_gate(
        speed_err_abs=2.0,
        speed_tol=1.0,
        mode="soft",
        min_scale=0.2,
        exponent=1.0,
    )
    assert gate == 0.5

    gate_min = compute_energy_reward_gate(
        speed_err_abs=10.0,
        speed_tol=1.0,
        mode="soft",
        min_scale=0.2,
        exponent=2.0,
    )
    assert gate_min == 0.2
