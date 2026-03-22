from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.ai_env import (
    compute_energy_reward_gate,
    compute_running_eta_penalty,
    compute_terminal_energy_bonus,
)


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


def test_compute_running_eta_penalty_tracks_episode_efficiency() -> None:
    eta_running, penalty = compute_running_eta_penalty(
        cum_p_shaft_pos=0.4,
        cum_p_in_pos=1.0,
        eta_clip=1.0,
        weight=2.0,
    )
    assert eta_running == 0.4
    assert penalty == 1.2


def test_compute_running_eta_penalty_disables_without_weight() -> None:
    eta_running, penalty = compute_running_eta_penalty(
        cum_p_shaft_pos=0.4,
        cum_p_in_pos=1.0,
        eta_clip=1.0,
        weight=0.0,
    )
    assert eta_running == 0.0
    assert penalty == 0.0


def test_compute_terminal_energy_bonus_requires_tracking_and_shaft_ratio() -> None:
    assert (
        compute_terminal_energy_bonus(
            eta_energy=0.8,
            mean_speed_error=1.5,
            speed_tol=1.0,
            p_shaft_pos_total=8.0,
            p_shaft_target_total=10.0,
            reward_scale=0.4,
            eta_target=0.5,
            shaft_ratio_min=0.75,
            gate_mode="hard",
        )
        == 0.0
    )
    assert (
        compute_terminal_energy_bonus(
            eta_energy=0.8,
            mean_speed_error=0.5,
            speed_tol=1.0,
            p_shaft_pos_total=6.0,
            p_shaft_target_total=10.0,
            reward_scale=0.4,
            eta_target=0.5,
            shaft_ratio_min=0.75,
            gate_mode="hard",
        )
        == 0.0
    )


def test_compute_terminal_energy_bonus_soft_mode_scales_with_eta_and_shaft_ratio() -> None:
    bonus = compute_terminal_energy_bonus(
        eta_energy=0.9,
        mean_speed_error=2.0,
        speed_tol=1.0,
        p_shaft_pos_total=8.0,
        p_shaft_target_total=10.0,
        reward_scale=0.5,
        eta_target=0.3,
        shaft_ratio_min=0.7,
        gate_mode="soft",
        gate_min_scale=0.2,
        gate_exponent=1.0,
    )
    assert abs(bonus - 0.12) < 1e-9
