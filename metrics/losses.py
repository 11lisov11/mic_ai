from __future__ import annotations

import numpy as np

K_FE_DEFAULT = 1e-4
K_SW_DEFAULT = 1e-5


def estimate_p_loss(
    i_d: np.ndarray,
    i_q: np.ndarray,
    omega_e: np.ndarray,
    *,
    rs: float,
    lm: float,
    f_sw: float,
    k_fe: float = K_FE_DEFAULT,
    k_sw: float = K_SW_DEFAULT,
) -> np.ndarray:
    """
    Simplified loss model:
    - Copper: P_cu ≈ 1.5 * Rs * (id^2 + iq^2)
    - Iron:   P_fe ≈ k_fe * omega_e^2 * psi^2, psi ≈ Lm * id
    - Switching: P_sw ≈ k_sw * f_sw * |i|
    """
    i_d = np.asarray(i_d, dtype=float)
    i_q = np.asarray(i_q, dtype=float)
    omega_e = np.asarray(omega_e, dtype=float)
    p_cu = 1.5 * float(rs) * (i_d * i_d + i_q * i_q)
    psi = float(lm) * i_d
    p_fe = float(k_fe) * (omega_e * omega_e) * (psi * psi)
    p_sw = float(k_sw) * float(f_sw) * np.sqrt(i_d * i_d + i_q * i_q)
    return p_cu + p_fe + p_sw


__all__ = ["estimate_p_loss", "K_FE_DEFAULT", "K_SW_DEFAULT"]
