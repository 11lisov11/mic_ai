from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.env import EnvConfig, create_default_env
from control.vector_foc import FocController
from control.v3_ternary import V3Controller
from control.scalar_vf import ScalarVfController
from control.esc_foc import EscFocController
from control.aff_foc import AffFocController
from control.ebs_foc import EbsFocController
from control.lmc_foc import LmcFocController
from control.load_map_foc import LoadMapFocController, LoadMapParams
from control.hybrid_v3_foc import HybridV3FocController
from metrics.losses import estimate_p_loss, K_FE_DEFAULT, K_SW_DEFAULT
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import dq_to_abc
from simulation.scenarios import get_scenario
from safety.supervisor import SafetySupervisor, SafetyLimits

from .base_driver import BaseDriver


def _load_env_config(path: str | Path) -> EnvConfig:
    from mic_ai.core.env import make_env_from_config

    env = make_env_from_config(str(path))
    return env.env_config


class SimDriver(BaseDriver):
    """Simulation driver implementing the BaseDriver API."""

    def __init__(self, env_config: EnvConfig | str | Path | None = None) -> None:
        if env_config is None:
            self._env = create_default_env()
        elif isinstance(env_config, (str, Path)):
            self._env = _load_env_config(env_config)
        else:
            self._env = env_config

        self._dt = float(self._env.sim.dt)
        self._mode = "FOC"
        self._rng = np.random.default_rng()

        self._motor = InductionMotorModel(self._env.motor)
        self._inverter = IdealInverter(self._env.inverter)
        self._controller = FocController(self._env.foc, self._env.motor, self._dt)
        self._scalar_controller = ScalarVfController(
            self._env.scalar_vf, self._dt, self._env.motor.p, self._env.inverter.Vdc
        )
        omega_base = 2.0 * math.pi * self._env.scalar_vf.f_max / self._env.motor.p
        self._v3_controller = V3Controller(self._env.foc, self._env.motor, self._dt, omega_base)
        self._esc_controller = EscFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._aff_controller = AffFocController(self._env.foc, self._env.motor, self._dt)
        self._ebs_controller = EbsFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._lmc_controller = LmcFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._lmap_controller = LoadMapFocController(self._env.foc, self._env.motor, self._dt)
        self._hyb_controller = HybridV3FocController(
            self._env.foc,
            self._env.motor,
            self._dt,
            omega_base,
            load_nom=float(self._env.sim.load_torque),
        )

        self._theta_mech = 0.0
        self._theta_e = 0.0
        self._omega_syn = 0.0
        self._t = 0.0
        self._last_id = 0.0
        self._last_iq = 0.0
        self._rr = float(getattr(self._env.motor, "Rr", 0.0))
        self._lr = float(getattr(self._env.motor, "Lr_sigma", 0.0) + getattr(self._env.motor, "Lm", 0.0))

        self._sigma_omega = float(getattr(self._env.sim, "sigma_omega", 0.0) or 0.0)
        self._sigma_i_abc = float(getattr(self._env.sim, "sigma_i_abc", 0.0) or 0.0)

        self._omega_ref_func, self._load_torque_func = get_scenario(self._env.sim.scenario_name, self._env)

        v_nom = self._env.inverter.Vdc / math.sqrt(3.0) if self._env.inverter.Vdc > 0.0 else None
        i_nom = float(getattr(self._env.foc, "iq_limit", 0.0) or 0.0)
        if i_nom <= 0.0:
            i_nom = float(getattr(self._env.motor, "I_n", 0.0) or 0.0)
        id_ref = float(getattr(self._env.foc, "id_ref", 0.0) or 0.0)
        if i_nom > 0.0 and id_ref != 0.0:
            i_nom = math.hypot(i_nom, id_ref) * 1.3
        omega_base = 2.0 * math.pi * self._env.scalar_vf.f_max / self._env.motor.p

        self._safety = SafetySupervisor(self._dt)
        self._safety.configure(
            SafetyLimits(
                i_max=i_nom if i_nom > 0.0 else None,
                v_max=v_nom if v_nom is not None else None,
                omega_max=1.5 * omega_base if omega_base > 0.0 else None,
            )
        )

        self._pending_action = (0.0, 0.0)
        self._last_obs: Dict[str, Any] = {}
        self._last_fault: Optional[str] = None
        self._stopped = False
        self._last_i_abc = (0.0, 0.0, 0.0)
        self._k_fe = K_FE_DEFAULT
        self._k_sw = K_SW_DEFAULT

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._motor = InductionMotorModel(self._env.motor)
        self._inverter = IdealInverter(self._env.inverter)
        self._controller = FocController(self._env.foc, self._env.motor, self._dt)
        self._scalar_controller = ScalarVfController(
            self._env.scalar_vf, self._dt, self._env.motor.p, self._env.inverter.Vdc
        )
        omega_base = 2.0 * math.pi * self._env.scalar_vf.f_max / self._env.motor.p
        self._v3_controller = V3Controller(self._env.foc, self._env.motor, self._dt, omega_base)
        self._esc_controller = EscFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._aff_controller = AffFocController(self._env.foc, self._env.motor, self._dt)
        self._ebs_controller = EbsFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._lmc_controller = LmcFocController(self._env.foc, self._env.motor, self._dt, omega_base)
        self._lmap_controller = LoadMapFocController(self._env.foc, self._env.motor, self._dt)
        self._hyb_controller = HybridV3FocController(
            self._env.foc,
            self._env.motor,
            self._dt,
            omega_base,
            load_nom=float(self._env.sim.load_torque),
        )

        self._theta_mech = 0.0
        self._theta_e = 0.0
        self._omega_syn = 0.0
        self._t = 0.0
        self._last_id = 0.0
        self._last_iq = 0.0
        self._pending_action = (0.0, 0.0)
        self._stopped = False
        self._last_fault = None
        self._last_i_abc = (0.0, 0.0, 0.0)
        self._safety.reset()

        self._omega_ref_func, self._load_torque_func = get_scenario(self._env.sim.scenario_name, self._env)
        self._last_obs = self._build_obs(
            omega_ref=0.0,
            load_torque=float(getattr(self._env.sim, "load_torque", 0.0)),
            v_d=0.0,
            v_q=0.0,
            v_abc=(0.0, 0.0, 0.0),
            i_d=0.0,
            i_q=0.0,
            i_abc=(0.0, 0.0, 0.0),
            omega_m=0.0,
            torque_e=0.0,
        )

    def set_mode(self, mode: str) -> None:
        mode_upper = str(mode).upper()
        if mode_upper not in ("FOC", "MIC", "V3", "SCALAR", "ESC", "AFF", "EBS", "LMC", "LMAP", "HYBRID"):
            raise ValueError(
                f"Unknown mode '{mode}' (expected 'FOC', 'MIC', 'V3', 'SCALAR', 'ESC', 'AFF', 'EBS', 'LMC', 'LMAP' or 'HYBRID')."
            )
        if mode_upper != self._mode:
            self._mode = mode_upper
            if self._mode == "FOC":
                self._controller.reset()
            elif self._mode == "V3":
                self._v3_controller.reset()
            elif self._mode == "SCALAR":
                self._scalar_controller.reset()
            elif self._mode == "ESC":
                self._esc_controller.reset()
            elif self._mode == "AFF":
                self._aff_controller.reset()
            elif self._mode == "EBS":
                self._ebs_controller.reset()
            elif self._mode == "LMC":
                self._lmc_controller.reset()
            elif self._mode == "LMAP":
                self._lmap_controller.reset()
            elif self._mode == "HYBRID":
                self._hyb_controller.reset()
            self._pending_action = (0.0, 0.0)

    def set_limits(self, limits: Dict[str, float]) -> None:
        self._safety.configure(limits)

    def apply_action(self, vd: float, vq: float) -> None:
        if self._stopped:
            return
        self._pending_action = (float(vd), float(vq))

    def step(self) -> None:
        if self._stopped:
            return

        t = self._t
        omega_ref = float(self._omega_ref_func(t)) if self._omega_ref_func is not None else 0.0
        load_torque = float(self._load_torque_func(t)) if self._load_torque_func is not None else 0.0

        if self._mode == "FOC":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "V3":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._v3_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "SCALAR":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._scalar_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "ESC":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._esc_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "AFF":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._aff_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "EBS":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._ebs_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "LMC":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._lmc_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "LMAP":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._lmap_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=load_torque,
                theta_mech=self._theta_mech,
            )
        elif self._mode == "HYBRID":
            omega_m_true = float(self._motor.state.omega_m)
            omega_m_meas = omega_m_true
            if self._sigma_omega > 0.0:
                omega_m_meas = omega_m_true + float(self._rng.normal(0.0, self._sigma_omega))

            if self._sigma_i_abc > 0.0:
                i_abc_meas = tuple(
                    float(x + self._rng.normal(0.0, self._sigma_i_abc)) for x in self._last_i_abc
                )
            else:
                i_abc_meas = self._last_i_abc

            v_d, v_q, theta_e, omega_syn, _info = self._hyb_controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=0.0,
                theta_mech=self._theta_mech,
                load_torque=load_torque,
            )
        else:
            v_d, v_q = self._pending_action
            theta_e = self._theta_e
            omega_m = float(self._motor.state.omega_m)
            omega_slip = 0.0
            if self._rr > 0.0 and self._lr > 1e-6:
                omega_slip = (self._rr / self._lr) * (self._last_iq / max(abs(self._last_id), 1e-6))
            omega_syn = self._env.motor.p * omega_m + omega_slip

        v_d, v_q = self._safety.sanitize_action((v_d, v_q))
        if self._safety.last_fault() is not None:
            self._stopped = True
            self._last_fault = self._safety.last_fault()
            return

        v_abc, (v_d, v_q) = self._inverter.output(v_d, v_q, theta_e, self._last_i_abc)
        state, i_d, i_q, torque_e, omega_m = self._motor.step(
            v_d, v_q, load_torque, self._dt, omega_syn=omega_syn
        )

        self._theta_mech += omega_m * self._dt
        if self._mode in ("FOC", "V3", "SCALAR", "ESC", "AFF", "EBS", "LMC", "LMAP", "HYBRID"):
            if self._mode == "FOC":
                controller = self._controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "V3":
                controller = self._v3_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "SCALAR":
                controller = self._scalar_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_e)
            elif self._mode == "ESC":
                controller = self._esc_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "AFF":
                controller = self._aff_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "EBS":
                controller = self._ebs_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "LMC":
                controller = self._lmc_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            elif self._mode == "LMAP":
                controller = self._lmap_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
            else:
                controller = self._hyb_controller
                self._theta_e = float(controller.theta_e)
                self._omega_syn = float(controller.omega_syn)
        else:
            self._theta_e += omega_syn * self._dt
            self._omega_syn = float(omega_syn)

        i_abc = dq_to_abc(i_d, i_q, self._theta_e)
        self._last_i_abc = i_abc
        self._last_id = float(i_d)
        self._last_iq = float(i_q)

        self._t += self._dt

        obs = self._build_obs(
            omega_ref=omega_ref,
            load_torque=load_torque,
            v_d=v_d,
            v_q=v_q,
            v_abc=v_abc,
            i_d=i_d,
            i_q=i_q,
            i_abc=i_abc,
            omega_m=omega_m,
            torque_e=torque_e,
        )

        if self._mode == "EBS":
            p_in = v_abc[0] * i_abc[0] + v_abc[1] * i_abc[1] + v_abc[2] * i_abc[2]
            p_mech = torque_e * omega_m
            p_loss = float(
                estimate_p_loss(
                    np.array([i_d]),
                    np.array([i_q]),
                    np.array([omega_syn]),
                    rs=float(self._env.motor.Rs),
                    lm=float(self._env.motor.Lm),
                    f_sw=float(self._env.inverter.f_pwm),
                    k_fe=self._k_fe,
                    k_sw=self._k_sw,
                )[0]
            )
            # Use loss-based budget for energy supervisor.
            self._ebs_controller.update_energy(p_loss, p_mech, omega_ref, omega_m)

        aborted, reason = self._safety.check_abort(obs)
        if aborted:
            self._stopped = True
            self._last_fault = reason

        self._last_obs = obs

    def read_obs(self) -> Dict[str, Any]:
        return dict(self._last_obs)

    def get_last_fault(self) -> Optional[str]:
        if self._last_fault is not None:
            return self._last_fault
        return self._safety.last_fault()

    def close(self) -> None:
        return None

    def _build_obs(
        self,
        *,
        omega_ref: float,
        load_torque: float,
        v_d: float,
        v_q: float,
        v_abc: Tuple[float, float, float],
        i_d: float,
        i_q: float,
        i_abc: Tuple[float, float, float],
        omega_m: float,
        torque_e: float,
    ) -> Dict[str, Any]:
        return {
            "t": float(self._t),
            "omega": float(omega_m),
            "omega_ref": float(omega_ref),
            "omega_syn": float(self._omega_syn),
            "id": float(i_d),
            "iq": float(i_q),
            "ia": float(i_abc[0]),
            "ib": float(i_abc[1]),
            "ic": float(i_abc[2]),
            "u_dc": float(self._env.inverter.Vdc),
            "theta_e": float(self._theta_e),
            "torque": float(torque_e),
            "load_torque": float(load_torque),
            "v_d": float(v_d),
            "v_q": float(v_q),
            "v_a": float(v_abc[0]),
            "v_b": float(v_abc[1]),
            "v_c": float(v_abc[2]),
            "flags": {
                "stopped": bool(self._stopped),
                "fault": self.get_last_fault(),
            },
        }


__all__ = ["SimDriver"]
