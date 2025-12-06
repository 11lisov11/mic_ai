from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from control.vector_foc import FocController
from models.transformations import abc_to_dq, dq_to_abc


@dataclass
class AiEnvConfig:
    episode_steps: int
    dt: float
    omega_ref: float
    w_speed_error: float
    w_current_rms: float
    delta_iq_max: float = 0.2
    control_mode: str = "foc_assist"  # "foc_assist" or "direct"
    i_base: float = 1.0
    lambda_int: float = 0.05


class MicAiAIEnv:
    """
    Thin wrapper over an existing motor environment to expose a Gym-like API for AI agents.
    The wrapper keeps the underlying FOC/Vf controller intact and converts agent actions
    (delta to speed/current references) into commands for the base environment.
    """

    def __init__(self, base_env: Any, ai_config: AiEnvConfig, curiosity: Any | None = None):
        self.base_env = base_env
        self.cfg = ai_config
        self.curiosity = curiosity
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

        self.dt = float(getattr(base_env, "dt", ai_config.dt))
        self._omega_base = max(float(getattr(base_env, "omega_base", ai_config.omega_ref)), 1e-6)
        self._omega_norm_base = max(abs(float(ai_config.omega_ref)), 1e-6)
        self._iq_to_speed_gain = 2.0
        self._last_obs_for_curiosity: Dict[str, float] | None = None
        self._i_base = max(float(ai_config.i_base), 1e-6)
        self._prev_delta_rel = 0.0
        self.control_mode = str(ai_config.control_mode).lower()

        # Support direct env without built-in controller.
        self._mode = "gym_like" if hasattr(base_env, "action_space") else "direct_voltage"
        self._controller: FocController | None = None
        self._theta_mech = 0.0
        self._last_currents_abc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_torque = 0.0
        self._t = 0.0

        if self._mode == "direct_voltage":
            env_cfg = getattr(base_env, "env_config", None)
            if env_cfg is None:
                raise ValueError("base_env must expose env_config when using direct_voltage mode")
            self._controller = FocController(env_cfg.foc, env_cfg.motor, self.dt)
            self._omega_base = (
                2.0 * np.pi * float(getattr(env_cfg.scalar_vf, "f_max", 50.0)) / float(env_cfg.motor.p)
            )
            # If user did not set i_base, approximate from nameplate current if available.
            if ai_config.i_base <= 0 and hasattr(env_cfg, "motor") and hasattr(env_cfg.motor, "I_n"):
                self._i_base = max(float(getattr(env_cfg.motor, "I_n", 1.0)), 1e-6)
        else:
            env_cfg = getattr(base_env, "env", None)
            if env_cfg and hasattr(env_cfg, "motor") and hasattr(env_cfg.motor, "I_n") and ai_config.i_base <= 0:
                self._i_base = max(float(getattr(env_cfg.motor, "I_n", 1.0)), 1e-6)

    def _reset_direct_voltage(self) -> None:
        self.base_env.reset()
        self._t = 0.0
        self._theta_mech = 0.0
        self._last_currents_abc = (0.0, 0.0, 0.0)
        self._last_torque = 0.0
        if self._controller:
            self._controller.reset()

    def _prepare_gym_env(self) -> None:
        # Override scenarios to use constant references during AI training.
        if hasattr(self.base_env, "omega_ref_func"):
            self.base_env.omega_ref_func = lambda _t: self.cfg.omega_ref
        if hasattr(self.base_env, "load_torque_func"):
            self.base_env.load_torque_func = lambda _t: 0.0

    def _omega_ref_from_env(self, t: float) -> float:
        if hasattr(self.base_env, "omega_ref_func"):
            return float(self.base_env.omega_ref_func(t))
        return float(self.cfg.omega_ref)

    def _load_torque_from_env(self, t: float) -> float:
        if hasattr(self.base_env, "load_torque_func"):
            return float(self.base_env.load_torque_func(t))
        env_cfg = getattr(getattr(self.base_env, "env", None), "sim", None)
        if env_cfg is not None and hasattr(env_cfg, "load_torque"):
            return float(getattr(env_cfg, "load_torque", 0.0))
        return 0.0

    def _current_rms(self, currents_abc: Tuple[float, float, float]) -> float:
        return float(np.sqrt(np.mean(np.square(currents_abc))))

    def _build_agent_obs(self, omega: float, omega_ref: float, i_d: float, i_q: float) -> Dict[str, float]:
        omega_base = self._omega_norm_base
        i_base = self._i_base
        return {
            "omega": float(omega),
            "omega_ref": float(omega_ref),
            "i_d": float(i_d),
            "i_q": float(i_q),
            "omega_norm": float(omega / omega_base),
            "omega_ref_norm": float(omega_ref / omega_base),
            "err_norm": float((omega_ref - omega) / omega_base),
            "id_norm": float(i_d / i_base),
            "iq_norm": float(i_q / i_base),
            "prev_delta_norm": float(self._prev_delta_rel),
        }

    def reset(self) -> Dict[str, float]:
        """
        Reset base environment and return initial observation dictionary.
        """
        if hasattr(self.base_env, "reset"):
            _ = self.base_env.reset()
        if self._mode == "gym_like":
            self._prepare_gym_env()
        else:
            self._reset_direct_voltage()

        self.step_count = 0
        self.history = []
        self._prev_delta_rel = 0.0
        if self.curiosity is not None and hasattr(self.curiosity, "prev_obs"):
            self.curiosity.prev_obs = None

        t_now = float(getattr(self.base_env, "t", 0.0))
        omega_ref = self._omega_ref_from_env(t_now)
        theta_e = 0.0
        controller = getattr(self.base_env, "controller", None)
        if controller is not None:
            theta_e = float(getattr(controller, "theta_e", 0.0))
        i_abc = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
        i_d, i_q = abc_to_dq(*i_abc, theta_e)
        motor = getattr(self.base_env, "motor", None)
        omega_meas = float(getattr(getattr(motor, "state", None), "omega_m", 0.0)) if motor is not None else 0.0
        obs = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d, i_q=i_q)
        self._last_obs_for_curiosity = obs
        return obs

    def _action_to_speed_ref(self, action: Dict[str, float]) -> float:
        base_ref = float(self.cfg.omega_ref)
        # Agent supplies delta_omega_ref; fallback to delta_iq_ref for backward compatibility.
        delta_speed = float(action.get("delta_omega_ref", 0.0))
        if "delta_omega_ref" not in action and "delta_iq_ref" in action:
            delta_speed = float(action.get("delta_iq_ref", 0.0) * self._iq_to_speed_gain)
        # Limit correction to +-30% of base_ref to give agent leverage but keep FOC stable.
        limit = 0.3 * max(abs(base_ref), 1e-3)
        delta_speed = float(np.clip(delta_speed, -limit, limit))
        return float(base_ref + delta_speed)

    def _extract_delta_rel(self, action: Any) -> float:
        if isinstance(action, dict):
            if "delta_iq_rel" in action:
                return float(action.get("delta_iq_rel", 0.0))
            if "delta_omega_ref" in action:
                return float(action.get("delta_omega_ref", 0.0))
            if "delta_iq_ref" in action:
                return float(action.get("delta_iq_ref", 0.0))
        try:
            return float(np.asarray(action).flatten()[0])
        except Exception:
            return 0.0

    def _step_foc_assist(self, action: Any) -> Tuple[np.ndarray, bool, Dict[str, Any], Tuple[float, float, float]]:
        controller: FocController | None = getattr(self.base_env, "controller", None)
        motor = getattr(self.base_env, "motor", None)
        inverter = getattr(self.base_env, "inverter", None)
        if controller is None or motor is None or inverter is None:
            raise RuntimeError("foc_assist mode requires base_env with controller, motor, and inverter")

        t_now = float(getattr(self.base_env, "t", 0.0))
        theta_mech = float(getattr(self.base_env, "theta_mech", 0.0))
        omega_ref_base = self._omega_ref_from_env(t_now)
        load_torque = self._load_torque_from_env(t_now)

        i_abc_prev = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
        theta_e_prev = float(getattr(controller, "theta_e", 0.0))
        omega_meas = float(getattr(getattr(motor, "state", None), "omega_m", 0.0))
        i_d_prev, i_q_prev = abc_to_dq(*i_abc_prev, theta_e_prev)

        delta_rel_raw = self._extract_delta_rel(action)
        delta_rel = float(np.clip(delta_rel_raw, -1.0, 1.0))
        delta_iq_scaled = delta_rel * float(self.cfg.delta_iq_max)

        iq_ref_base = float(controller.pi_speed.step(omega_ref_base - omega_meas))
        iq_limit = getattr(getattr(controller, "params", None), "iq_limit", None)
        if iq_limit is not None:
            iq_ref_base = float(np.clip(iq_ref_base, -iq_limit, iq_limit))

        iq_ref_total = iq_ref_base * (1.0 + delta_iq_scaled)
        if iq_limit is not None:
            iq_ref_total = float(np.clip(iq_ref_total, -iq_limit, iq_limit))

        id_ref = getattr(getattr(controller, "params", None), "id_ref", 0.0)
        if id_ref is None:
            id_ref = 0.0

        e_id = id_ref - i_d_prev
        e_iq = iq_ref_total - i_q_prev
        v_d = controller.pi_id.step(e_id)
        v_q = -controller.pi_iq.step(e_iq)

        v_limit = getattr(getattr(controller, "params", None), "v_limit", None)
        if v_limit is not None:
            mag = math.hypot(v_d, v_q)
            if mag > v_limit and mag > 0:
                scale = v_limit / mag
                v_d *= scale
                v_q *= scale

        omega_syn = controller.p * omega_ref_base
        controller.theta_e = theta_e_prev + omega_syn * self.dt
        controller.omega_syn = omega_syn

        v_abc, (v_d_out, v_q_out) = inverter.output(v_d, v_q, theta_e_prev)
        state, i_d_next, i_q_next, torque_e, omega_m_next = motor.step(
            v_d_out, v_q_out, load_torque=load_torque, dt=self.dt, omega_syn=omega_syn
        )

        theta_mech += omega_m_next * self.dt
        i_abc_next = dq_to_abc(i_d_next, i_q_next, theta_e_prev)

        # Keep base_env fields in sync for callers that inspect them.
        self.base_env.theta_mech = theta_mech
        self.base_env.last_currents_abc = tuple(float(x) for x in i_abc_next)
        self.base_env.last_torque = float(torque_e)
        self.base_env.t = t_now + self.dt
        self.base_env.w_mech = float(omega_m_next)

        obs_raw = np.array(
            [omega_m_next, omega_ref_base, torque_e, *i_abc_next, v_d_out, v_q_out],
            dtype=np.float32,
        )
        info: Dict[str, Any] = {
            "omega_ref": omega_ref_base,
            "omega_meas": float(omega_m_next),
            "i_abc": self.base_env.last_currents_abc,
            "v_abc": v_abc,
            "theta_e": theta_e_prev,
            "omega_syn": omega_syn,
            "iq_ref_base": iq_ref_base,
            "iq_ref_total": iq_ref_total,
            "delta_iq_rel": delta_rel,
            "delta_iq_applied": delta_iq_scaled,
            "action_raw": float(delta_rel_raw),
            "load_torque": load_torque,
        }
        done = False
        return obs_raw, done, info, (i_d_next, i_q_next, delta_rel)

    def _step_gym(self, action: Dict[str, float]) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        omega_target = self._action_to_speed_ref(action)
        omega_norm = np.clip(omega_target / self._omega_base, -1.0, 1.0)
        obs_raw, _, done, info = self.base_env.step(np.array([omega_norm], dtype=np.float32))
        if not isinstance(info, dict):
            info = {}
        info.setdefault("omega_ref", omega_target)
        info.setdefault("omega_meas", float(obs_raw[0]) if len(obs_raw) > 0 else 0.0)
        return obs_raw, bool(done), info

    def _step_direct_voltage(self, action: Dict[str, float]) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        if self._controller is None:
            raise RuntimeError("Controller is not initialized for direct voltage mode")
        omega_target = self._action_to_speed_ref(action)
        theta_e = float(getattr(self._controller, "theta_e", 0.0))
        i_dq = abc_to_dq(*self._last_currents_abc, theta_e)
        i_d, i_q = i_dq
        torque_e = self._last_torque

        v_d, v_q, theta_e, omega_syn, ctrl_info = self._controller.step(
            t=self._t,
            omega_ref=omega_target,
            omega_m=getattr(self.base_env.motor.state, "omega_m", 0.0),
            i_abc=self._last_currents_abc,
            torque_e=torque_e,
            theta_mech=self._theta_mech,
        )
        v_abc, (v_d_out, v_q_out) = self.base_env.inverter.output(v_d, v_q, theta_e)
        state, i_d_next, i_q_next, torque_e, omega_m = self.base_env.motor.step(
            v_d_out, v_q_out, load_torque=0.0, dt=self.dt, omega_syn=omega_syn
        )

        self._theta_mech += omega_m * self.dt
        self._t += self.dt
        i_abc = dq_to_abc(i_d_next, i_q_next, theta_e)
        self._last_currents_abc = tuple(float(x) for x in i_abc)
        self._last_torque = float(torque_e)

        # Keep base_env fields in sync for callers that inspect them.
        self.base_env.t = getattr(self.base_env, "t", 0.0) + self.dt
        self.base_env.theta_e = theta_e
        self.base_env.i_d = float(i_d_next)
        self.base_env.i_q = float(i_q_next)
        self.base_env.w_mech = float(omega_m)

        obs_raw = np.array([omega_m, omega_target, torque_e, *i_abc, 0.0, 0.0], dtype=np.float32)
        info: Dict[str, Any] = {
            "omega_ref": omega_target,
            "omega_meas": float(omega_m),
            "i_abc": i_abc,
            "v_abc": v_abc,
            "theta_e": theta_e,
            "omega_syn": omega_syn,
        }
        info.update(ctrl_info)
        done = False
        return obs_raw, done, info

    def _extract_currents(self, info: Dict[str, Any]) -> Tuple[float, float, Tuple[float, float, float]]:
        i_abc = info.get("i_abc")
        if i_abc is None and hasattr(self.base_env, "last_currents_abc"):
            i_abc = getattr(self.base_env, "last_currents_abc")
        if i_abc is None:
            i_abc = (0.0, 0.0, 0.0)
        theta_e = info.get("theta_e", float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0)))
        i_d, i_q = abc_to_dq(*i_abc, theta_e)
        return float(i_d), float(i_q), tuple(float(x) for x in i_abc)

    def _extract_voltages(self, info: Dict[str, Any]) -> Tuple[float, float]:
        v_abc = info.get("v_abc")
        if v_abc is None:
            v_abc = (0.0, 0.0, 0.0)
        theta_e = info.get("theta_e", 0.0)
        v_d, v_q = abc_to_dq(*v_abc, theta_e)
        return float(v_d), float(v_q)

    def _compute_reward(
        self,
        omega: float,
        omega_ref: float,
        i_rms: float,
        obs_prev: Dict[str, float],
        obs_next: Dict[str, float],
    ) -> tuple[float, dict]:
        speed_err = abs(omega_ref - omega)
        speed_err_norm = speed_err / max(abs(omega_ref), 1e-6)

        current_norm = i_rms / max(self._i_base, 1e-6)

        r_tracking = -self.cfg.w_speed_error * speed_err_norm
        r_current = -self.cfg.w_current_rms * current_norm
        reward = r_tracking + r_current
        if speed_err_norm < 0.05:
            reward += 1.0

        r_int = 0.0
        if self.curiosity is not None:
            r_int = float(self.curiosity.compute_intrinsic_reward(obs_prev, obs_next))
            reward += self.cfg.lambda_int * r_int

        reward = float(max(-5.0, min(2.0, reward)))
        return reward, {
            "speed_err": speed_err,
            "speed_err_norm": speed_err_norm,
            "current_norm": current_norm,
            "r_tracking": r_tracking,
            "r_current": r_current,
            "r_int": r_int,
            "i_rms": i_rms,
        }

    def step(self, action: Any) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Run one simulation step using the provided agent action.
        Returns observation dict, total reward, done flag, and info.
        """
        if self.control_mode == "foc_assist":
            obs_raw, done_env, info, extra = self._step_foc_assist(action)
            i_d_next, i_q_next, delta_used = extra
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            i_rms = self._current_rms(i_abc)

            prev_obs = self._last_obs_for_curiosity if self._last_obs_for_curiosity is not None else None
            obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d_next, i_q=i_q_next)
            if prev_obs is None:
                prev_obs = obs_next

            reward, reward_info = self._compute_reward(
                omega=omega_meas,
                omega_ref=omega_ref,
                i_rms=i_rms,
                obs_prev=prev_obs,
                obs_next=obs_next,
            )
            self._last_obs_for_curiosity = obs_next
            self._prev_delta_rel = delta_used

            self.step_count += 1
            env_cfg_sim = getattr(getattr(self.base_env, "env", None), "sim", None)
            done_sim_time = False
            if env_cfg_sim is not None and hasattr(env_cfg_sim, "t_end"):
                done_sim_time = float(getattr(self.base_env, "t", 0.0)) >= float(getattr(env_cfg_sim, "t_end", 0.0))
            done = done_env or done_sim_time or self.step_count >= self.cfg.episode_steps

            info.update(
                {
                    **reward_info,
                    "speed_error": omega_meas - omega_ref,
                    "i_rms": i_rms,
                    "i_d": float(i_d_next),
                    "i_q": float(i_q_next),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_error": omega_meas - omega_ref,
                    "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                    "i_rms": i_rms,
                    "reward": reward,
                    "t": info["t"],
                    "action": delta_used,
                }
            )
            return obs_next, float(reward), bool(done), info

        # Fallback: legacy direct speed delta control.
        action_dict = action if isinstance(action, dict) else {"delta_omega_ref": float(self._extract_delta_rel(action))}
        if self._mode == "gym_like":
            obs_raw, done_env, info = self._step_gym(action_dict)
        else:
            obs_raw, done_env, info = self._step_direct_voltage(action_dict)

        omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
        omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
        i_d, i_q, i_abc = self._extract_currents(info)
        v_d, v_q = self._extract_voltages(info)

        obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d, i_q=i_q)
        prev_obs = self._last_obs_for_curiosity if self._last_obs_for_curiosity is not None else obs_next
        i_rms = self._current_rms(i_abc)
        total_reward, reward_info = self._compute_reward(
            omega=omega_meas, omega_ref=omega_ref, i_rms=i_rms, obs_prev=prev_obs, obs_next=obs_next
        )
        self._last_obs_for_curiosity = obs_next
        self._prev_delta_rel = float(self._extract_delta_rel(action))

        self.step_count += 1
        done = done_env or self.step_count >= self.cfg.episode_steps

        info.update(
            {
                "speed_error": omega_meas - omega_ref,
                **reward_info,
                "t": float(getattr(self.base_env, "t", self._t)),
                "action": float(self._extract_delta_rel(action)),
                "v_d": v_d,
                "v_q": v_q,
            }
        )
        self.history.append(
            {
                "obs": obs_next,
                "speed_error": omega_meas - omega_ref,
                "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                "i_rms": reward_info.get("i_rms", 0.0),
                "reward": total_reward,
                "t": info["t"],
                "action": info["action"],
            }
        )
        return obs_next, float(total_reward), bool(done), info


__all__ = ["AiEnvConfig", "MicAiAIEnv"]
