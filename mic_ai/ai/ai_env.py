from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from control.vector_foc import FocController
from models.transformations import abc_to_dq, dq_to_abc
from .world_model import SimpleWorldModel, WorldModel
from .curiosity import WorldModelCuriosity


@dataclass
class AiEnvConfig:
    episode_steps: int
    dt: float
    omega_ref: float
    w_speed_error: float
    w_current_rms: float
    delta_iq_max: float = 0.5
    control_mode: str = "foc_assist"  # "foc_assist", "ai_speed", "ai_current", or "direct"
    i_base: float = 1.0
    i_max: float | None = None
    v_max: float | None = None
    lambda_int: float = 0.05
    wm_lr: float = 1e-4
    curiosity_beta: float = 0.1
    wm_batch_size: int = 32
    wm_update_interval: int = 10
    w_ext_scale: float = 1.0
    w_int_scale: float = 0.1
    sigma_omega: float = 0.05
    sigma_id: float = 0.03
    sigma_iq: float = 0.03
    drift_every_episodes: int = 5
    drift_scale: float = 0.04
    enable_id_control: bool = False
    delta_id_max: float = 0.5
    phase: str = "improve"  # "explore" or "improve"
    action_penalty: float = 0.01
    baseline_speed_err: float = 0.0
    baseline_current_rms: float = 0.0
    ext_scale: float = 1.0
    r_int_clip: float = 3.0
    reward_clip: float = 5.0
    omega_ref_max: float | None = None
    w_action_delta: float = 0.02
    w_action_activity: float = 0.02
    w_ai_voltage_speed: float = 0.3
    w_ai_voltage_current: float = 0.05
    w_ai_voltage_power: float = 0.2
    w_ai_voltage_action: float = 0.02
    ai_voltage_speed_tol: float = 0.5
    w_ai_id_speed: float = 1.0
    w_ai_id_power: float = 2.0
    w_ai_id_smooth: float = 0.05
    w_ai_id_mag: float = 0.0
    id_ref_alpha: float = 1.0
    id_ref_rate_limit: float | None = None
    id_ref_gate_speed_tol: float | None = None
    id_ref_gate_speed_tol_rel: float | None = None
    id_ref_gate_min_scale: float = 0.0
    id_ref_gate_exponent: float = 1.0
    id_ref_min: float = 0.0
    id_ref_max: float | None = None
    ai_id_ref_relative: bool = False
    ai_id_speed_tol: float = 0.5
    ai_id_speed_tol_rel: float | None = None
    foc_assist_reward_mode: str = "baseline"
    w_foc_speed: float = 1.0
    w_foc_power: float = 0.5
    w_foc_current: float = 0.1
    w_foc_action: float = 0.01
    foc_speed_tol: float = 0.5
    p_el_tau: float = 0.02
    load_torque_override: float | None = None
    curriculum_omega_pu: tuple[float, ...] = (0.3, 0.5, 0.7)
    curriculum_stage_episodes: tuple[int, ...] = (150, 300, 450)
    omega_piecewise_steps: tuple[int, ...] = (150, 300)
    omega_piecewise_multipliers: tuple[float, ...] = (1.0, 0.5, 1.2)
    override_load_torque: bool = True
    exploration_sigma_start: float = 0.3
    exploration_sigma_end: float = 0.05
    exploration_sigma_decay_episodes: int = 100
    i_soft_limit: float = 1.2
    i_soft_penalty: float = 0.5
    i_hard_limit: float = 2.0
    reward_min: float = -3.0
    reward_max: float = 0.5
    reward_min_td3: float = -5.0
    reward_max_td3: float = 0.0
    override_omega_ref: bool = True


class MicAiAIEnv:
    """
    Thin wrapper over an existing motor environment to expose a Gym-like API for AI agents.
    The wrapper keeps the underlying FOC/Vf controller intact and converts agent actions
    (delta to speed/current references) into commands for the base environment.
    """

    def __init__(
        self,
        base_env: Any,
        ai_config: AiEnvConfig,
        curiosity: WorldModelCuriosity | None = None,
        world_model: SimpleWorldModel | None = None,
        world_input_keys: List[str] | None = None,
        world_target_keys: List[str] | None = None,
    ):
        self.base_env = base_env
        self.cfg = ai_config
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

        self.dt = float(getattr(base_env, "dt", ai_config.dt))
        self._omega_base = max(float(getattr(base_env, "omega_base", ai_config.omega_ref)), 1e-6)
        self._omega_norm_base = max(abs(float(ai_config.omega_ref)), 1e-6)
        self._omega_ref_max = float(ai_config.omega_ref_max) if ai_config.omega_ref_max is not None else self._omega_norm_base
        self._omega_nominal = self._omega_norm_base
        self._iq_to_speed_gain = 2.0
        self._last_obs_norm: Dict[str, float] | None = None
        self._i_base = max(float(ai_config.i_base), 1e-6)
        self._i_max = float(ai_config.i_max) if ai_config.i_max is not None else None
        self._v_max = float(ai_config.v_max) if ai_config.v_max is not None else None
        self._v_nominal = 1.0
        self._v_scale = 0.15
        self._prev_delta_rel = 0.0
        self._prev_action_id_rel = 0.0
        self._prev_vd_norm = 0.0
        self._prev_vq_norm = 0.0
        self._omega_prev = 0.0
        self._prev_id_ref = 0.0
        self._id_ref_base = 0.0
        self.control_mode = str(ai_config.control_mode).lower()
        self._cum_r_ext = 0.0
        self._cum_r_int = 0.0
        self._cum_p_in = 0.0
        self._cum_p_in_pos = 0.0
        self._cum_i_rms_abc = 0.0
        self._p_el_filt = 0.0
        self._p_el_filt = 0.0
        # Use a single source of truth for episode length; allow optional alias episode_max_steps on config.
        self.episode_max_steps = int(max(getattr(ai_config, "episode_max_steps", ai_config.episode_steps), 1))
        self.cfg.episode_steps = self.episode_max_steps

        self.world_input_keys = world_input_keys or [
            "omega_norm",
            "omega_ref_norm",
            "err_norm",
            "id_norm",
            "iq_norm",
            "u_dc_norm",
            "load_torque_norm",
            "prev_delta_norm",
            "prev_delta_id_norm",
            "action_iq_norm",
            "action_id_norm",
            "action_vd_norm",
            "action_vq_norm",
        ]
        self.world_target_keys = world_target_keys or ["omega_norm", "id_norm", "iq_norm"]

        if world_model is None:
            world_model = SimpleWorldModel(len(self.world_input_keys), len(self.world_target_keys), hidden_sizes=(64, 64), lr=self.cfg.wm_lr)
        if curiosity is None:
            curiosity = WorldModelCuriosity(world_model, beta=self.cfg.curiosity_beta)
        self.world_model = world_model
        self.curiosity = curiosity

        # Support direct env without built-in controller.
        self._mode = "gym_like" if hasattr(base_env, "action_space") else "direct_voltage"
        self._controller: FocController | None = None
        self._theta_mech = 0.0
        self._last_currents_abc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_torque = 0.0
        self._t = 0.0
        self._episode_idx = 0
        self._curriculum_stage_idx = 0
        self._curriculum_ref = self.cfg.omega_ref
        self._omega_piecewise_steps = tuple(int(x) for x in getattr(self.cfg, "omega_piecewise_steps", (150, 300)))
        self._omega_piecewise_multipliers = tuple(float(x) for x in getattr(self.cfg, "omega_piecewise_multipliers", (1.0, 0.5, 1.2)))
        self._wm_buffer_x: List[np.ndarray] = []
        self._wm_buffer_y: List[np.ndarray] = []
        self._drift_motor_params = getattr(getattr(base_env, "env", None), "motor", None)
        self.phase = self.cfg.phase
        self._cum_speed_err = 0.0
        self._cum_current_rms = 0.0
        self._u_dc_norm = 1.0
        self._load_base = 1.0
        self._hard_terminated = False
        self._overcurrent_steps = 0
        base_v_limit = 1.0
        self._slip_max = max(self._omega_ref_max, self._omega_norm_base, 1e-6)

        env_cfg = getattr(base_env, "env_config", None) if self._mode == "direct_voltage" else getattr(base_env, "env", None)
        if env_cfg and hasattr(env_cfg, "motor") and hasattr(env_cfg.motor, "I_n") and ai_config.i_base <= 0:
            self._i_base = max(float(getattr(env_cfg.motor, "I_n", 1.0)), 1e-6)

        if env_cfg is not None:
            inv = getattr(env_cfg, "inverter", None)
            if inv is not None and hasattr(inv, "Vdc"):
                vdc = float(getattr(inv, "Vdc", 0.0))
                self._u_dc_norm = vdc / max(vdc, 1e-6) if vdc != 0 else 0.0
                base_v_limit = 0.8 * vdc / math.sqrt(3.0) if vdc != 0 else base_v_limit
                self._v_nominal = base_v_limit
            sim_cfg = getattr(env_cfg, "sim", None)
            if sim_cfg is not None and hasattr(sim_cfg, "load_torque"):
                self._load_base = max(abs(float(getattr(sim_cfg, "load_torque", 0.0))), 1.0)
            if self._i_max is None:
                iq_limit_cfg = getattr(getattr(env_cfg, "foc", None), "iq_limit", None)
                if iq_limit_cfg is not None:
                    self._i_max = float(iq_limit_cfg)
            foc_cfg = getattr(env_cfg, "foc", None)
            if foc_cfg is not None and getattr(foc_cfg, "id_ref", None) is not None:
                self._id_ref_base = float(getattr(foc_cfg, "id_ref", 0.0))
        if self._i_max is None or self._i_max <= 0:
            self._i_max = self._i_base
        if self.control_mode == "ai_voltage":
            cfg_v = getattr(env_cfg, "ai_v_max", None) if env_cfg is not None else None
            v_cfg = self._v_max if self._v_max is not None else cfg_v
            if v_cfg is None or v_cfg <= 0:
                self._v_max = self._v_scale * base_v_limit
            elif v_cfg <= 5.0:
                # Treat small values as per-unit multiplier of available DC amplitude.
                self._v_max = float(v_cfg * base_v_limit)
            else:
                self._v_max = float(v_cfg)
        else:
            if (self._v_max is None or self._v_max <= 0) and base_v_limit > 0:
                self._v_max = base_v_limit
        if self._v_max is None or self._v_max <= 0:
            self._v_max = 1.0

    def set_scenario(self, name: str) -> None:
        env_cfg = getattr(self.base_env, "env", None)
        if env_cfg is None or not hasattr(env_cfg, "sim"):
            return
        sim_cfg = replace(env_cfg.sim, scenario_name=str(name))
        self.base_env.env = replace(env_cfg, sim=sim_cfg)

    def _reset_direct_voltage(self) -> None:
        self.base_env.reset()
        self._t = 0.0
        self._theta_mech = 0.0
        self._last_currents_abc = (0.0, 0.0, 0.0)
        self._last_torque = 0.0

    def _prepare_gym_env(self) -> None:
        # Override scenarios to use constant references during AI training.
        if bool(getattr(self.cfg, "override_omega_ref", True)) and hasattr(self.base_env, "omega_ref_func"):
            self.base_env.omega_ref_func = lambda _t: self.cfg.omega_ref
        if hasattr(self.base_env, "load_torque_func") and self.control_mode != "ai_voltage":
            if getattr(self.cfg, "load_torque_override", None) is not None:
                load = float(getattr(self.cfg, "load_torque_override", 0.0))
                self.base_env.load_torque_func = lambda _t, load=load: load
            elif bool(getattr(self.cfg, "override_load_torque", True)):
                self.base_env.load_torque_func = lambda _t: 0.0

    def _omega_ref_from_env(self, t: float) -> float:
        if not bool(getattr(self.cfg, "override_omega_ref", True)):
            omega_ref_func = getattr(self.base_env, "omega_ref_func", None)
            if omega_ref_func is not None:
                try:
                    return float(omega_ref_func(t))
                except Exception:
                    return float(self.cfg.omega_ref)
        boundaries = list(self._omega_piecewise_steps)
        multipliers = list(self._omega_piecewise_multipliers)
        step_idx = self.step_count
        seg_idx = 0
        for boundary in boundaries:
            if step_idx >= boundary:
                seg_idx += 1
        seg_idx = min(seg_idx, len(multipliers) - 1)
        omega_target = float(self._curriculum_ref * multipliers[seg_idx])
        return omega_target

    def _load_torque_from_env(self, t: float) -> float:
        if hasattr(self.base_env, "load_torque_func"):
            return float(self.base_env.load_torque_func(t))
        env_cfg = getattr(getattr(self.base_env, "env", None), "sim", None)
        if env_cfg is not None and hasattr(env_cfg, "load_torque"):
            return float(getattr(env_cfg, "load_torque", 0.0))
        return 0.0

    def _current_rms(self, currents_abc: Tuple[float, float, float]) -> float:
        return float(np.sqrt(np.mean(np.square(currents_abc))))

    def _build_agent_obs(self, omega: float, omega_ref: float, i_d: float, i_q: float, load_torque: float = 0.0, omega_syn: float | None = None) -> Dict[str, float]:
        omega_base = self._omega_nominal
        i_base = self._i_base
        theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
        if omega_syn is None:
            controller = getattr(self.base_env, "controller", None)
            if controller is not None:
                omega_syn = float(getattr(controller, "omega_syn", 0.0))
        omega_syn = float(omega_syn if omega_syn is not None else 0.0)
        slip = omega_syn - omega
        slip_base = max(self._slip_max, abs(omega_syn), 1e-6)

        def _clip(v: float, low: float = -1.0, high: float = 1.0) -> float:
            if not math.isfinite(v):
                return 0.0
            return float(max(low, min(high, v)))

        id_norm = _clip(i_d / max(i_base, 1e-6), low=-5.0, high=5.0)
        iq_norm = _clip(i_q / max(i_base, 1e-6), low=-5.0, high=5.0)
        load_norm = _clip(load_torque / max(self._load_base, 1e-6), low=-5.0, high=5.0)
        obs = {
            "omega": float(omega) if math.isfinite(omega) else 0.0,
            "omega_ref": float(omega_ref) if math.isfinite(omega_ref) else 0.0,
            "i_d": float(i_d) if math.isfinite(i_d) else 0.0,
            "i_q": float(i_q) if math.isfinite(i_q) else 0.0,
            "omega_norm": _clip(omega / omega_base),
            "omega_ref_norm": _clip(omega_ref / omega_base),
            "err_norm": _clip((omega_ref - omega) / omega_base),
            "id_norm": id_norm,
            "iq_norm": iq_norm,
            "slip": float(slip) if math.isfinite(slip) else 0.0,
            "slip_norm": _clip(slip / slip_base),
            "prev_delta_norm": float(self._prev_delta_rel),
            "prev_delta_id_norm": float(self._prev_action_id_rel),
            "u_dc_norm": float(self._u_dc_norm),
            "load_torque_norm": load_norm,
            "sin_theta_e": _clip(math.sin(theta_e), low=-1e3, high=1e3),
            "cos_theta_e": _clip(math.cos(theta_e), low=-1e3, high=1e3),
            "last_action_vd": float(self._prev_vd_norm),
            "last_action_vq": float(self._prev_vq_norm),
        }
        return obs

    def _encode_world_input(self, obs_norm: Dict[str, float], action_norm: Dict[str, float]) -> np.ndarray:
        enriched = dict(obs_norm)
        enriched["action_iq_norm"] = float(action_norm.get("iq", 0.0))
        enriched["action_id_norm"] = float(action_norm.get("id", 0.0))
        enriched["action_vd_norm"] = float(action_norm.get("vd", 0.0))
        enriched["action_vq_norm"] = float(action_norm.get("vq", 0.0))
        enriched["delta_rel"] = enriched["action_iq_norm"]
        enriched["delta_id_rel"] = enriched["action_id_norm"]
        arr = np.array([float(enriched.get(k, 0.0)) for k in self.world_input_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
        return arr

    def _encode_world_target(self, obs_norm: Dict[str, float]) -> np.ndarray:
        arr = np.array([float(obs_norm.get(k, 0.0)) for k in self.world_target_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
        return arr

    def _apply_drift_if_needed(self) -> None:
        if self.cfg.drift_every_episodes <= 0:
            return
        if self._episode_idx % self.cfg.drift_every_episodes != 0:
            return
        env_cfg = getattr(self.base_env, "env", None)
        if env_cfg is None or not hasattr(env_cfg, "motor"):
            return
        motor = getattr(env_cfg, "motor")
        try:
            drift = 1.0 + self.cfg.drift_scale * (2.0 * np.random.rand() - 1.0)
            motor_drifted = replace(
                motor,
                Lm=float(motor.Lm * drift),
                Rr=float(motor.Rr * drift),
                B=float(motor.B * drift),
            )
            env_cfg_drifted = replace(env_cfg, motor=motor_drifted)
            # Recreate gym env if possible.
            from simulation.gym_env import InductionMotorEnv

            self.base_env = InductionMotorEnv(env_cfg_drifted)
        except Exception:
            # fail-safe: ignore drift if cannot apply
            pass

    def reset(self) -> Dict[str, float]:
        """
        Reset base environment and return initial observation dictionary.
        """
        self._episode_idx += 1
        self._apply_drift_if_needed()
        if hasattr(self.base_env, "reset"):
            _ = self.base_env.reset()
        if self._mode == "gym_like":
            self._prepare_gym_env()
        else:
            self._reset_direct_voltage()

        self.step_count = 0
        self.history = []
        self._prev_delta_rel = 0.0
        self._prev_action_id_rel = 0.0
        self._prev_vd = 0.0
        self._prev_vq = 0.0
        self._prev_vd_norm = 0.0
        self._prev_vq_norm = 0.0
        self._omega_prev = 0.0
        self._prev_id_ref = float(self._id_ref_base)
        self._cum_speed_err = 0.0
        self._cum_current_rms = 0.0
        self._cum_r_ext = 0.0
        self._cum_r_int = 0.0
        self._cum_p_in = 0.0
        self._cum_p_in_pos = 0.0
        self._cum_i_rms_abc = 0.0
        self._hard_terminated = False
        self._overcurrent_steps = 0
        stages = tuple(getattr(self.cfg, "curriculum_omega_pu", (0.3, 0.5, 0.7)))
        boundaries = tuple(getattr(self.cfg, "curriculum_stage_episodes", (150, 300, 450)))
        ep_idx = self._episode_idx
        stage_idx = 0
        for idx, boundary in enumerate(boundaries):
            if ep_idx >= boundary:
                stage_idx = idx + 1
        stage_idx = min(stage_idx, max(len(stages) - 1, 0))
        self._curriculum_stage_idx = stage_idx
        stage_pu = stages[stage_idx] if stages else 0.3
        self._curriculum_ref = stage_pu * self._omega_nominal
        if self.cfg.baseline_speed_err <= 0.0:
            self.cfg.baseline_speed_err = max(abs(self.cfg.omega_ref) * self.cfg.episode_steps, 1e-3)
        if self.cfg.baseline_current_rms <= 0.0:
            self.cfg.baseline_current_rms = max(self._i_base * self.cfg.episode_steps * 0.5, 1e-3)
        if self.cfg.ext_scale <= 0.0:
            self.cfg.ext_scale = max(
                (self.cfg.baseline_speed_err + self.cfg.baseline_current_rms) / max(self.cfg.episode_steps, 1),
                1.0,
            )
        t_now = float(getattr(self.base_env, "t", 0.0))
        omega_ref = self._omega_ref_from_env(t_now)
        self._omega_ref_max = max(self._omega_ref_max, abs(omega_ref), 1e-6)
        theta_e = 0.0
        controller = getattr(self.base_env, "controller", None)
        if controller is not None:
            theta_e = float(getattr(controller, "theta_e", 0.0))
        i_abc = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
        i_d, i_q = abc_to_dq(*i_abc, theta_e)
        motor = getattr(self.base_env, "motor", None)
        omega_meas = float(getattr(getattr(motor, "state", None), "omega_m", 0.0)) if motor is not None else 0.0
        load_torque = self._load_torque_from_env(t_now)
        obs = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d, i_q=i_q, load_torque=load_torque)
        self._last_obs_norm = obs
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

    def _extract_delta_id(self, action: Any) -> float:
        if isinstance(action, dict):
            if "delta_id_rel" in action:
                return float(action.get("delta_id_rel", 0.0))
        try:
            arr = np.asarray(action).flatten()
            if arr.size > 1:
                return float(arr[1])
        except Exception:
            pass
        return 0.0

    def _extract_current_action(self, action: Any) -> Tuple[float, float]:
        if isinstance(action, dict):
            iq = float(action.get("i_q_ref_norm", action.get("iq_ref_norm", action.get("delta_iq_rel", 0.0))))
            id_ref = float(action.get("i_d_ref_norm", action.get("id_ref_norm", action.get("delta_id_rel", 0.0))))
            return iq, id_ref
        arr = np.asarray(action).flatten()
        iq = float(arr[0]) if arr.size > 0 else 0.0
        id_ref = float(arr[1]) if arr.size > 1 else 0.0
        return iq, id_ref

    def _saturate_current_vector(self, id_norm: float, iq_norm: float) -> Tuple[float, float]:
        """Limit sqrt(id^2+iq^2) to i_max (in absolute amps)."""
        i_d = id_norm * self._i_base
        i_q = iq_norm * self._i_base
        mag = math.hypot(i_d, i_q)
        limit = max(self._i_max, 1e-6)
        if mag > limit and mag > 0:
            scale = limit / mag
            i_d *= scale
            i_q *= scale
        return float(i_q / self._i_base), float(i_d / self._i_base)

    def _step_foc_assist(
        self, delta_rel: float, delta_id_rel: float
    ) -> Tuple[np.ndarray, bool, Dict[str, Any], Tuple[float, float, float, float]]:
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
        omega_true = float(getattr(getattr(motor, "state", None), "omega_m", 0.0))
        omega_meas = omega_true + np.random.randn() * self.cfg.sigma_omega
        i_d_prev, i_q_prev = abc_to_dq(*i_abc_prev, theta_e_prev)
        i_d_prev += np.random.randn() * self.cfg.sigma_id
        i_q_prev += np.random.randn() * self.cfg.sigma_iq

        delta_rel = float(np.clip(delta_rel, -1.0, 1.0))
        delta_id_rel = float(np.clip(delta_id_rel, -1.0, 1.0))
        delta_iq_scaled = delta_rel * float(self.cfg.delta_iq_max)
        delta_id_scaled = 0.0
        if self.cfg.enable_id_control:
            delta_id_scaled = delta_id_rel * float(self.cfg.delta_id_max)

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
        id_ref_total = id_ref + delta_id_scaled * max(1.0, abs(id_ref))

        e_id = id_ref_total - i_d_prev
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
            "id_ref_total": id_ref_total,
            "delta_iq_rel": delta_rel,
            "delta_iq_applied": delta_iq_scaled,
            "action_raw": float(delta_rel),
            "load_torque": load_torque,
        }
        done = False
        return obs_raw, done, info, (i_d_next, i_q_next, delta_rel, delta_id_scaled)

    def _step_ai_speed(
        self, iq_ref_norm: float, id_ref_norm: float
    ) -> Tuple[np.ndarray, bool, Dict[str, Any], Tuple[float, float, float, float]]:
        controller: FocController | None = getattr(self.base_env, "controller", None)
        motor = getattr(self.base_env, "motor", None)
        inverter = getattr(self.base_env, "inverter", None)
        if controller is None or motor is None or inverter is None:
            raise RuntimeError("ai_speed mode requires base_env with controller, motor, and inverter")

        t_now = float(getattr(self.base_env, "t", 0.0))
        theta_mech = float(getattr(self.base_env, "theta_mech", 0.0))
        omega_ref_base = self._omega_ref_from_env(t_now)
        load_torque = self._load_torque_from_env(t_now)

        i_abc_prev = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
        theta_e_prev = float(getattr(controller, "theta_e", 0.0))
        omega_true = float(getattr(getattr(motor, "state", None), "omega_m", 0.0))
        omega_meas = omega_true + np.random.randn() * self.cfg.sigma_omega
        i_d_prev, i_q_prev = abc_to_dq(*i_abc_prev, theta_e_prev)
        i_d_prev += np.random.randn() * self.cfg.sigma_id
        i_q_prev += np.random.randn() * self.cfg.sigma_iq

        iq_ref_norm = float(np.clip(iq_ref_norm, -1.0, 1.0))
        id_ref_norm = float(np.clip(id_ref_norm, -1.0, 1.0))
        i_limit = max(self._i_max, 1e-6)
        iq_ref_cmd = float(np.clip(iq_ref_norm * self._i_base, -i_limit, i_limit))
        if self.cfg.enable_id_control:
            id_ref_cmd = float(np.clip(id_ref_norm * self._i_base, -i_limit, i_limit))
        else:
            id_ref_cmd = float(getattr(getattr(controller, "params", None), "id_ref", 0.0))

        e_id = id_ref_cmd - i_d_prev
        e_iq = iq_ref_cmd - i_q_prev
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
            "iq_ref_cmd": iq_ref_cmd,
            "id_ref_cmd": id_ref_cmd,
            "action_iq_norm": iq_ref_norm,
            "action_id_norm": id_ref_norm,
            "load_torque": load_torque,
        }
        done = False
        return obs_raw, done, info, (i_d_next, i_q_next, iq_ref_norm, id_ref_norm)

    def _step_ai_id_ref(self, id_ref_norm: float) -> Tuple[np.ndarray, bool, Dict[str, Any], Tuple[float, float, float]]:
        """
        Efficiency-assist mode: keep FOC speed loop, let agent choose only id_ref.
        """
        controller: FocController | None = getattr(self.base_env, "controller", None)
        if controller is None:
            raise RuntimeError("ai_id_ref mode requires base_env with controller")

        id_ref_norm = float(np.clip(id_ref_norm, -1.0, 1.0))
        current_limit = max(float(self._i_max if self._i_max is not None else self._i_base), 1e-6)
        id_min = max(0.0, float(getattr(self.cfg, "id_ref_min", 0.0)))
        id_max_cfg = getattr(self.cfg, "id_ref_max", None)
        id_max = float(id_max_cfg) if id_max_cfg is not None else float(current_limit)
        id_max = float(np.clip(id_max, id_min, current_limit))
        gate_scale = 1.0
        gate_tol_abs = getattr(self.cfg, "id_ref_gate_speed_tol", None)
        gate_tol_rel = getattr(self.cfg, "id_ref_gate_speed_tol_rel", None)
        if gate_tol_abs is not None or gate_tol_rel is not None:
            t_now = float(getattr(self.base_env, "t", 0.0))
            omega_ref_prev = self._omega_ref_from_env(t_now)
            omega_prev = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
            omega_ref_scale = max(abs(omega_ref_prev), 1e-6)
            gate_tol = 0.0
            if gate_tol_abs is not None:
                gate_tol = max(gate_tol, float(gate_tol_abs))
            if gate_tol_rel is not None:
                gate_tol = max(gate_tol, float(gate_tol_rel) * omega_ref_scale)
            if gate_tol > 0.0:
                err_prev = abs(omega_ref_prev - omega_prev)
                gate_scale = max(0.0, 1.0 - err_prev / gate_tol)
                gate_exp = float(getattr(self.cfg, "id_ref_gate_exponent", 1.0))
                if gate_exp != 1.0:
                    gate_scale = gate_scale**gate_exp
                gate_min = float(getattr(self.cfg, "id_ref_gate_min_scale", 0.0))
                gate_scale = max(gate_scale, gate_min)
        if bool(getattr(self.cfg, "ai_id_ref_relative", False)):
            delta = id_ref_norm * float(getattr(self.cfg, "delta_id_max", 0.5)) * gate_scale
            id_ref_cmd = self._id_ref_base + delta * max(1.0, abs(self._id_ref_base))
        else:
            id_ref_raw = id_min + 0.5 * (id_ref_norm + 1.0) * (id_max - id_min)
            id_ref_cmd = self._id_ref_base + gate_scale * (id_ref_raw - self._id_ref_base)
        id_ref_cmd = float(np.clip(id_ref_cmd, id_min, id_max))

        alpha = float(getattr(self.cfg, "id_ref_alpha", 1.0))
        if 0.0 < alpha < 1.0:
            id_ref_cmd = alpha * id_ref_cmd + (1.0 - alpha) * float(self._prev_id_ref)

        rate = getattr(self.cfg, "id_ref_rate_limit", None)
        if rate is not None:
            rate = float(rate)
            if rate > 0.0:
                max_delta = rate * self.dt
                prev = float(self._prev_id_ref)
                id_ref_cmd = prev + float(np.clip(id_ref_cmd - prev, -max_delta, max_delta))

        id_ref_cmd = float(np.clip(id_ref_cmd, id_min, id_max))

        controller.params = replace(controller.params, id_ref=float(id_ref_cmd))
        obs_raw, _reward, done_env, info = self.base_env.step(None)
        omega_meas = float(obs_raw[0] if len(obs_raw) > 0 else info.get("omega_meas", 0.0))
        theta_e = float(info.get("theta_e", 0.0))
        i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
        i_d_next, i_q_next = abc_to_dq(*i_abc, theta_e)
        info["omega_meas"] = omega_meas
        info["id_ref_cmd"] = float(id_ref_cmd)
        info["id_ref_gate_scale"] = float(gate_scale)
        info["action_id_ref_norm"] = float(id_ref_norm)
        return obs_raw, bool(done_env), info, (i_d_next, i_q_next, float(id_ref_cmd))

    def _step_ai_voltage(
        self, vd_norm: float, vq_norm: float
    ) -> Tuple[np.ndarray, bool, Dict[str, Any], Tuple[float, float, float, float]]:
        controller: FocController | None = getattr(self.base_env, "controller", None)
        motor = getattr(self.base_env, "motor", None)
        inverter = getattr(self.base_env, "inverter", None)
        if motor is None or inverter is None:
            raise RuntimeError("ai_voltage mode requires base_env with motor and inverter")

        t_now = float(getattr(self.base_env, "t", 0.0))
        theta_mech = float(getattr(self.base_env, "theta_mech", 0.0))
        omega_ref_base = self._omega_ref_from_env(t_now)
        load_torque = self._load_torque_from_env(t_now)

        i_abc_prev = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
        theta_e_prev = float(getattr(controller, "theta_e", 0.0)) if controller is not None else 0.0
        omega_true = float(getattr(getattr(motor, "state", None), "omega_m", 0.0))
        omega_meas = omega_true + np.random.randn() * self.cfg.sigma_omega
        i_d_prev, i_q_prev = abc_to_dq(*i_abc_prev, theta_e_prev)
        i_d_prev += np.random.randn() * self.cfg.sigma_id
        i_q_prev += np.random.randn() * self.cfg.sigma_iq

        alpha_lp = 0.1
        vd_norm = alpha_lp * vd_norm + (1.0 - alpha_lp) * self._prev_vd_norm
        vq_norm = alpha_lp * vq_norm + (1.0 - alpha_lp) * self._prev_vq_norm

        i_mag_prev = math.hypot(i_d_prev, i_q_prev)
        current_limit = max(float(self._i_max if self._i_max is not None else self._i_base), 1e-6)
        i_pu = i_mag_prev / current_limit
        # Gentle safety envelope: only start damping near the configured current limit.
        if i_pu > 0.95:
            vd_norm *= 0.5
            vq_norm *= 0.5
        elif i_pu > 0.8:
            vd_norm *= 0.8
            vq_norm *= 0.8

        v_limit = max(float(self._v_max), 1e-6)
        v_d_cmd = float(np.clip(vd_norm, -1.0, 1.0)) * v_limit
        v_q_cmd = float(np.clip(vq_norm, -1.0, 1.0)) * v_limit
        mag_abs = math.hypot(v_d_cmd, v_q_cmd)
        if mag_abs > v_limit and mag_abs > 0.0:
            scale = v_limit / mag_abs
            v_d_cmd *= scale
            v_q_cmd *= scale
        v_d_cmd_norm = v_d_cmd / v_limit if v_limit > 0 else 0.0
        v_q_cmd_norm = v_q_cmd / v_limit if v_limit > 0 else 0.0

        p_val = getattr(getattr(motor, "params", None), "p", getattr(controller, "p", 2))
        omega_syn = float(p_val * omega_ref_base)
        theta_e = theta_e_prev + omega_syn * self.dt

        v_abc, (v_d_out, v_q_out) = inverter.output(v_d_cmd, v_q_cmd, theta_e_prev)
        state, i_d_next, i_q_next, torque_e, omega_m_next = motor.step(
            v_d_out, v_q_out, load_torque=load_torque, dt=self.dt, omega_syn=omega_syn
        )
        invalid_state = not all(np.isfinite([i_d_next, i_q_next, torque_e, omega_m_next]))
        if invalid_state:
            i_d_next = float(np.nan_to_num(i_d_next, nan=0.0, posinf=0.0, neginf=0.0))
            i_q_next = float(np.nan_to_num(i_q_next, nan=0.0, posinf=0.0, neginf=0.0))
            torque_e = float(np.nan_to_num(torque_e, nan=0.0, posinf=0.0, neginf=0.0))
            omega_m_next = float(np.nan_to_num(omega_m_next, nan=0.0, posinf=0.0, neginf=0.0))

        theta_mech += omega_m_next * self.dt
        i_abc_next = dq_to_abc(i_d_next, i_q_next, theta_e_prev)

        self.base_env.theta_mech = theta_mech
        self.base_env.last_currents_abc = tuple(float(x) for x in i_abc_next)
        self.base_env.last_torque = float(torque_e)
        self.base_env.t = t_now + self.dt
        self.base_env.w_mech = float(omega_m_next)
        if controller is not None:
            controller.theta_e = theta_e
            controller.omega_syn = omega_syn

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
            "action_vd_norm": v_d_cmd_norm,
            "action_vq_norm": v_q_cmd_norm,
            "load_torque": load_torque,
            "invalid_state": invalid_state,
        }
        done = False
        return obs_raw, done, info, (i_d_next, i_q_next, info["action_vd_norm"], info["action_vq_norm"])

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

    def _extract_p_in(self, info: Dict[str, Any]) -> tuple[float, float]:
        p_in = info.get("p_in")
        if p_in is None:
            v_abc = info.get("v_abc", (0.0, 0.0, 0.0))
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            try:
                p_in = float(v_abc[0] * i_abc[0] + v_abc[1] * i_abc[1] + v_abc[2] * i_abc[2])
            except Exception:
                p_in = 0.0
        p_in = float(np.nan_to_num(float(p_in), nan=0.0, posinf=0.0, neginf=0.0))
        p_in = float(np.clip(p_in, -1e9, 1e9))
        return p_in, max(0.0, p_in)

    def _compute_ext_reward(self, omega: float, omega_ref: float, i_rms: float) -> tuple[float, dict]:
        speed_err = abs(omega_ref - omega)
        r_ext = -self.cfg.w_speed_error * speed_err - self.cfg.w_current_rms * i_rms
        r_ext = float(np.clip(r_ext, -self.cfg.reward_clip, 0.0))
        return r_ext, {
            "speed_err": speed_err,
            "speed_err_norm": speed_err / max(abs(omega_ref), 1e-6),
            "current_norm": i_rms / max(self._i_base, 1e-6),
            "r_tracking": -self.cfg.w_speed_error * speed_err,
            "r_current": -self.cfg.w_current_rms * i_rms,
            "i_rms": i_rms,
            "r_ext": r_ext,
        }

    def step(self, action: Any) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Run one simulation step using the provided agent action.
        Returns observation dict, total reward, done flag, and info.
        """
        if self.control_mode == "ai_voltage":
            vd_norm, vq_norm = self._extract_current_action(action)
            vd_norm = float(np.clip(vd_norm, -1.0, 1.0))
            vq_norm = float(np.clip(vq_norm, -1.0, 1.0))
            obs_prev_norm = self._last_obs_norm
            if obs_prev_norm is None:
                t_now = float(getattr(self.base_env, "t", 0.0))
                omega_ref_cur = self._omega_ref_from_env(t_now)
                theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
                i_abc_cur = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
                i_d_cur, i_q_cur = abc_to_dq(*i_abc_cur, theta_e)
                omega_cur = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
                load_cur = self._load_torque_from_env(t_now)
                obs_prev_norm = self._build_agent_obs(omega=omega_cur, omega_ref=omega_ref_cur, i_d=i_d_cur, i_q=i_q_cur, load_torque=load_cur)

            obs_raw, done_env, info, extra = self._step_ai_voltage(vd_norm, vq_norm)
            i_d_next, i_q_next, vd_applied, vq_applied = extra
            action_norm = {"vd": vd_applied, "vq": vq_applied}
            x_t = self._encode_world_input(obs_prev_norm, action_norm)
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            omega_syn = float(info.get("omega_syn", 0.0))
            load_torque = float(info.get("load_torque", 0.0))
            psi_r_est = float(getattr(getattr(getattr(self.base_env, "motor", None), "params", None), "Lm", 0.0) * i_d_next)
            momentum_estimate = float(i_q_next * psi_r_est)

            prev_vd_norm = float(self._prev_vd_norm)
            prev_vq_norm = float(self._prev_vq_norm)
            action_norm_val = float(math.hypot(vd_applied, vq_applied))
            action_delta = float(math.hypot(vd_applied - prev_vd_norm, vq_applied - prev_vq_norm))

            # Update "last action" state before building the next observation so that
            # the agent sees the action it actually just applied.
            self._prev_delta_rel = vd_applied
            self._prev_action_id_rel = vq_applied
            self._prev_vd_norm = vd_applied
            self._prev_vq_norm = vq_applied

            obs_next = self._build_agent_obs(
                omega=omega_meas,
                omega_ref=omega_ref,
                i_d=i_d_next,
                i_q=i_q_next,
                load_torque=load_torque,
                omega_syn=omega_syn,
            )
            y_true = self._encode_world_target(obs_next)

            wm_loss = 0.0
            if self.curiosity is not None and self.world_model is not None:
                self._wm_buffer_x.append(x_t)
                self._wm_buffer_y.append(y_true)
                if len(self._wm_buffer_x) >= self.cfg.wm_batch_size and (self.step_count % self.cfg.wm_update_interval == 0):
                    xs = np.stack(self._wm_buffer_x, axis=0)
                    ys = np.stack(self._wm_buffer_y, axis=0)
                    wm_loss = float(self.curiosity.update_model(xs, ys))
                    self._wm_buffer_x.clear()
                    self._wm_buffer_y.clear()
            elif self.world_model is not None:
                wm_loss = float(self.world_model.update(x_t, y_true))

            omega_ref_scale = max(abs(omega_ref), 1e-6)
            speed_err_abs = abs(omega_meas - omega_ref)
            err_norm = speed_err_abs / omega_ref_scale
            i_rms = math.sqrt(i_d_next**2 + i_q_next**2)
            current_limit = max(float(self._i_max if self._i_max is not None else self._i_base), 1e-6)
            i_rms_norm = i_rms / current_limit

            i_soft_limit = float(getattr(self.cfg, "i_soft_limit", 0.0))
            i_excess = max(0.0, float(i_rms) - max(i_soft_limit, 0.0))
            i_excess_norm = i_excess / current_limit

            v_abc = info.get("v_abc", (0.0, 0.0, 0.0))
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            try:
                i_rms_abc = float(np.sqrt(np.mean(np.square(np.asarray(i_abc, dtype=float)))))
            except Exception:
                i_rms_abc = float(i_rms)
            try:
                p_in = float(v_abc[0] * i_abc[0] + v_abc[1] * i_abc[1] + v_abc[2] * i_abc[2])
            except Exception:
                p_in = 0.0
            p_in = float(np.nan_to_num(p_in, nan=0.0, posinf=0.0, neginf=0.0))
            p_in_pos = max(0.0, p_in)
            p_base = max(float(self._v_nominal) * current_limit, 1e-6)
            p_in_norm = p_in_pos / p_base
            # Curriculum for efficiency: while far from the speed target, do not punish current.
            speed_tol = float(getattr(self.cfg, "ai_voltage_speed_tol", 0.5))
            if speed_err_abs > speed_tol:
                i_excess_norm = 0.0

            w_speed = float(getattr(self.cfg, "w_ai_voltage_speed", 1.0))
            w_current = float(getattr(self.cfg, "w_ai_voltage_current", 0.1))
            w_power = float(getattr(self.cfg, "w_ai_voltage_power", 0.2))
            w_action = float(getattr(self.cfg, "w_ai_voltage_action", 0.0))

            action_cost = action_norm_val + 0.5 * action_delta
            cost = w_speed * err_norm + w_current * i_excess_norm + w_power * p_in_norm + w_action * action_cost
            r_raw = float(getattr(self.cfg, "reward_max", 1.0)) - cost
            reward = float(np.clip(r_raw, float(getattr(self.cfg, "reward_min", -10.0)), float(getattr(self.cfg, "reward_max", 1.0))))

            self._cum_speed_err += speed_err_abs
            self._cum_current_rms += i_rms
            self._cum_i_rms_abc += i_rms_abc
            self._cum_p_in += p_in
            self._cum_p_in_pos += p_in_pos
            self._cum_r_ext += reward
            self._cum_r_int += 0.0
            self._omega_prev = omega_meas

            self._last_obs_norm = obs_next
            self.step_count += 1

            over_speed = abs(omega_meas) > 1.5 * max(self._omega_ref_max, abs(omega_ref), abs(self.cfg.omega_ref), 1e-6)
            hard_over_i = i_rms > current_limit
            if hard_over_i:
                self._overcurrent_steps += 1
            invalid_state = bool(info.get("invalid_state", False) or not np.isfinite(reward))
            done = over_speed or hard_over_i or invalid_state or self.step_count >= self.episode_max_steps
            done_reason = "none"
            if over_speed:
                done_reason = "over_speed"
            elif hard_over_i:
                done_reason = "hard_current"
            elif invalid_state:
                done_reason = "invalid_state"
            elif self.step_count >= self.episode_max_steps:
                done_reason = "max_steps"
            if done and done_reason != "max_steps":
                self._hard_terminated = True

            info.update(
                {
                    "speed_err": speed_err_abs,
                    "speed_err_norm": err_norm,
                    "current_norm": i_rms_norm,
                    "i_rms": i_rms,
                    "i_rms_abc": i_rms_abc,
                    "i_rms_norm": i_rms_norm,
                    "p_in": p_in,
                    "p_in_pos": p_in_pos,
                    "p_in_norm": p_in_norm,
                    "speed_error": omega_meas - omega_ref,
                    "i_d": float(i_d_next),
                    "i_q": float(i_q_next),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                    "action_vd_norm": vd_applied,
                    "action_vq_norm": vq_applied,
                    "action_norm": action_norm_val,
                    "wm_loss": wm_loss,
                    "r_ext": reward,
                    "r_int": 0.0,
                    "r_raw": r_raw,
                    "done_reason": done_reason,
                    "hard_over_i": hard_over_i,
                    "over_speed": over_speed,
                    "overcurrent_steps": self._overcurrent_steps,
                    "momentum_estimate": momentum_estimate,
                    "reward_sigma": getattr(self.cfg, "exploration_sigma_start", 0.0),
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_error": omega_meas - omega_ref,
                    "speed_err_norm": err_norm,
                    "i_rms": i_rms,
                    "current_rms": i_rms,
                    "i_rms_abc": i_rms_abc,
                    "p_in": p_in,
                    "p_in_pos": p_in_pos,
                    "p_in_norm": p_in_norm,
                    "omega_ref": omega_ref,
                    "reward": reward,
                    "r_ext": reward,
                    "r_int": 0.0,
                    "wm_loss": wm_loss,
                    "t": info["t"],
                    "action": (vd_applied, vq_applied),
                    "r_raw": r_raw,
                    "done_reason": done_reason if done else "",
                    "hard_over_i": hard_over_i,
                    "over_speed": over_speed,
                    "overcurrent_steps": self._overcurrent_steps,
                    "momentum_estimate": momentum_estimate,
                }
            )
            return obs_next, float(reward), bool(done), info

        if self.control_mode == "ai_id_ref":
            try:
                id_ref_norm = float(np.asarray(action).flatten()[0])
            except Exception:
                id_ref_norm = 0.0
            if not math.isfinite(id_ref_norm):
                id_ref_norm = 0.0
            id_ref_norm = float(max(-1.0, min(1.0, id_ref_norm)))

            obs_prev_norm = self._last_obs_norm
            if obs_prev_norm is None:
                t_now = float(getattr(self.base_env, "t", 0.0))
                omega_ref_cur = self._omega_ref_from_env(t_now)
                theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
                i_abc_cur = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
                i_d_cur, i_q_cur = abc_to_dq(*i_abc_cur, theta_e)
                omega_cur = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
                load_cur = self._load_torque_from_env(t_now)
                obs_prev_norm = self._build_agent_obs(omega=omega_cur, omega_ref=omega_ref_cur, i_d=i_d_cur, i_q=i_q_cur, load_torque=load_cur)

            obs_raw, done_env, info, extra = self._step_ai_id_ref(id_ref_norm)
            i_d_next, i_q_next, id_ref_cmd = extra
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            omega_syn = float(info.get("omega_syn", 0.0))
            load_torque = float(info.get("load_torque", 0.0))

            obs_next = self._build_agent_obs(
                omega=omega_meas,
                omega_ref=omega_ref,
                i_d=i_d_next,
                i_q=i_q_next,
                load_torque=load_torque,
                omega_syn=omega_syn,
            )

            omega_ref_scale = max(abs(omega_ref), 1e-6)
            speed_err_abs = abs(omega_meas - omega_ref)
            err_norm = speed_err_abs / omega_ref_scale
            i_rms = float(math.hypot(i_d_next, i_q_next))
            if not math.isfinite(i_rms):
                i_rms = 0.0
            current_limit = max(float(self._i_max if self._i_max is not None else self._i_base), 1e-6)
            p_in, p_in_pos = self._extract_p_in(info)
            p_base = max(float(self._v_nominal) * current_limit, 1e-6)
            p_in_norm = min(p_in_pos / p_base, 10.0)

            speed_tol = float(getattr(self.cfg, "ai_id_speed_tol", getattr(self.cfg, "ai_voltage_speed_tol", 0.5)))
            speed_tol_rel = getattr(self.cfg, "ai_id_speed_tol_rel", None)
            if speed_tol_rel is not None:
                speed_tol = max(speed_tol, float(speed_tol_rel) * omega_ref_scale)
            w_speed = float(getattr(self.cfg, "w_ai_id_speed", 1.0))
            w_power = float(getattr(self.cfg, "w_ai_id_power", 2.0))
            w_smooth = float(getattr(self.cfg, "w_ai_id_smooth", 0.05))
            w_mag = float(getattr(self.cfg, "w_ai_id_mag", 0.0))
            i_soft_limit = float(getattr(self.cfg, "i_soft_limit", 0.0))
            i_soft_penalty = float(getattr(self.cfg, "i_soft_penalty", 0.0))
            d_id = float(id_ref_cmd - float(self._prev_id_ref))
            d_id_norm = d_id / max(current_limit, 1e-6)
            id_dev = abs(float(id_ref_cmd) - float(self._id_ref_base))
            id_dev_norm = id_dev / max(current_limit, 1e-6)

            power_term = w_power * p_in_norm
            if speed_err_abs > speed_tol:
                power_term *= 0.0
            i_excess = max(0.0, i_rms - max(i_soft_limit, 0.0))
            i_excess_norm = i_excess / current_limit
            current_term = i_soft_penalty * i_excess_norm
            cost = w_speed * err_norm + power_term + current_term + w_smooth * (d_id_norm**2) + w_mag * id_dev_norm
            r_raw = float(getattr(self.cfg, "reward_max", 1.0)) - cost
            reward = float(np.clip(r_raw, float(getattr(self.cfg, "reward_min", -10.0)), float(getattr(self.cfg, "reward_max", 1.0))))

            hard_limit = max(current_limit, float(getattr(self.cfg, "i_hard_limit", current_limit)))
            hard_over_i = i_rms > hard_limit
            invalid_state = not (
                math.isfinite(omega_meas)
                and math.isfinite(i_d_next)
                and math.isfinite(i_q_next)
                and math.isfinite(p_in)
            )
            if invalid_state or hard_over_i:
                reward = float(getattr(self.cfg, "reward_min", -10.0))
                r_raw = reward

            self._cum_speed_err += speed_err_abs
            self._cum_current_rms += i_rms
            self._cum_p_in += p_in
            self._cum_p_in_pos += p_in_pos
            self._cum_r_ext += reward
            self._cum_r_int += 0.0
            self._last_obs_norm = obs_next
            self._prev_id_ref = float(id_ref_cmd)
            self.step_count += 1

            done = self.step_count >= self.episode_max_steps or hard_over_i or invalid_state
            info.update(
                {
                    "speed_err": speed_err_abs,
                    "speed_err_norm": err_norm,
                    "i_rms": i_rms,
                    "p_in": p_in,
                    "p_in_pos": p_in_pos,
                    "p_in_norm": p_in_norm,
                    "id_ref_cmd": float(id_ref_cmd),
                    "d_id_norm": d_id_norm,
                    "reward": reward,
                    "r_raw": r_raw,
                    "hard_over_i": hard_over_i,
                    "invalid_state": invalid_state,
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_err_norm": err_norm,
                    "i_rms": i_rms,
                    "p_in": p_in,
                    "p_in_pos": p_in_pos,
                    "p_in_norm": p_in_norm,
                    "omega_ref": omega_ref,
                    "reward": reward,
                    "r_raw": r_raw,
                    "action": (id_ref_norm,),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                }
            )
            return obs_next, float(reward), bool(done), info

        if self.control_mode == "ai_current":
            iq_ref_norm, id_ref_norm = self._extract_current_action(action)
            iq_ref_norm, id_ref_norm = np.clip([iq_ref_norm, id_ref_norm], -1.0, 1.0)
            iq_ref_norm, id_ref_norm = self._saturate_current_vector(id_ref_norm, iq_ref_norm)

            obs_prev_norm = self._last_obs_norm
            if obs_prev_norm is None:
                t_now = float(getattr(self.base_env, "t", 0.0))
                omega_ref_cur = self._omega_ref_from_env(t_now)
                theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
                i_abc_cur = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
                i_d_cur, i_q_cur = abc_to_dq(*i_abc_cur, theta_e)
                omega_cur = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
                load_cur = self._load_torque_from_env(t_now)
                obs_prev_norm = self._build_agent_obs(omega=omega_cur, omega_ref=omega_ref_cur, i_d=i_d_cur, i_q=i_q_cur, load_torque=load_cur)

            obs_raw, done_env, info, extra = self._step_ai_speed(iq_ref_norm, id_ref_norm)
            i_d_next, i_q_next, act_iq_norm, act_id_norm = extra
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            load_torque = float(info.get("load_torque", 0.0))

            obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d_next, i_q=i_q_next, load_torque=load_torque)

            speed_err = abs(omega_ref - omega_meas)
            current_rms = math.sqrt(i_d_next**2 + i_q_next**2)
            err_norm = speed_err / max(abs(omega_ref), 1e-6)
            current_norm = current_rms / max(self._i_base, 1e-6)
            reward = -self.cfg.w_speed_error * err_norm - self.cfg.w_current_rms * current_norm
            reward = float(np.clip(reward, -self.cfg.reward_clip, 0.0))

            self._cum_speed_err += speed_err
            self._cum_current_rms += current_rms
            self._cum_r_ext += reward

            self._last_obs_norm = obs_next
            self._prev_delta_rel = act_iq_norm
            self._prev_action_id_rel = act_id_norm
            self.step_count += 1
            env_cfg_sim = getattr(getattr(self.base_env, "env", None), "sim", None)
            done_sim_time = False
            if env_cfg_sim is not None and hasattr(env_cfg_sim, "t_end"):
                done_sim_time = float(getattr(self.base_env, "t", 0.0)) >= float(getattr(env_cfg_sim, "t_end", 0.0))
            done = done_env or done_sim_time or self.step_count >= self.cfg.episode_steps

            reward_info = {
                "speed_err": speed_err,
                "speed_err_norm": err_norm,
                "current_norm": current_norm,
                "i_rms": current_rms,
                "r_ext": reward,
                "r_int": 0.0,
            }
            info.update(
                {
                    **reward_info,
                    "speed_error": omega_meas - omega_ref,
                    "i_d": float(i_d_next),
                    "i_q": float(i_q_next),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                    "action_iq_norm": act_iq_norm,
                    "action_id_norm": act_id_norm,
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_error": omega_meas - omega_ref,
                    "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                    "i_rms": current_rms,
                    "current_rms": current_rms,
                    "reward": reward,
                    "r_ext": reward,
                    "r_int": 0.0,
                    "wm_loss": 0.0,
                    "t": info["t"],
                    "action": (act_iq_norm, act_id_norm),
                }
            )
            return obs_next, float(reward), bool(done), info

        if self.control_mode == "foc_assist":
            delta_rel = float(np.clip(self._extract_delta_rel(action), -1.0, 1.0))
            delta_id_rel = float(np.clip(self._extract_delta_id(action), -1.0, 1.0))
            obs_prev_norm = self._last_obs_norm
            if obs_prev_norm is None:
                t_now = float(getattr(self.base_env, "t", 0.0))
                omega_ref_cur = self._omega_ref_from_env(t_now)
                theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
                i_abc_cur = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
                i_d_cur, i_q_cur = abc_to_dq(*i_abc_cur, theta_e)
                omega_cur = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
                load_cur = self._load_torque_from_env(t_now)
                obs_prev_norm = self._build_agent_obs(omega=omega_cur, omega_ref=omega_ref_cur, i_d=i_d_cur, i_q=i_q_cur, load_torque=load_cur)

            action_norm = {"iq": delta_rel, "id": delta_id_rel}
            x_t = self._encode_world_input(obs_prev_norm, action_norm)
            obs_raw, done_env, info, extra = self._step_foc_assist(delta_rel, delta_id_rel)
            i_d_next, i_q_next, delta_used, delta_id_used = extra
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            load_torque = float(info.get("load_torque", 0.0))
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            i_rms = self._current_rms(i_abc)

            obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d_next, i_q=i_q_next, load_torque=load_torque)
            y_true = self._encode_world_target(obs_next)

            wm_loss = 0.0
            r_int = 0.0
            if self.curiosity is not None and self.world_model is not None:
                self._wm_buffer_x.append(x_t)
                self._wm_buffer_y.append(y_true)
                if len(self._wm_buffer_x) >= self.cfg.wm_batch_size and (self.step_count % self.cfg.wm_update_interval == 0):
                    xs = np.stack(self._wm_buffer_x, axis=0)
                    ys = np.stack(self._wm_buffer_y, axis=0)
                    wm_loss = float(self.curiosity.update_model(xs, ys))
                    self._wm_buffer_x.clear()
                    self._wm_buffer_y.clear()
                r_int = float(min(self.curiosity.compute_intrinsic_reward(x_t, y_true), self.cfg.r_int_clip))

            speed_err = abs(omega_ref - omega_meas)
            self._cum_speed_err += speed_err
            self._cum_current_rms += i_rms
            self._cum_r_int += r_int

            p_in, p_in_pos = self._extract_p_in(info)
            self._cum_p_in += p_in
            self._cum_p_in_pos += p_in_pos
            p_base = max(float(self._v_nominal) * max(self._i_max, self._i_base), 1e-6)
            p_in_norm = min(p_in_pos / p_base, 10.0)
            tau = float(getattr(self.cfg, "p_el_tau", 0.0))
            if tau > 0.0:
                alpha = math.exp(-self.dt / max(tau, 1e-6))
                self._p_el_filt = alpha * self._p_el_filt + (1.0 - alpha) * p_in_pos
            else:
                self._p_el_filt = p_in_pos
            p_el_norm = min(self._p_el_filt / p_base, 10.0)

            reward_mode = str(getattr(self.cfg, "foc_assist_reward_mode", "baseline")).lower()
            action_energy = delta_used**2 + delta_id_used**2
            if reward_mode == "energy":
                speed_tol = float(getattr(self.cfg, "foc_speed_tol", 0.5))
                err_norm = speed_err / max(abs(omega_ref), 1e-6)
                current_norm = i_rms / max(self._i_base, 1e-6)
                w_speed = float(getattr(self.cfg, "w_foc_speed", 1.0))
                w_power = float(getattr(self.cfg, "w_foc_power", 0.5))
                w_current = float(getattr(self.cfg, "w_foc_current", 0.1))
                w_action = float(getattr(self.cfg, "w_foc_action", 0.01))
                if speed_err > speed_tol:
                    cost = w_speed * err_norm + w_current * current_norm + w_action * action_energy
                else:
                    cost = w_speed * err_norm + w_power * p_el_norm + w_current * current_norm + w_action * action_energy
                reward = float(np.clip(float(getattr(self.cfg, "reward_max", 1.0)) - cost,
                                       float(getattr(self.cfg, "reward_min", -10.0)),
                                       float(getattr(self.cfg, "reward_max", 1.0))))
                r_ext_step = reward
                self._cum_r_ext += reward
            else:
                J_speed = self._cum_speed_err / max(self.step_count + 1, 1)
                J_current = self._cum_current_rms / max(self.step_count + 1, 1)
                delta_speed = self.cfg.baseline_speed_err - J_speed
                delta_current = self.cfg.baseline_current_rms - J_current
                r_ext_episode = self.cfg.w_speed_error * delta_speed + self.cfg.w_current_rms * delta_current
                r_ext_norm = r_ext_episode / max(self.cfg.ext_scale, 1e-6)
                r_ext_step = r_ext_norm / max(self.cfg.episode_steps, 1)
                self._cum_r_ext += r_ext_step
                if self.phase == "explore":
                    r_ext_step = 0.0
                    reward = self.cfg.w_int_scale * r_int - self.cfg.action_penalty * action_energy
                else:
                    reward = self.cfg.w_ext_scale * r_ext_step - self.cfg.action_penalty * action_energy
                reward = float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))

            self._last_obs_norm = obs_next
            self._prev_delta_rel = delta_used
            self._prev_action_id_rel = delta_id_used
            self.step_count += 1
            env_cfg_sim = getattr(getattr(self.base_env, "env", None), "sim", None)
            done_sim_time = False
            if env_cfg_sim is not None and hasattr(env_cfg_sim, "t_end"):
                done_sim_time = float(getattr(self.base_env, "t", 0.0)) >= float(getattr(env_cfg_sim, "t_end", 0.0))
            done = done_env or done_sim_time or self.step_count >= self.cfg.episode_steps

            reward_info = {
                "speed_err": speed_err,
                "speed_err_norm": speed_err / max(abs(omega_ref), 1e-6),
                "current_norm": i_rms / max(self._i_base, 1e-6),
                "i_rms": i_rms,
                "p_in": p_in,
                "p_in_pos": p_in_pos,
                "p_in_norm": p_in_norm,
                "p_el_filt": self._p_el_filt,
                "r_ext": r_ext_step,
                "r_int": r_int,
                "reward_mode": reward_mode,
            }

            info.update(
                {
                    **reward_info,
                    "speed_error": omega_meas - omega_ref,
                    "i_d": float(i_d_next),
                    "i_q": float(i_q_next),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                    "r_ext": r_ext_step,
                    "r_int": r_int,
                    "wm_loss": wm_loss,
                    "delta_used": delta_used,
                    "delta_id_used": delta_id_used,
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_error": omega_meas - omega_ref,
                    "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                    "i_rms": i_rms,
                    "current_rms": i_rms,
                    "reward": reward,
                    "r_ext": r_ext_step,
                    "r_int": r_int,
                    "wm_loss": wm_loss,
                    "t": info["t"],
                    "action": (delta_used, delta_id_used),
                }
            )
            return obs_next, float(reward), bool(done), info

        if self.control_mode == "ai_speed":
            iq_ref_norm, id_ref_norm = self._extract_current_action(action)
            obs_prev_norm = self._last_obs_norm
            if obs_prev_norm is None:
                t_now = float(getattr(self.base_env, "t", 0.0))
                omega_ref_cur = self._omega_ref_from_env(t_now)
                theta_e = float(getattr(getattr(self.base_env, "controller", None), "theta_e", 0.0))
                i_abc_cur = getattr(self.base_env, "last_currents_abc", (0.0, 0.0, 0.0))
                i_d_cur, i_q_cur = abc_to_dq(*i_abc_cur, theta_e)
                omega_cur = float(getattr(getattr(getattr(self.base_env, "motor", None), "state", None), "omega_m", 0.0))
                load_cur = self._load_torque_from_env(t_now)
                obs_prev_norm = self._build_agent_obs(omega=omega_cur, omega_ref=omega_ref_cur, i_d=i_d_cur, i_q=i_q_cur, load_torque=load_cur)
            action_norm = {"iq": iq_ref_norm, "id": id_ref_norm}
            x_t = self._encode_world_input(obs_prev_norm, action_norm)
            obs_raw, done_env, info, extra = self._step_ai_speed(iq_ref_norm, id_ref_norm)
            i_d_next, i_q_next, act_iq_norm, act_id_norm = extra
            omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
            omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
            load_torque = float(info.get("load_torque", 0.0))
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))

            obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d_next, i_q=i_q_next, load_torque=load_torque)
            y_true = self._encode_world_target(obs_next)

            wm_loss = 0.0
            r_int = 0.0
            if self.curiosity is not None and self.world_model is not None:
                self._wm_buffer_x.append(x_t)
                self._wm_buffer_y.append(y_true)
                if len(self._wm_buffer_x) >= self.cfg.wm_batch_size and (self.step_count % self.cfg.wm_update_interval == 0):
                    xs = np.stack(self._wm_buffer_x, axis=0)
                    ys = np.stack(self._wm_buffer_y, axis=0)
                    wm_loss = float(self.curiosity.update_model(xs, ys))
                    self._wm_buffer_x.clear()
                    self._wm_buffer_y.clear()
                r_int = float(min(self.curiosity.compute_intrinsic_reward(x_t, y_true), self.cfg.r_int_clip))

            speed_err = abs(omega_ref - omega_meas)
            current_rms = math.sqrt(i_d_next**2 + i_q_next**2)
            self._cum_speed_err += speed_err
            self._cum_current_rms += current_rms
            self._cum_r_int += r_int

            r_ext, reward_info = self._compute_ext_reward(omega_meas, omega_ref, current_rms)
            self._cum_r_ext += r_ext
            action_energy = act_iq_norm**2 + act_id_norm**2
            reward = r_ext - self.cfg.action_penalty * action_energy
            reward = float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))

            self._last_obs_norm = obs_next
            self._prev_delta_rel = act_iq_norm
            self._prev_action_id_rel = act_id_norm
            self.step_count += 1
            env_cfg_sim = getattr(getattr(self.base_env, "env", None), "sim", None)
            done_sim_time = False
            if env_cfg_sim is not None and hasattr(env_cfg_sim, "t_end"):
                done_sim_time = float(getattr(self.base_env, "t", 0.0)) >= float(getattr(env_cfg_sim, "t_end", 0.0))
            done = done_env or done_sim_time or self.step_count >= self.cfg.episode_steps

            reward_info.update(
                {
                    "r_int": r_int,
                    "r_ext_episode": r_ext * max(self.cfg.episode_steps, 1),
                }
            )
            info.update(
                {
                    **reward_info,
                    "speed_error": omega_meas - omega_ref,
                    "i_d": float(i_d_next),
                    "i_q": float(i_q_next),
                    "t": float(getattr(self.base_env, "t", 0.0)),
                    "r_ext": r_ext,
                    "r_int": r_int,
                    "wm_loss": wm_loss,
                    "action_iq_norm": act_iq_norm,
                    "action_id_norm": act_id_norm,
                }
            )
            self.history.append(
                {
                    "obs": obs_next,
                    "speed_error": omega_meas - omega_ref,
                    "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                    "i_rms": current_rms,
                    "current_rms": current_rms,
                    "reward": reward,
                    "r_ext": r_ext,
                    "r_int": r_int,
                    "wm_loss": wm_loss,
                    "t": info["t"],
                    "action": (act_iq_norm, act_id_norm),
                }
            )
            return obs_next, float(reward), bool(done), info

        action_dict = action if isinstance(action, dict) else {"delta_omega_ref": float(self._extract_delta_rel(action))}
        if self._mode == "gym_like":
            obs_raw, done_env, info = self._step_gym(action_dict)
        else:
            obs_raw, done_env, info = self._step_direct_voltage(action_dict)

        omega_meas = float(info.get("omega_meas", obs_raw[0] if len(obs_raw) > 0 else 0.0))
        omega_ref = float(info.get("omega_ref", self.cfg.omega_ref))
        i_d, i_q, i_abc = self._extract_currents(info)
        v_d, v_q = self._extract_voltages(info)

        obs_next = self._build_agent_obs(omega=omega_meas, omega_ref=omega_ref, i_d=i_d, i_q=i_q, load_torque=self._load_torque_from_env(float(getattr(self.base_env, "t", 0.0))))
        if self._last_obs_norm is None:
            self._last_obs_norm = obs_next
        action_norm = {"iq": float(self._extract_delta_rel(action)), "id": float(self._extract_delta_id(action))}
        x_t = self._encode_world_input(self._last_obs_norm, action_norm)
        y_true = self._encode_world_target(obs_next)

        wm_loss = 0.0
        r_int = 0.0
        if self.curiosity is not None and self.world_model is not None:
            self._wm_buffer_x.append(x_t)
            self._wm_buffer_y.append(y_true)
            if len(self._wm_buffer_x) >= self.cfg.wm_batch_size and (self.step_count % self.cfg.wm_update_interval == 0):
                xs = np.stack(self._wm_buffer_x, axis=0)
                ys = np.stack(self._wm_buffer_y, axis=0)
                wm_loss = float(self.curiosity.update_model(xs, ys))
                self._wm_buffer_x.clear()
                self._wm_buffer_y.clear()
            r_int = float(min(self.curiosity.compute_intrinsic_reward(x_t, y_true), self.cfg.r_int_clip))

        speed_err = abs(omega_ref - omega_meas)
        self._cum_speed_err += speed_err
        i_rms = self._current_rms(i_abc)
        self._cum_current_rms += i_rms
        self._cum_r_int += r_int

        r_ext, reward_info = self._compute_ext_reward(omega_meas, omega_ref, i_rms)
        self._cum_r_ext += r_ext
        total_reward = r_ext + self.cfg.w_int_scale * r_int - self.cfg.action_penalty * (float(self._extract_delta_rel(action)) ** 2)
        total_reward = float(np.clip(total_reward, -self.cfg.reward_clip, self.cfg.reward_clip))

        self._last_obs_norm = obs_next
        self._prev_delta_rel = float(self._extract_delta_rel(action))

        self.step_count += 1
        done = done_env or self.step_count >= self.cfg.episode_steps

        reward_info.update(
            {
                "r_int": r_int,
                "r_ext_episode": r_ext * max(self.cfg.episode_steps, 1),
            }
        )
        info.update(
            {
                "speed_error": omega_meas - omega_ref,
                **reward_info,
                "t": float(getattr(self.base_env, "t", self._t)),
                "action": float(self._extract_delta_rel(action)),
                "v_d": v_d,
                "v_q": v_q,
                "wm_loss": wm_loss,
            }
        )
        self.history.append(
            {
                "obs": obs_next,
                "speed_error": omega_meas - omega_ref,
                "speed_err_norm": reward_info.get("speed_err_norm", 0.0),
                "i_rms": reward_info.get("i_rms", 0.0),
                "current_rms": reward_info.get("i_rms", 0.0),
                "reward": total_reward,
                "r_ext": reward_info.get("r_ext", 0.0),
                "r_int": r_int,
                "wm_loss": wm_loss,
                "t": info["t"],
                "action": info["action"],
            }
        )
        return obs_next, float(total_reward), bool(done), info

    def episode_metrics(self) -> Dict[str, float]:
        steps = max(self.step_count, 1)
        total_reward = float(sum(float(item.get("reward", 0.0)) for item in self.history))
        total_reward_raw = float(sum(float(item.get("r_raw", item.get("reward", 0.0))) for item in self.history))
        wm_loss_mean = float(np.mean([item.get("wm_loss", 0.0) for item in self.history])) if self.history else 0.0
        mean_r_ext = self._cum_r_ext / steps
        mean_r_int = self._cum_r_int / steps
        i_rms_values = [float(item.get("i_rms", item.get("current_rms", 0.0))) for item in self.history]
        i_rms_max = float(max(i_rms_values)) if i_rms_values else 0.0
        action_norm_vals: List[float] = []
        omega_vals: List[float] = []
        momentum_vals: List[float] = []
        for item in self.history:
            act = item.get("action")
            if isinstance(act, (list, tuple)) and len(act) >= 2:
                action_norm_vals.append(float(math.hypot(float(act[0]), float(act[1]))))
            elif isinstance(act, (list, tuple)) and len(act) == 1:
                try:
                    action_norm_vals.append(abs(float(act[0])))
                except Exception:
                    action_norm_vals.append(0.0)
            elif act is not None:
                try:
                    action_norm_vals.append(abs(float(act)))
                except Exception:
                    action_norm_vals.append(0.0)
            obs_item = item.get("obs", {})
            if isinstance(obs_item, dict) and "omega" in obs_item:
                omega_vals.append(float(obs_item.get("omega", 0.0)))
            if "momentum_estimate" in item:
                momentum_vals.append(float(item.get("momentum_estimate", 0.0)))
        mean_action_norm = float(np.mean(action_norm_vals)) if action_norm_vals else 0.0
        omega_mean = float(np.mean(omega_vals)) if omega_vals else 0.0
        delta_speed = float(omega_vals[-1] - omega_vals[0]) if len(omega_vals) >= 2 else 0.0
        momentum_mean = float(np.mean(momentum_vals)) if momentum_vals else 0.0
        return {
            "mean_speed_error": self._cum_speed_err / steps,
            "mean_current_rms": self._cum_current_rms / steps,
            "mean_i_rms_abc": self._cum_i_rms_abc / steps,
            "mean_p_in": self._cum_p_in / steps,
            "mean_p_in_pos": self._cum_p_in_pos / steps,
            "total_reward": total_reward,
            "total_reward_raw": total_reward_raw,
            "mean_reward": total_reward / steps,
            "mean_r_ext": mean_r_ext,
            "mean_r_int": mean_r_int,
            "wm_loss_mean": wm_loss_mean,
            "steps": steps,
            "i_rms_max": i_rms_max,
            "action_norm": mean_action_norm,
            "hard_terminated": int(self._hard_terminated),
            "overcurrent_steps": int(self._overcurrent_steps),
            "omega_mean": omega_mean,
            "delta_speed": delta_speed,
            "momentum_mean": momentum_mean,
        }


__all__ = ["AiEnvConfig", "MicAiAIEnv"]
