from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisor, AiIdRefSupervisorConfig
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.checkpoint_adaptation import adapt_checkpoint_state_dict_for_model
from mic_ai.tools.scenario_compare import _infer_hidden_sizes, _resolve_feature_keys
from tools.uno_q_protocol import Command, Telemetry


@dataclass(frozen=True)
class Air56IdRefParams:
    id_ref_base: float
    id_ref_min: float
    id_ref_max: float
    id_ref_alpha: float
    delta_id_max: float
    gate_speed_tol_abs: Optional[float]
    gate_speed_tol_rel: Optional[float]
    gate_min_scale: float
    gate_exponent: float
    ai_id_relative: bool
    allow_positive_delta: bool


@dataclass
class Air56PolicyBundle:
    name: str
    agent: PPOVoltageAgent
    params: Air56IdRefParams
    supervisor: Optional[AiIdRefSupervisor]


@dataclass
class Air56BridgeRuntimeState:
    last_id_ref: float
    prev_t_ms: Optional[int] = None
    last_telem_t_ms: int = 0
    prev_load_est_nm: float = 0.0
    secondary_left: int = 0


@dataclass(frozen=True)
class Air56BridgeStep:
    command: Command
    t_ms: int
    active_name: str
    omega_meas: float
    omega_ref: float
    iq_amp: float
    load_est_nm: float
    id_ref: float
    gate_scale: float
    enable_ai: int


class BaseTransport:
    def recv(self, size: int) -> bytes:
        raise NotImplementedError

    def send(self, payload: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UdpTransport(BaseTransport):
    def __init__(self, listen: str, send: str) -> None:
        listen_host, listen_port = _parse_host_port(listen)
        send_host, send_port = _parse_host_port(send)
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind((listen_host, listen_port))
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_addr = (send_host, send_port)

    def recv(self, size: int) -> bytes:
        while True:
            payload, _addr = self._rx.recvfrom(size)
            if len(payload) == size:
                return payload

    def send(self, payload: bytes) -> None:
        self._tx.sendto(payload, self._send_addr)

    def close(self) -> None:
        self._rx.close()
        self._tx.close()


class SerialTransport(BaseTransport):
    def __init__(self, port: str, baud: int, timeout_s: float) -> None:
        try:
            import serial  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyserial is required for --transport serial") from exc
        self._timeout_s = float(timeout_s)
        self._ser = serial.Serial(port=port, baudrate=int(baud), timeout=float(timeout_s))

    def recv(self, size: int) -> bytes:
        buf = bytearray()
        deadline = time.monotonic() + max(float(self._timeout_s), 1.0e-6)
        while len(buf) < size:
            chunk = self._ser.read(size - len(buf))
            if not chunk:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"serial frame timeout: got {len(buf)}/{size} bytes")
                continue
            buf.extend(chunk)
        return bytes(buf)

    def send(self, payload: bytes) -> None:
        self._ser.write(payload)
        self._ser.flush()

    def close(self) -> None:
        self._ser.close()


def _parse_host_port(text: str) -> Tuple[str, int]:
    try:
        host, port_text = str(text).rsplit(":", 1)
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"endpoint must be HOST:PORT: {text}") from exc
    host = host.strip()
    if not host:
        raise ValueError(f"endpoint host is empty: {text}")
    if port < 1 or port > 65535:
        raise ValueError(f"endpoint port out of range: {port}")
    return host, port


def _resolve_existing_file(path_text: str, label: str, *, root: Path = ROOT) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _finite_float(value: object, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite: {out}")
    return out


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if int(args.baud) <= 0:
        raise ValueError("--baud must be positive")
    serial_timeout = _finite_float(args.serial_timeout, "--serial-timeout")
    if serial_timeout <= 0.0:
        raise ValueError("--serial-timeout must be positive")
    fault_mask = int(args.fault_mask)
    if fault_mask < 0 or fault_mask > 0xFFFF:
        raise ValueError("--fault-mask must be in uint16 range")
    id_min = _finite_float(args.id_min, "--id-min")
    id_max = _finite_float(args.id_max, "--id-max")
    if id_min < 0.0:
        raise ValueError("--id-min must be non-negative")
    if id_min > id_max:
        raise ValueError("--id-min must be <= --id-max")
    cmd_rate_limit = _finite_float(args.cmd_rate_limit_a_per_s, "--cmd-rate-limit-a-per-s")
    if cmd_rate_limit < 0.0:
        raise ValueError("--cmd-rate-limit-a-per-s must be non-negative")
    if args.load_est_gain is not None:
        load_est_gain = _finite_float(args.load_est_gain, "--load-est-gain")
        if load_est_gain < 0.0:
            raise ValueError("--load-est-gain must be non-negative")
    if int(args.log_every) < 0:
        raise ValueError("--log-every must be non-negative")


def _validate_id_ref_window(*, id_ref_base: float, id_min: float, id_max: float) -> None:
    base = _finite_float(id_ref_base, "id_ref_base")
    lo = _finite_float(id_min, "id_min")
    hi = _finite_float(id_max, "id_max")
    if lo < 0.0:
        raise ValueError("--id-min must be non-negative")
    if lo > hi:
        raise ValueError("--id-min must be <= --id-max")
    if base < lo or base > hi:
        raise ValueError("FOC base id_ref must be inside --id-min/--id-max")


def _validate_id_ref_params(params: Air56IdRefParams) -> None:
    _validate_id_ref_window(
        id_ref_base=params.id_ref_base,
        id_min=params.id_ref_min,
        id_max=params.id_ref_max,
    )
    alpha = _finite_float(params.id_ref_alpha, "id_ref_alpha")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("id_ref_alpha must be in [0, 1]")
    delta_id_max = _finite_float(params.delta_id_max, "delta_id_max")
    if delta_id_max < 0.0:
        raise ValueError("delta_id_max must be non-negative")
    if params.gate_speed_tol_abs is not None:
        gate_abs = _finite_float(params.gate_speed_tol_abs, "gate_speed_tol_abs")
        if gate_abs < 0.0:
            raise ValueError("gate_speed_tol_abs must be non-negative")
    if params.gate_speed_tol_rel is not None:
        gate_rel = _finite_float(params.gate_speed_tol_rel, "gate_speed_tol_rel")
        if gate_rel < 0.0:
            raise ValueError("gate_speed_tol_rel must be non-negative")
    gate_min_scale = _finite_float(params.gate_min_scale, "gate_min_scale")
    if gate_min_scale < 0.0 or gate_min_scale > 1.0:
        raise ValueError("gate_min_scale must be in [0, 1]")
    gate_exponent = _finite_float(params.gate_exponent, "gate_exponent")
    if gate_exponent < 0.0:
        raise ValueError("gate_exponent must be non-negative")


def _status_fault(status: int, fault_mask: int) -> bool:
    status_val = int(status)
    mask = int(fault_mask)
    if mask == 0:
        return status_val != 0
    return (status_val & mask) != 0


def _clip_action_scalar(value: float) -> float:
    action = float(value)
    if not math.isfinite(action):
        raise ValueError(f"AI action must be finite: {action}")
    return float(max(-1.0, min(1.0, action)))


def _infer_action_scalar(agent: PPOVoltageAgent, obs: dict[str, float]) -> float:
    with torch.no_grad():
        state_t = agent._to_tensor(obs).unsqueeze(0)
        mu, _std, _value = agent.net(state_t)
    value = float(mu.squeeze(0).cpu().numpy()[0])
    return _clip_action_scalar(value)


def _build_obs(
    *,
    telem: Telemetry,
    omega_base: float,
    i_base: float,
    pole_pairs: int,
    rr: float,
    lr_total: float,
    load_torque_nm: float,
    load_base_nm: float,
) -> dict[str, float]:
    omega_ref = float(telem.omega_ref)
    omega = float(telem.omega_meas)
    err = omega_ref - omega
    slip = 0.0
    if lr_total > 1e-6:
        slip = (rr / lr_total) * (float(telem.i_q) / max(abs(float(telem.i_d)), 1e-6))
    omega_syn = pole_pairs * omega + slip
    slip_norm = slip / max(abs(omega_ref), 1e-6)
    return {
        "omega_norm": omega / max(omega_base, 1e-6),
        "omega_ref_norm": omega_ref / max(omega_base, 1e-6),
        "err_norm": err / max(omega_base, 1e-6),
        "id_norm": float(telem.i_d) / max(i_base, 1e-6),
        "iq_norm": float(telem.i_q) / max(i_base, 1e-6),
        "slip_norm": slip_norm,
        "load_torque_norm": float(load_torque_nm) / max(load_base_nm, 1e-6),
        "omega_syn_norm": omega_syn / max(omega_base, 1e-6),
    }


def _estimate_load_nm_from_iq(iq_amp: float, gain_nm_per_a: float) -> float:
    return float(max(0.0, abs(float(iq_amp)) * max(float(gain_nm_per_a), 0.0)))


def _compute_gate_scale(
    *,
    omega_ref: float,
    omega_meas: float,
    params: Air56IdRefParams,
) -> tuple[float, float]:
    omega_ref_scale = max(abs(float(omega_ref)), 1e-6)
    gate_tol = 0.0
    if params.gate_speed_tol_abs is not None:
        gate_tol = max(gate_tol, float(params.gate_speed_tol_abs))
    if params.gate_speed_tol_rel is not None:
        gate_tol = max(gate_tol, float(params.gate_speed_tol_rel) * omega_ref_scale)
    if gate_tol <= 0.0:
        return 1.0, 0.0
    err = abs(float(omega_ref) - float(omega_meas))
    gate_scale = max(0.0, 1.0 - err / gate_tol)
    if float(params.gate_exponent) != 1.0:
        gate_scale = gate_scale ** float(params.gate_exponent)
    gate_scale = max(gate_scale, float(params.gate_min_scale))
    return float(gate_scale), float(gate_tol)


def _action_to_id_ref(
    *,
    action: float,
    prev_id_ref: float,
    omega_ref: float,
    omega_meas: float,
    params: Air56IdRefParams,
) -> tuple[float, float, float]:
    _validate_id_ref_params(params)
    action = _clip_action_scalar(action)
    gate_scale, gate_tol = _compute_gate_scale(
        omega_ref=float(omega_ref),
        omega_meas=float(omega_meas),
        params=params,
    )
    if params.ai_id_relative:
        delta_raw = action * float(params.delta_id_max)
        if (not bool(params.allow_positive_delta)) and delta_raw > 0.0:
            delta_raw = 0.0
        delta = delta_raw * gate_scale if delta_raw < 0.0 else delta_raw
        id_ref_cmd = float(params.id_ref_base) + delta * max(1.0, abs(float(params.id_ref_base)))
    else:
        id_ref_raw = float(params.id_ref_min) + 0.5 * (action + 1.0) * (float(params.id_ref_max) - float(params.id_ref_min))
        if id_ref_raw < float(params.id_ref_base):
            id_ref_cmd = float(params.id_ref_base) + gate_scale * (id_ref_raw - float(params.id_ref_base))
        else:
            id_ref_cmd = id_ref_raw
    id_ref_cmd = float(max(float(params.id_ref_min), min(float(params.id_ref_max), id_ref_cmd)))
    if gate_scale < 0.5:
        id_ref_cmd = max(id_ref_cmd, float(params.id_ref_base))
        alpha = 1.0
    else:
        alpha = float(params.id_ref_alpha)
    if 0.0 < alpha < 1.0:
        id_ref_cmd = alpha * id_ref_cmd + (1.0 - alpha) * float(prev_id_ref)
    id_ref_cmd = float(max(float(params.id_ref_min), min(float(params.id_ref_max), id_ref_cmd)))
    return id_ref_cmd, float(gate_scale), float(gate_tol)


def _load_id_ref_params(env_cfg: object, prefix: str, *, id_ref_min: float, id_ref_max: float) -> Air56IdRefParams:
    gate_rel = getattr(env_cfg, f"{prefix}id_ref_gate_speed_tol_rel", None)
    gate_abs = getattr(env_cfg, f"{prefix}id_ref_gate_speed_tol", None)
    foc = getattr(env_cfg, "foc", None)
    id_ref_base = float(getattr(foc, "id_ref", 0.0) or 0.0)
    params = Air56IdRefParams(
        id_ref_base=id_ref_base,
        id_ref_min=float(id_ref_min),
        id_ref_max=float(id_ref_max),
        id_ref_alpha=float(getattr(env_cfg, f"{prefix}id_ref_alpha", 1.0)),
        delta_id_max=float(getattr(env_cfg, f"{prefix}delta_id_max", 0.1)),
        gate_speed_tol_abs=None if gate_abs is None else float(gate_abs),
        gate_speed_tol_rel=None if gate_rel is None else float(gate_rel),
        gate_min_scale=float(getattr(env_cfg, f"{prefix}id_ref_gate_min_scale", 0.0)),
        gate_exponent=float(getattr(env_cfg, f"{prefix}id_ref_gate_exponent", 1.0)),
        ai_id_relative=bool(getattr(env_cfg, f"{prefix}id_ref_relative", False)),
        allow_positive_delta=bool(getattr(env_cfg, f"{prefix}id_ref_allow_positive_delta", True)),
    )
    _validate_id_ref_params(params)
    return params


def _load_supervisor(env_cfg: object, enabled_attr: str, prefix: str, *, omega_nominal: float) -> Optional[AiIdRefSupervisor]:
    if not bool(getattr(env_cfg, enabled_attr, False)):
        return None
    cfg = AiIdRefSupervisorConfig(
        enabled=True,
        speed_tol_rel=float(getattr(env_cfg, f"{prefix}speed_tol_rel", 0.05)),
        speed_tol_abs=float(getattr(env_cfg, f"{prefix}speed_tol_abs", 0.0)),
        omega_min_pu=float(getattr(env_cfg, f"{prefix}omega_min", 0.1)),
        update_steps=int(getattr(env_cfg, f"{prefix}update", 20)),
        dither_amp=float(getattr(env_cfg, f"{prefix}dither", 0.04)),
        bias_step=float(getattr(env_cfg, f"{prefix}step", 0.01)),
        bias_max=float(getattr(env_cfg, f"{prefix}bias_max", 0.25)),
        objective=str(getattr(env_cfg, f"{prefix}objective", "specific_power")),
        shaft_eps=float(getattr(env_cfg, f"{prefix}shaft_eps", 10.0)),
        reset_decay=float(getattr(env_cfg, f"{prefix}reset_decay", 0.98)),
        objective_clip=float(getattr(env_cfg, f"{prefix}objective_clip", 10.0)),
        idle_enable=bool(getattr(env_cfg, f"{prefix}idle_enable", False)),
        idle_omega_pu=float(getattr(env_cfg, f"{prefix}idle_omega_min", 0.05)),
        idle_action=float(getattr(env_cfg, f"{prefix}idle_action", -1.0)),
        idle_blend=float(getattr(env_cfg, f"{prefix}idle_blend", 1.0)),
        idle_exit_boost_steps=int(getattr(env_cfg, f"{prefix}idle_exit_boost", 0)),
        idle_exit_action=float(getattr(env_cfg, f"{prefix}idle_exit_action", 1.0)),
        idle_bias_decay=float(getattr(env_cfg, f"{prefix}idle_bias_decay", 0.95)),
    )
    sup = AiIdRefSupervisor(cfg, omega_nominal=float(omega_nominal))
    sup.reset()
    return sup


def _load_agent(checkpoint: Path) -> PPOVoltageAgent:
    state = torch.load(checkpoint, map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)
    feature_keys = _resolve_feature_keys(None, state)
    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=1, device="cpu", hidden_sizes=hidden)
    adapted_state, _ = adapt_checkpoint_state_dict_for_model(
        state,
        agent.net.state_dict(),
        target_control_mode="ai_id_ref",
    )
    agent.net.load_state_dict(adapted_state, strict=False)
    agent.set_action_std(1e-6)
    return agent


def _build_bundle(
    *,
    name: str,
    checkpoint: Path,
    env_cfg: object,
    prefix: str,
    enabled_attr: str,
    supervisor_prefix: str,
    id_ref_min: float,
    id_ref_max: float,
    omega_nominal: float,
) -> Air56PolicyBundle:
    return Air56PolicyBundle(
        name=name,
        agent=_load_agent(checkpoint),
        params=_load_id_ref_params(env_cfg, prefix, id_ref_min=id_ref_min, id_ref_max=id_ref_max),
        supervisor=_load_supervisor(env_cfg, enabled_attr, supervisor_prefix, omega_nominal=omega_nominal),
    )


def _should_switch_secondary(
    *,
    load_est_nm: float,
    prev_load_est_nm: float,
    speed_err: float,
    omega_ref: float,
    threshold_nm: float,
    positive_only: bool,
    speed_err_threshold_rel: float,
    speed_err_threshold_abs: float,
) -> bool:
    delta = float(load_est_nm) - float(prev_load_est_nm)
    if positive_only:
        if delta <= float(threshold_nm):
            return False
    else:
        if abs(delta) <= float(threshold_nm):
            return False
    gate = max(float(speed_err_threshold_abs), float(speed_err_threshold_rel) * max(abs(float(omega_ref)), 1e-6))
    if gate > 0.0 and float(speed_err) < gate:
        return False
    return True


def _clamp_rate(prev_value: float, target_value: float, max_delta: float) -> float:
    if float(max_delta) <= 0.0:
        return float(prev_value)
    delta = float(target_value) - float(prev_value)
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return float(prev_value + delta)


def _send_fallback_command(
    transport: Optional[BaseTransport],
    *,
    t_ms: int,
    id_ref_base: float,
    crc: bool,
    dry_run: bool = False,
) -> None:
    if transport is None:
        return
    if bool(dry_run):
        print("[air56_unoq_bridge] dry-run: fallback send skipped", flush=True)
        return
    try:
        cmd = Command(t_ms=int(t_ms), enable_ai=0, id_ref=float(id_ref_base), crc=0)
        transport.send(cmd.pack_with_crc() if bool(crc) else cmd.pack())
        print(
            f"[air56_unoq_bridge] fallback sent: enable_ai=0 id_ref={float(id_ref_base):.3f}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - best effort during shutdown
        print(f"[air56_unoq_bridge] fallback send failed: {exc}", flush=True)


def _build_startup_self_check_report(
    *,
    config_path: Path,
    primary_checkpoint: Path,
    secondary_checkpoint: Optional[Path],
    transport_name: str,
    crc: bool,
    disable_on_fault: bool,
    disable_on_guard: bool,
    id_min: float,
    id_max: float,
    fallback_id_ref: float,
) -> dict[str, object]:
    return {
        "config_exists": bool(config_path.is_file()),
        "config_path": str(config_path),
        "primary_checkpoint_exists": bool(primary_checkpoint.is_file()),
        "primary_checkpoint": str(primary_checkpoint),
        "secondary_checkpoint_required": secondary_checkpoint is not None,
        "secondary_checkpoint_exists": True if secondary_checkpoint is None else bool(secondary_checkpoint.is_file()),
        "secondary_checkpoint": "" if secondary_checkpoint is None else str(secondary_checkpoint),
        "transport_ready": True,
        "transport": str(transport_name),
        "crc_enabled": bool(crc),
        "disable_on_fault": bool(disable_on_fault),
        "disable_on_guard": bool(disable_on_guard),
        "id_min": float(id_min),
        "id_max": float(id_max),
        "fallback_id_ref": float(fallback_id_ref),
        "fallback_inside_limits": bool(float(id_min) <= float(fallback_id_ref) <= float(id_max)),
    }


def _process_telemetry_step(
    *,
    telem: Telemetry,
    primary: Air56PolicyBundle,
    secondary: Optional[Air56PolicyBundle],
    state: Air56BridgeRuntimeState,
    omega_nominal: float,
    i_base: float,
    pole_pairs: int,
    rr: float,
    lr_total: float,
    load_est_gain: float,
    load_base_nm: float,
    load_delta_threshold: float,
    positive_only: bool,
    latch_steps: int,
    speed_err_threshold_rel: float,
    speed_err_threshold_abs: float,
    fault_mask: int,
    disable_on_fault: bool,
    disable_on_guard: bool,
    cmd_rate_limit_a_per_s: float,
    id_min: float,
    id_max: float,
) -> Air56BridgeStep:
    t_ms = int(telem.t_ms)
    state.last_telem_t_ms = t_ms
    if state.prev_t_ms is None:
        dt_s = 0.01
    else:
        dt_s = max(0.001, min(0.1, (t_ms - state.prev_t_ms) / 1000.0))
    state.prev_t_ms = t_ms

    omega_ref = float(telem.omega_ref)
    omega_meas = float(telem.omega_meas)
    speed_err = abs(omega_ref - omega_meas)
    load_est_nm = _estimate_load_nm_from_iq(telem.i_q, load_est_gain)

    active = primary
    if secondary is not None:
        trigger = _should_switch_secondary(
            load_est_nm=load_est_nm,
            prev_load_est_nm=state.prev_load_est_nm,
            speed_err=speed_err,
            omega_ref=omega_ref,
            threshold_nm=load_delta_threshold,
            positive_only=positive_only,
            speed_err_threshold_rel=speed_err_threshold_rel,
            speed_err_threshold_abs=speed_err_threshold_abs,
        )
        if trigger:
            state.secondary_left = max(int(latch_steps), 1)
        if state.secondary_left > 0:
            active = secondary
            state.secondary_left -= 1
    state.prev_load_est_nm = load_est_nm

    obs = _build_obs(
        telem=telem,
        omega_base=omega_nominal,
        i_base=i_base,
        pole_pairs=int(pole_pairs),
        rr=rr,
        lr_total=lr_total,
        load_torque_nm=load_est_nm,
        load_base_nm=load_base_nm,
    )
    action = _infer_action_scalar(active.agent, obs)
    gate_open = False
    if active.supervisor is not None:
        action, gate_open = active.supervisor.adjust_action(
            action,
            omega_ref=omega_ref,
            omega=omega_meas,
        )
    id_ref_cmd, gate_scale, gate_tol = _action_to_id_ref(
        action=action,
        prev_id_ref=state.last_id_ref,
        omega_ref=omega_ref,
        omega_meas=omega_meas,
        params=active.params,
    )

    enable_ai = 1
    hard_guard = gate_tol > 0.0 and speed_err > gate_tol
    if _status_fault(int(telem.status), int(fault_mask)) and bool(disable_on_fault):
        enable_ai = 0
        id_ref_cmd = float(active.params.id_ref_base)
    elif hard_guard and bool(disable_on_guard):
        enable_ai = 0
        id_ref_cmd = float(active.params.id_ref_base)

    rate_limit = max(0.0, float(cmd_rate_limit_a_per_s)) * dt_s
    if rate_limit > 0.0:
        id_ref_cmd = _clamp_rate(state.last_id_ref, id_ref_cmd, rate_limit)
    state.last_id_ref = float(max(float(id_min), min(float(id_max), id_ref_cmd)))

    if active.supervisor is not None:
        p_in_pos = max(0.0, float(telem.p_in))
        p_shaft_pos = max(0.0, abs(omega_meas) * load_est_nm)
        active.supervisor.update(
            p_in_pos=p_in_pos,
            p_shaft_pos=p_shaft_pos,
            gate_open=gate_open,
        )

    cmd = Command(t_ms=t_ms, enable_ai=enable_ai, id_ref=state.last_id_ref, crc=0)
    return Air56BridgeStep(
        command=cmd,
        t_ms=t_ms,
        active_name=active.name,
        omega_meas=omega_meas,
        omega_ref=omega_ref,
        iq_amp=float(telem.i_q),
        load_est_nm=load_est_nm,
        id_ref=state.last_id_ref,
        gate_scale=gate_scale,
        enable_ai=enable_ai,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AIR56 UNO Q Linux bridge (serial/UDP).")
    parser.add_argument("--config", default="config/env_research_air56_025kw.py")
    parser.add_argument("--transport", choices=["serial", "udp"], default="serial")
    parser.add_argument("--serial-port", default="/dev/ttyHS0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--serial-timeout", type=float, default=0.05)
    parser.add_argument("--listen", default="0.0.0.0:9000")
    parser.add_argument("--send", default="127.0.0.1:9001")
    parser.add_argument("--mode", choices=["primary", "hybrid"], default="hybrid")
    parser.add_argument("--fault-mask", type=int, default=0)
    parser.add_argument("--disable-on-fault", action="store_true")
    parser.add_argument("--disable-on-guard", action="store_true")
    parser.add_argument("--crc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cmd-rate-limit-a-per-s", type=float, default=12.0)
    parser.add_argument("--id-min", type=float, default=1.10)
    parser.add_argument("--id-max", type=float, default=1.70)
    parser.add_argument("--load-est-gain", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    _validate_runtime_args(args)

    config_path = _resolve_existing_file(str(args.config), "--config")
    env_cfg = make_env_from_config(str(config_path)).env_config
    motor = env_cfg.motor
    foc = env_cfg.foc
    _validate_id_ref_window(
        id_ref_base=float(getattr(foc, "id_ref", 0.0) or 0.0),
        id_min=float(args.id_min),
        id_max=float(args.id_max),
    )
    omega_nominal = float(2.0 * math.pi * float(env_cfg.scalar_vf.f_max) / max(int(motor.p), 1))
    i_base = float(max(getattr(motor, "I_n", 1.0), 1e-6))
    rr = float(getattr(motor, "Rr", 0.0))
    lr_total = float(getattr(motor, "Lr_sigma", 0.0) + getattr(motor, "Lm", 0.0))
    torque_nom = float(250.0) / max(2.0 * math.pi * 1380.0 / 60.0, 1e-6)
    load_est_gain = float(args.load_est_gain) if args.load_est_gain is not None else float(torque_nom / i_base)
    load_base_nm = max(float(getattr(env_cfg.sim, "load_torque", 0.0)), 0.2)

    primary_ckpt = _resolve_existing_file(
        str(getattr(env_cfg, "ai_eval_checkpoint_path")),
        "primary checkpoint",
    )
    primary = _build_bundle(
        name="primary",
        checkpoint=primary_ckpt,
        env_cfg=env_cfg,
        prefix="ai_eval_",
        enabled_attr="ai_eval_supervisor_enabled",
        supervisor_prefix="ai_eval_sup_",
        id_ref_min=float(args.id_min),
        id_ref_max=float(args.id_max),
        omega_nominal=omega_nominal,
    )

    secondary: Optional[Air56PolicyBundle] = None
    secondary_ckpt: Optional[Path] = None
    hybrid_enabled = bool(args.mode == "hybrid" and getattr(env_cfg, "ai_eval_hybrid_enabled", False))
    if hybrid_enabled:
        secondary_ckpt_text = str(getattr(env_cfg, "ai_eval_hybrid_secondary_checkpoint_path", "")).strip()
        if secondary_ckpt_text:
            secondary_ckpt = _resolve_existing_file(secondary_ckpt_text, "secondary checkpoint")
            secondary = _build_bundle(
                name="secondary",
                checkpoint=secondary_ckpt,
                env_cfg=env_cfg,
                prefix="ai_eval_hybrid_secondary_",
                enabled_attr="ai_eval_hybrid_secondary_supervisor_enabled",
                supervisor_prefix="ai_eval_hybrid_secondary_sup_",
                id_ref_min=float(args.id_min),
                id_ref_max=float(args.id_max),
                omega_nominal=omega_nominal,
            )

    load_delta_threshold = float(getattr(env_cfg, "ai_eval_hybrid_load_delta_threshold", 0.05))
    positive_only = bool(getattr(env_cfg, "ai_eval_hybrid_positive_only", True))
    latch_steps = int(getattr(env_cfg, "ai_eval_hybrid_latch_steps", 0))
    speed_err_threshold_rel = float(getattr(env_cfg, "ai_eval_hybrid_speed_err_threshold_rel", 0.0))
    speed_err_threshold_abs = float(getattr(env_cfg, "ai_eval_hybrid_speed_err_threshold_abs", 0.0))

    if args.transport == "udp":
        transport: Optional[BaseTransport] = UdpTransport(args.listen, args.send)
    else:
        transport = SerialTransport(args.serial_port, args.baud, args.serial_timeout)

    fallback_id_ref = float(primary.params.id_ref_base)
    startup_self_check = _build_startup_self_check_report(
        config_path=config_path,
        primary_checkpoint=primary_ckpt,
        secondary_checkpoint=secondary_ckpt,
        transport_name=str(args.transport),
        crc=bool(args.crc),
        disable_on_fault=bool(args.disable_on_fault),
        disable_on_guard=bool(args.disable_on_guard),
        id_min=float(args.id_min),
        id_max=float(args.id_max),
        fallback_id_ref=fallback_id_ref,
    )
    print(f"[air56_unoq_bridge] config={config_path}", flush=True)
    print(f"[air56_unoq_bridge] primary={primary_ckpt}", flush=True)
    if secondary is not None:
        print(f"[air56_unoq_bridge] secondary={getattr(env_cfg, 'ai_eval_hybrid_secondary_checkpoint_path')}", flush=True)
    print(
        f"[air56_unoq_bridge] transport={args.transport} mode={args.mode} crc={bool(args.crc)}",
        flush=True,
    )
    print(
        "[air56_unoq_bridge] startup_self_check={}".format(
            json.dumps(startup_self_check, sort_keys=True, ensure_ascii=False)
        ),
        flush=True,
    )

    state = Air56BridgeRuntimeState(last_id_ref=float(getattr(foc, "id_ref", 0.0) or 0.0))
    packets = 0

    try:
        while True:
            payload = transport.recv(Telemetry._struct.size)
            telem = Telemetry.unpack(payload)
            packets += 1

            step = _process_telemetry_step(
                telem=telem,
                primary=primary,
                secondary=secondary,
                state=state,
                omega_nominal=omega_nominal,
                i_base=i_base,
                pole_pairs=int(motor.p),
                rr=rr,
                lr_total=lr_total,
                load_est_gain=load_est_gain,
                load_base_nm=load_base_nm,
                load_delta_threshold=load_delta_threshold,
                positive_only=positive_only,
                latch_steps=latch_steps,
                speed_err_threshold_rel=speed_err_threshold_rel,
                speed_err_threshold_abs=speed_err_threshold_abs,
                fault_mask=int(args.fault_mask),
                disable_on_fault=bool(args.disable_on_fault),
                disable_on_guard=bool(args.disable_on_guard),
                cmd_rate_limit_a_per_s=float(args.cmd_rate_limit_a_per_s),
                id_min=float(args.id_min),
                id_max=float(args.id_max),
            )
            if not args.dry_run:
                transport.send(step.command.pack_with_crc() if args.crc else step.command.pack())

            if args.log_every > 0 and packets % int(args.log_every) == 0:
                print(
                    "[air56_unoq_bridge] pkt={} mode={} omega={:.2f}/{:.2f} iq={:.3f} load={:.3f} id_ref={:.3f} gate={:.3f} enable={}".format(
                        packets,
                        step.active_name,
                        step.omega_meas,
                        step.omega_ref,
                        step.iq_amp,
                        step.load_est_nm,
                        step.id_ref,
                        step.gate_scale,
                        step.enable_ai,
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("[air56_unoq_bridge] stopped", flush=True)
        _send_fallback_command(
            transport,
            t_ms=state.last_telem_t_ms,
            id_ref_base=fallback_id_ref,
            crc=bool(args.crc),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[air56_unoq_bridge] fatal: {exc}", flush=True)
        _send_fallback_command(
            transport,
            t_ms=state.last_telem_t_ms,
            id_ref_base=fallback_id_ref,
            crc=bool(args.crc),
            dry_run=bool(args.dry_run),
        )
        raise
    finally:
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    main()
