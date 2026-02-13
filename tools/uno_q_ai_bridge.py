from __future__ import annotations

"""Linux-side AI bridge for UNO Q (UDP-based example).

Receives telemetry packets, runs an optional id_ref policy, and sends commands back.
"""

import argparse
import math
import socket
from pathlib import Path
from typing import Dict, Tuple

from control.id_ref_lut import IdRefLut
from mic_ai.core.env import make_env_from_config
from tools.uno_q_protocol import Command, Telemetry


def _parse_host_port(text: str) -> Tuple[str, int]:
    host, port = text.rsplit(":", 1)
    return host.strip(), int(port)


def _infer_hidden_sizes(state: Dict) -> tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    try:
        return int(w0.shape[0]), int(w2.shape[0])
    except Exception:
        return None


def _status_fault(status: int, fault_mask: int) -> bool:
    status_val = int(status)
    mask = int(fault_mask)
    if mask == 0:
        return status_val != 0
    return (status_val & mask) != 0


def _apply_gates(
    speed_err: float,
    tol: float,
    status: int,
    id_ref_base: float,
    id_ref_cmd: float,
    disable_on_guard: bool,
    disable_on_fault: bool,
    fault_mask: int,
) -> Tuple[int, float, bool]:
    fault = _status_fault(status, fault_mask)
    guard = speed_err > tol
    enable_ai = 1
    if (fault and disable_on_fault) or (guard and disable_on_guard):
        enable_ai = 0
    if fault or guard:
        return enable_ai, float(id_ref_base), True
    return enable_ai, float(id_ref_cmd), False


def _action_to_float(action: object) -> float:
    if hasattr(action, "item") and not hasattr(action, "__len__"):
        try:
            return float(action.item())
        except Exception:
            pass
    try:
        return float(action[0])  # type: ignore[index]
    except Exception:
        return float(action)


def _build_obs(
    telem: Telemetry,
    omega_base: float,
    i_base: float,
    p: int,
    rr: float,
    lr: float,
    load_torque: float,
    load_base: float,
) -> Dict[str, float]:
    omega_ref = telem.omega_ref
    omega = telem.omega_meas
    err = omega_ref - omega

    slip = 0.0
    if lr > 1e-6:
        slip = (rr / lr) * (telem.i_q / max(abs(telem.i_d), 1e-6))
    omega_syn = p * omega + slip
    slip_norm = slip / max(abs(omega_ref), 1e-6)

    obs = {
        "omega_norm": omega / max(omega_base, 1e-6),
        "omega_ref_norm": omega_ref / max(omega_base, 1e-6),
        "err_norm": err / max(omega_base, 1e-6),
        "id_norm": telem.i_d / max(i_base, 1e-6),
        "iq_norm": telem.i_q / max(i_base, 1e-6),
        "slip_norm": slip_norm,
        "load_torque_norm": load_torque / max(load_base, 1e-6) if load_base else 0.0,
    }
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="UNO Q AI bridge (UDP).")
    parser.add_argument("--config", required=True, help="Env config path (.py)")
    parser.add_argument("--listen", default="0.0.0.0:9000", help="UDP listen host:port")
    parser.add_argument("--send", default="127.0.0.1:9001", help="UDP send host:port")
    parser.add_argument("--checkpoint", default=None, help="Path to best_actor.pth")
    parser.add_argument("--lut", default=None, help="Path to id_ref_lut.json (MCU-style lookup)")
    parser.add_argument("--relative", action="store_true", help="Interpret action as delta around id_ref_base")
    parser.add_argument("--delta-id-max", type=float, default=None)
    parser.add_argument("--id-min", type=float, default=0.0)
    parser.add_argument("--id-max", type=float, default=None)
    parser.add_argument("--omega-base", type=float, default=None)
    parser.add_argument("--load-torque", type=float, default=None)
    parser.add_argument("--speed-tol", type=float, default=None)
    parser.add_argument("--speed-tol-rel", type=float, default=None)
    parser.add_argument("--crc", action="store_true", help="Compute CRC16/CCITT on command payloads")
    parser.add_argument("--disable-on-guard", action="store_true", help="Disable AI when speed error exceeds tolerance")
    parser.add_argument("--disable-on-fault", action="store_true", help="Disable AI when telemetry status indicates fault")
    parser.add_argument("--fault-mask", type=int, default=0, help="Bitmask for fault bits (0 = any nonzero)")
    parser.add_argument("--dry-run", action="store_true", help="Do not send commands, only log")
    args = parser.parse_args()

    env_cfg = make_env_from_config(args.config).env_config
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    p = int(getattr(env_cfg.motor, "p", 2))
    rr = float(getattr(env_cfg.motor, "Rr", 0.0))
    lr = float(getattr(env_cfg.motor, "Lr_sigma", 0.0) + getattr(env_cfg.motor, "Lm", 0.0))

    omega_base = float(args.omega_base) if args.omega_base is not None else float(2.0 * math.pi * 10.0 / max(p, 1))
    load_torque = float(args.load_torque) if args.load_torque is not None else float(getattr(env_cfg.sim, "load_torque", 0.0))
    load_base = max(abs(load_torque), 1.0)

    id_ref_base = float(getattr(getattr(env_cfg, "foc", None), "id_ref", 0.0) or 0.0)
    current_limit = float(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base) or i_base)
    id_max = float(args.id_max) if args.id_max is not None else max(i_base * 1.5, id_ref_base * 1.2, current_limit)
    id_min = max(0.0, float(args.id_min))
    delta_id_max = float(args.delta_id_max) if args.delta_id_max is not None else float(getattr(env_cfg, "ai_delta_id_max", 0.3))

    speed_tol = float(args.speed_tol) if args.speed_tol is not None else float(getattr(env_cfg, "ai_id_speed_tol", 0.5))
    speed_tol_rel = args.speed_tol_rel
    if speed_tol_rel is None:
        speed_tol_rel = getattr(env_cfg, "ai_id_speed_tol_rel", None)

    policy = None
    feature_keys = None
    lut = None
    if args.lut:
        lut = IdRefLut.from_json(args.lut)
    if args.checkpoint:
        try:
            import torch
            from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
            from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Torch/agent unavailable: {exc}")

        state = torch.load(Path(args.checkpoint), map_location="cpu")
        hidden = _infer_hidden_sizes(state) or (128, 128)
        agent = PPOVoltageAgent(feature_keys=FEATURE_KEYS, action_dim=1, device="cpu", hidden_sizes=hidden)
        agent.net.load_state_dict(state, strict=False)
        agent.set_action_std(1e-6)
        policy = agent
        feature_keys = FEATURE_KEYS

    listen_host, listen_port = _parse_host_port(args.listen)
    send_host, send_port = _parse_host_port(args.send)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((listen_host, listen_port))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[bridge] listening on {listen_host}:{listen_port}, sending to {send_host}:{send_port}")

    last_id_ref = id_ref_base
    buf = bytearray(Telemetry._struct.size)
    view = memoryview(buf)
    while True:
        size, _addr = rx.recvfrom_into(view)
        if size != Telemetry._struct.size:
            continue
        telem = Telemetry.unpack(view)

        omega_ref = telem.omega_ref
        omega_meas = telem.omega_meas
        speed_err = abs(omega_ref - omega_meas)
        omega_scale = max(abs(omega_ref), 1e-6)
        tol = speed_tol
        if speed_tol_rel is not None:
            tol = max(tol, float(speed_tol_rel) * omega_scale)

        id_ref_cmd = id_ref_base
        if lut is not None:
            id_ref_cmd = float(lut.query(omega_ref, load_torque))
            action_val = 0.0
        elif policy is not None:
            obs = _build_obs(telem, omega_base, i_base, p, rr, lr, load_torque, load_base)
            action, _logp, _v = policy.act(obs)
            action_val = _action_to_float(action)
        else:
            action_val = 0.0
        if lut is None and policy is not None:
            if speed_err > tol:
                id_ref_cmd = id_ref_base
            else:
                if args.relative:
                    id_ref_cmd = id_ref_base + action_val * delta_id_max * max(1.0, abs(id_ref_base))
                else:
                    id_ref_cmd = id_min + 0.5 * (action_val + 1.0) * (id_max - id_min)
        id_ref_cmd = float(max(id_min, min(id_max, id_ref_cmd)))

        enable_ai, id_ref_cmd, gated = _apply_gates(
            speed_err=speed_err,
            tol=tol,
            status=int(telem.status),
            id_ref_base=id_ref_base,
            id_ref_cmd=id_ref_cmd,
            disable_on_guard=bool(args.disable_on_guard),
            disable_on_fault=bool(args.disable_on_fault),
            fault_mask=int(args.fault_mask),
        )

        if not gated:
            # simple rate limit
            max_delta = delta_id_max
            id_ref_cmd = float(last_id_ref + max(-max_delta, min(max_delta, id_ref_cmd - last_id_ref)))
        last_id_ref = id_ref_cmd

        cmd = Command(t_ms=telem.t_ms, enable_ai=int(enable_ai), id_ref=id_ref_cmd, crc=0)
        if not args.dry_run:
            payload = cmd.pack_with_crc() if args.crc else cmd.pack()
            tx.sendto(payload, (send_host, send_port))


if __name__ == "__main__":
    main()
