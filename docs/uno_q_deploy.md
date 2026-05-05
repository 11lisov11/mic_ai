# UNO Q Deployment Plan (Linux AI + STM PWM)

UNO Q hardware note: the board couples a Linux-capable Qualcomm Dragonwing QRB2210 MPU
with an STM32U585 MCU for real-time control, so the Linux/STM split below maps directly
to the target hardware.

Ready AIR56 package:

- [arduino/air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready)
- dedicated bridge: [air56_unoq_bridge.py](C:/mic_theory/tools/air56_unoq_bridge.py)
- staged bring-up protocol: [air56_unoq_bringup.md](C:/mic_theory/docs/air56_unoq_bringup.md)

Readiness note: the repository contains the split deploy package and the
adapter contract, not a board-specific FOC/HAL implementation. A motor-connected
build must implement the `air56_foc_*` functions declared in
`arduino/air56_unoq_ready/firmware/air56_unoq_example/air56_unoq_hw_port.h`.

## 1) Split of responsibilities
- Linux (QRB2210): runs AI policy at low rate (50-200 Hz). Output is id_ref (flux reference) or delta_id_ref.
- STM32U585: runs FOC speed/current loops at high rate (PWM 10-20 kHz, current loop ~2-5 kHz).
- Safety: STM owns hard limits and fallbacks; Linux is advisory.

## 2) Data flow
STM -> Linux telemetry (every 5-20 ms):
- t_ms: timestamp
- omega_meas (rad/s)
- omega_ref (rad/s)
- i_d, i_q (A)
- v_dc (V)
- i_rms (A)
- p_in_total (W) [optional]
- status flags (fault, saturation, overcurrent)

Linux -> STM command (every 5-20 ms):
- enable_ai (0/1)
- id_ref_cmd (A) or delta_id_norm (-1..1)
- optional: iq_ref_scale (for torque shaping)

## 3) Message format (binary, fixed size)
Use fixed-point to avoid parsing overhead. Example:

STM->Linux (struct, little-endian):
- u32 t_ms
- i16 omega_meas_q10 (rad/s * 1024)
- i16 omega_ref_q10
- i16 id_q10 (A * 1024)
- i16 iq_q10
- u16 vdc_q8 (V * 256)
- i16 i_rms_q10
- i16 p_in_q2 (W * 4)  [optional]
- u16 status

Linux->STM:
- u32 t_ms
- u8  enable_ai
- i16 id_ref_q10 (A * 1024) OR i16 delta_id_q14 (-1..1 * 16384)
- u16 crc

CRC: CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) computed over the full
command packet with the CRC field set to 0. Enable in the bridge with `--crc`
and mirror the same algorithm on STM.

## 4) Safety gates (STM side)
- If enable_ai == 0 or comm timeout > 100 ms -> fallback to nominal id_ref.
- Clamp id_ref in [id_ref_min, id_ref_max].
- Slew rate limit: |d(id_ref)/dt| <= rate_limit.
- Speed error gate: if |omega_ref - omega_meas| > speed_tol -> freeze id_ref (or revert to nominal).
- Hard overcurrent -> immediate fallback + fault flag.

## 4.1) STM32U585 hardware adapter
The AIR56 package separates production hardware binding from the protocol loop:

- `air56_unoq_hw_port.h`: production adapter contract.
- `air56_unoq_hw_mock.h`: opt-in mock adapter for no-motor loopback only.
- `platformio.ini`: STM32U585 mock compile target and production-port target.

Production builds must not define `AIR56_UNOQ_USE_MOCK_HW`.

## 5) Policy inputs mapping (ai_id_ref)
FEATURE_KEYS in training:
- omega_norm, omega_ref_norm, err_norm, id_norm, iq_norm, slip_norm, load_torque_norm

Ensure telemetry provides these or allow Linux to compute:
- slip_norm = (omega_syn - omega_meas) / slip_base
- load_torque_norm from estimator or fixed value

## 6) Export options
- Linux runtime: keep PyTorch model for flexibility.
- Optional MCU fallback: distill to TinyStudent and export C header.
  - See mic_ai/ai/distill_voltage.py (export_tiny_to_c_header).
  - For id_ref policy, use action_dim=1 and same export path.
- LUT-only fallback (fastest on Arduino):
  - Generate LUT JSON with `mic_ai/tools/id_ref_lut.py`.
  - Export C headers:
    - `python -m mic_ai.tools.export_id_ref_lut_c --lut outputs/id_ref_lut_motor1/id_ref_lut.json --out arduino/id_ref_lut_motor1.h --symbol-prefix unoq_motor1`
    - `python -m mic_ai.tools.export_id_ref_lut_c --lut outputs/id_ref_lut_motor2/id_ref_lut.json --out arduino/id_ref_lut_motor2.h --symbol-prefix unoq_motor2`
  - Include `arduino/uno_q_protocol.h` + `arduino/uno_q_control.h` in the firmware and use
    `*_id_ref_query_q()` to get the next id_ref in fixed-point.

## 7) Bring-up checklist
- Verify sensors: current offsets, speed scaling, Vdc scaling.
- Tune FOC PI loops first (STM only).
- Enable AI with speed_tol gate + tight id_ref limits.
- Compare FOC vs AI using scenario_compare outputs.
- Gradually relax limits and increase AI update rate.

## 8) Linux bridge example
- See `tools/uno_q_ai_bridge.py` for a minimal UDP bridge that parses telemetry,
  runs an id_ref policy, and sends commands back.
- Safety flags: `--disable-on-guard` (speed error gate -> enable_ai=0),
  `--disable-on-fault` with optional `--fault-mask` for status bits,
  and `--crc` to enable CRC16 on commands.

## 9) Arduino/Q firmware building blocks
- Protocol: `arduino/uno_q_protocol.h` (packed structs + CRC16).
- Safety gates and slew limit: `arduino/uno_q_control.h`.
- LUT headers (generated): `arduino/id_ref_lut_motor1.h`, `arduino/id_ref_lut_motor2.h`.
