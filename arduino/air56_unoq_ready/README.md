# AIR56 UNO Q Deploy Package

This package is the AIR56 hardware-productization path for UNO Q split control.

Important: in this repository `UNO Q` means the split board architecture:

- `Qualcomm Dragonwing QRB2210` on Linux for the AI decision layer
- `STM32U585` for hard realtime FOC, safety, rate limiting, and fallback

This package is not a classic AVR Arduino Uno sketch.

## Current Readiness

Research/release readiness is complete for the 3-motor project. Hardware deployment is split into two states:

- Ready in repo: protocol, Linux bridge, safety gates, id_ref supervisor path, startup checks, mock firmware compile target, and bring-up documentation.
- Must be supplied per board: the real STM32U585 FOC/inverter hardware adapter that maps telemetry and `id_ref` commands to the actual current loop.

The firmware no longer contains fake production sensor readings inside `air56_unoq_example.ino`. Production builds include `air56_unoq_hw_port.h`, which declares the required `air56_foc_*` symbols. A separate mock adapter exists only for loopback and compile-smoke checks.

## Directory Layout

- `firmware/air56_unoq_example/`
  - `air56_unoq_example.ino`: STM32U585 protocol/safety loop
  - `air56_unoq_hw_port.h`: production FOC/inverter adapter contract
  - `air56_unoq_hw_port_template.cpp.example`: copy-and-fill template for the real STM32U585 project
  - `air56_unoq_hw_mock.h`: mock adapter for no-motor loopback only
  - `platformio.ini`: reproducible STM32U585 PlatformIO targets
- `linux/`
  - `run_air56_unoq_bridge.sh`: Linux launch wrapper
  - `run_air56_unoq_bridge.ps1`: Windows/bench launch wrapper
  - `air56_unoq_bridge.service`: env-based systemd unit
  - `air56_unoq_bridge.env.example`: `/etc/default` template
- `../../tools/air56_unoq_bridge.py`: QRB2210 Linux AI bridge
- `../../tools/air56_unoq_stage0_loopback.py`: protocol self-test for Stage 0
- `../../tools/run_air56_unoq_deploy_smoke.py`: one-command repo-side deploy smoke runner
- `../../docs/air56_unoq_bringup.md`: staged hardware bring-up protocol

## STM32U585 Firmware Build

Mock/loopback compile-smoke target:

```bash
pio run -d arduino/air56_unoq_ready/firmware/air56_unoq_example -e air56_unoq_stm32u585_mock
```

If PlatformIO is not installed on the workstation, run the host static compile smoke:

```bash
python tools/check_air56_unoq_firmware_static.py
```

Run the full repo-side deploy smoke:

```bash
python tools/run_air56_unoq_deploy_smoke.py
```

Run the production-critical coverage gate:

```bash
python tools/check_air56_unoq_coverage_gate.py
```

Current gate:

- total AIR56 deploy subset: `>=75%`
- protocol, Stage 0 loopback, firmware static compile, deploy smoke runner: `>=95%`
- Linux bridge helper/runtime module floor: `>=75%`

Production port target:

```bash
pio run -d arduino/air56_unoq_ready/firmware/air56_unoq_example -e air56_unoq_stm32u585_port
```

The production target intentionally does not define `AIR56_UNOQ_USE_MOCK_HW`. It must be linked with board code implementing:

- `air56_foc_get_omega_meas_rad_s()`
- `air56_foc_get_omega_ref_rad_s()`
- `air56_foc_get_id_amp()`
- `air56_foc_get_iq_amp()`
- `air56_foc_get_vdc_volt()`
- `air56_foc_get_irms_amp()`
- `air56_foc_get_pin_watt()`
- `air56_foc_get_status_bits()`
- `air56_foc_set_id_ref_amp(float id_ref_amp)`

Use `air56_unoq_hw_port_template.cpp.example` as the starting point for that board code.

Do not connect a motor with `AIR56_UNOQ_USE_MOCK_HW` enabled.

## Linux Bridge

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run manually:

```bash
MIC_THEORY_ROOT=/opt/mic_theory \
SERIAL_PORT=/dev/ttyHS0 \
BAUD=921600 \
CONFIG_PATH=/opt/mic_theory/config/env_research_air56_025kw.py \
MODE=hybrid \
./arduino/air56_unoq_ready/linux/run_air56_unoq_bridge.sh
```

Install systemd service:

```bash
sudo cp arduino/air56_unoq_ready/linux/air56_unoq_bridge.env.example /etc/default/air56_unoq_bridge
sudo editor /etc/default/air56_unoq_bridge
sudo cp arduino/air56_unoq_ready/linux/air56_unoq_bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now air56_unoq_bridge
```

The bridge performs startup checks before entering control:

- config file exists
- primary checkpoint exists
- secondary hybrid checkpoint exists when enabled
- serial transport can open the configured port

On shutdown or fatal runtime error it sends one best-effort fallback command: `enable_ai=0`, `id_ref=base`.

## Guardrails

- MCU telemetry/command period: `10 ms`
- command timeout fallback: `100 ms`
- speed telemetry scale: `rad/s * 128` in int16, chosen so AIR56 nominal speed fits without wraparound
- STM32U585 always owns FOC, current limits, faults, and fallback
- QRB2210/Linux only adjusts `id_ref`; it does not own the fast current loop
- frequent fallback/gate events mean the first suspects are sensor scaling, speed feedback, UART stability, or FOC current-loop tuning

## AIR56 Verified Research Gain

Strict verified release: `paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release`.

AIR56 release metrics:

- `avg_power_saving_pct_mean = +1.024%`
- `avg_power_saving_pct_min = +0.901%`
- `avg_eta_gain_pct_mean = +0.123%`
- `avg_eta_gain_pct_min = +0.104%`
- `start_stop_power_saving_pct_mean = +1.835%`
- `start_stop_power_saving_pct_min = +1.528%`

These are simulation/release results. Physical AIR56 acceptance must be proven with the staged bring-up protocol before calling the board deployment complete.
