# AIR56 UNO Q Ready Package

Готовый deploy-пакет под `AIR56` для нашей `UNO Q`.

Важно: в этом репозитории `UNO Q` это не классическая `Arduino Uno` на AVR.
Подразумевается связка:

- `Qualcomm Dragonwing QRB2210` на Linux
- `STM32U585` для жесткого realtime-контроля

Именно под такую split-архитектуру и собран пакет:

- `STM32U585` держит FOC, safety, rate limit и fallback
- `QRB2210/Linux` считает нашу `MIC/AI` команду `id_ref`

Это самый практичный путь, если нужна низкая задержка и стабильность:

- быстрые контуры остаются на MCU
- high-level `id_ref` обновляется редко, без перегруза MCU
- при обрыве связи или fault система сама падает обратно в базовый `FOC`

## Что лежит в папке

- `firmware/air56_unoq_example/`
  - пример STM-прошивки под UART-протокол `UNO Q`
  - command timeout
  - CRC
  - safety gating
  - slew-rate limit
- `linux/`
  - готовые launch-скрипты Linux/Windows
  - запускают dedicated bridge [air56_unoq_bridge.py](C:/mic_theory/tools/air56_unoq_bridge.py)
  - systemd unit: [air56_unoq_bridge.service](C:/mic_theory/arduino/air56_unoq_ready/linux/air56_unoq_bridge.service)

## Что именно разворачивать

Рекомендуемый production path:

1. Прошивка STM из `firmware/air56_unoq_example/`
2. Linux-side bridge из `linux/run_air56_unoq_bridge.sh`

Это ближе всего к нашей финальной `AIR56` архитектуре из проекта.

## Почему не весь AI внутрь MCU

Потому что для `AIR56` финальный выигрыш получен не “голым LUT”, а более богатым runtime:

- primary actor
- secondary actor для load-step-like событий
- online gating / smoothing
- supervisor around `id_ref`

На `UNO Q` это надо оставлять на Linux-стороне, иначе на MCU получится либо тяжело, либо хуже по качеству.

## Что реально получено по AIR56 в финальном релизе

По strict verified release:

- `avg_power_saving_pct_mean = +1.024%`
- `avg_power_saving_pct_min = +0.901%`
- `avg_eta_gain_pct_mean = +0.123%`
- `avg_eta_gain_pct_min = +0.104%`
- `start_stop_power_saving_pct_mean = +1.835%`
- `start_stop_power_saving_pct_min = +1.528%`

Артефакты:

- [motor_air56_tuning_report.md](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_air56_tuning_report.md)
- [motor_tuning_acceptance_summary.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_tuning_acceptance_summary.json)

## Что надо подключить в STM-прошивке

В `air56_unoq_example.ino` специально оставлены integration hooks:

- `air56_read_omega_meas_rad_s()`
- `air56_read_omega_ref_rad_s()`
- `air56_read_id_amp()`
- `air56_read_iq_amp()`
- `air56_read_vdc_volt()`
- `air56_read_irms_amp()`
- `air56_read_pin_watt()`
- `air56_apply_id_ref_amp()`

Их надо привязать к вашему реальному FOC/inverter layer.

## Запуск Linux bridge

Linux:

```bash
cd /path/to/repo
./arduino/air56_unoq_ready/linux/run_air56_unoq_bridge.sh /dev/ttyHS0
```

Linux systemd:

```bash
sudo cp arduino/air56_unoq_ready/linux/air56_unoq_bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now air56_unoq_bridge
```

Windows:

```powershell
Set-Location C:\mic_theory
.\arduino\air56_unoq_ready\linux\run_air56_unoq_bridge.ps1 -SerialPort COM5
```

## Практические guardrails

- MCU telemetry/command period: `10 ms`
- command timeout fallback: `100 ms`
- MCU всегда держит собственный `FOC id_ref_base`
- Linux bridge разрешено только подправлять `id_ref`, не ломать быстрый контур
- если на железе увидишь частые fallback/gate events, сначала лечить:
  - scaling датчиков
  - speed feedback
  - UART stability
  - FOC current loop

## Если нужен режим “совсем без лагов”

Оставляй этот split-path и не пытайся таскать тяжелую логику в MCU.
На этой плате правильный realtime-path такой:

- `STM32U585` = hard realtime
- `QRB2210` = AI decision layer

Это и есть режим, который дает лучший баланс:

- качество выше базового FOC
- latency под контролем
- safety остается на MCU
