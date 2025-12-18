# Embedded Reference (C)

This folder contains small, dependency-free C reference implementations of the core control math used by this project:

- Clarke/Park transforms (`abc` ↔ `dq`)
- PI controller (with the same anti-windup semantics as `control/vector_foc.py`)
- FOC step (voltage commands `v_d/v_q`)

These files are meant to be portable to MCU firmware (e.g., STM32U585 on Arduino UNO Q) and can be compiled on a host
for regression tests.

