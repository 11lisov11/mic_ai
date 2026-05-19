from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FW_DIR = ROOT / "arduino" / "air56_unoq_ready" / "firmware" / "air56_unoq_example"


ARDUINO_STUB = r"""
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

class MockSerialPort {
 public:
  void begin(unsigned long) {}
  int available() { return 0; }
  std::size_t readBytes(char *, std::size_t) { return 0; }
  std::size_t write(const uint8_t *, std::size_t len) { return len; }
  void println(const char *) {}
};

extern MockSerialPort Serial;
extern MockSerialPort Serial1;

inline uint32_t millis() {
  return 0u;
}

template <typename T>
inline T max(T lhs, T rhs) {
  return lhs > rhs ? lhs : rhs;
}
"""


WRAPPER = r"""
#include "Arduino.h"

MockSerialPort Serial;
MockSerialPort Serial1;

#if !defined(AIR56_UNOQ_USE_MOCK_HW)
static float g_production_port_id_ref_amp = 1.35f;

extern "C" float air56_foc_get_omega_meas_rad_s(void) {
  return 144.5f;
}

extern "C" float air56_foc_get_omega_ref_rad_s(void) {
  return 144.5f;
}

extern "C" float air56_foc_get_id_amp(void) {
  return g_production_port_id_ref_amp;
}

extern "C" float air56_foc_get_iq_amp(void) {
  return 0.4f;
}

extern "C" float air56_foc_get_vdc_volt(void) {
  return 24.0f;
}

extern "C" float air56_foc_get_irms_amp(void) {
  return 1.45f;
}

extern "C" float air56_foc_get_pin_watt(void) {
  return 42.0f;
}

extern "C" uint16_t air56_foc_get_status_bits(void) {
  return 0u;
}

extern "C" void air56_foc_set_id_ref_amp(float id_ref_amp) {
  g_production_port_id_ref_amp = id_ref_amp;
}
#endif

#include "air56_unoq_example.ino"

int main() {
  setup();
  loop();
  return 0;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Host static compile smoke for AIR56 UNO Q firmware.")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument(
        "--mode",
        choices=["mock", "production-port"],
        default="mock",
        help="mock uses AIR56_UNOQ_USE_MOCK_HW; production-port links through supplied air56_foc_* shim.",
    )
    args = parser.parse_args()

    compiler = shutil.which(str(args.compiler))
    if compiler is None:
        raise SystemExit(f"compiler not found: {args.compiler}")

    with tempfile.TemporaryDirectory(prefix="air56_unoq_static_") as tmp_text:
        tmp = Path(tmp_text)
        (tmp / "Arduino.h").write_text(ARDUINO_STUB, encoding="utf-8")
        wrapper = tmp / "air56_unoq_static_compile.cpp"
        wrapper.write_text(WRAPPER, encoding="utf-8")
        out_exe = tmp / f"air56_unoq_static_{args.mode}"
        build_flags = ["-DAIR56_UNOQ_USE_MOCK_HW=1"] if args.mode == "mock" else ["-DAIR56_UNOQ_PRODUCTION_PORT=1"]
        cmd = [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *build_flags,
            "-I",
            str(tmp),
            "-I",
            str(FW_DIR),
            str(wrapper),
            "-o",
            str(out_exe),
        ]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
