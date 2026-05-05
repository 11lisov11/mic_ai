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
#include "air56_unoq_example.ino"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Host static compile smoke for AIR56 UNO Q firmware.")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()

    compiler = shutil.which(str(args.compiler))
    if compiler is None:
        raise SystemExit(f"compiler not found: {args.compiler}")

    with tempfile.TemporaryDirectory(prefix="air56_unoq_static_") as tmp_text:
        tmp = Path(tmp_text)
        (tmp / "Arduino.h").write_text(ARDUINO_STUB, encoding="utf-8")
        wrapper = tmp / "air56_unoq_static_compile.cpp"
        wrapper.write_text(WRAPPER, encoding="utf-8")
        out_obj = tmp / "air56_unoq_static_compile.o"
        cmd = [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DAIR56_UNOQ_USE_MOCK_HW=1",
            "-I",
            str(tmp),
            "-I",
            str(FW_DIR),
            "-c",
            str(wrapper),
            "-o",
            str(out_obj),
        ]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
