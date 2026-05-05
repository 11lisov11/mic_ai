from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SmokeStep:
    name: str
    command: list[str]
    returncode: int
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _run_step(name: str, command: list[str], *, dry_run: bool) -> SmokeStep:
    start = time.perf_counter()
    if dry_run:
        return SmokeStep(name=name, command=command, returncode=0, elapsed_s=0.0)
    proc = subprocess.run(command, cwd=ROOT)
    return SmokeStep(
        name=name,
        command=command,
        returncode=int(proc.returncode),
        elapsed_s=round(time.perf_counter() - start, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIR56 UNO Q deploy smoke checks.")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    steps = [
        _run_step(
            "python_compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                "tools/air56_unoq_bridge.py",
                "tools/air56_unoq_stage0_loopback.py",
                "tools/check_air56_unoq_firmware_static.py",
            ],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "stage0_loopback",
            [sys.executable, "tools/air56_unoq_stage0_loopback.py", "--packets", "16"],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "firmware_static_compile",
            [sys.executable, "tools/check_air56_unoq_firmware_static.py"],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "targeted_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_uno_q_protocol.py",
                "tests/test_uno_q_bridge.py",
                "tests/test_air56_unoq_bridge.py",
                "tests/test_air56_unoq_deploy_package.py",
                "tests/test_air56_unoq_stage0_loopback.py",
            ],
            dry_run=bool(args.dry_run),
        ),
    ]
    passed = all(step.passed for step in steps)
    report = {
        "passed": passed,
        "steps": [asdict(step) | {"passed": step.passed} for step in steps],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
