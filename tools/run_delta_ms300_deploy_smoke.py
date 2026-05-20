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
    return SmokeStep(name=name, command=command, returncode=int(proc.returncode), elapsed_s=round(time.perf_counter() - start, 3))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Delta MS300 AIR56 repo-side deploy smoke checks.")
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
                "tools/delta_ms300_modbus.py",
                "tools/delta_ms300_modbus_bridge.py",
                "tools/run_delta_ms300_deploy_smoke.py",
            ],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "dry_run_self_check",
            [sys.executable, "tools/delta_ms300_modbus_bridge.py", "--dry-run", "self-check"],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "dry_run_read_once",
            [sys.executable, "tools/delta_ms300_modbus_bridge.py", "--dry-run", "read-once"],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "dry_run_stage0_read_only",
            [
                sys.executable,
                "tools/delta_ms300_modbus_bridge.py",
                "--dry-run",
                "stage0",
                "--probe-frequency-hz",
                "1.0",
            ],
            dry_run=bool(args.dry_run),
        ),
        _run_step(
            "targeted_pytest",
            [sys.executable, "-m", "pytest", "-q", "tests/test_delta_ms300_modbus.py"],
            dry_run=bool(args.dry_run),
        ),
    ]
    report = {"passed": all(step.passed for step in steps), "steps": [asdict(step) | {"passed": step.passed} for step in steps]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
