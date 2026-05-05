from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / ".tmp_pytest" / "coverage_air56_unoq_gate.json"

TESTS = [
    "tests/test_uno_q_protocol.py",
    "tests/test_air56_unoq_stage0_loopback.py",
    "tests/test_air56_unoq_bridge.py",
    "tests/test_air56_unoq_tooling.py",
    "tests/test_air56_unoq_deploy_package.py",
]

COV_MODULES = [
    "tools.uno_q_protocol",
    "tools.air56_unoq_stage0_loopback",
    "tools.air56_unoq_bridge",
    "tools.check_air56_unoq_firmware_static",
    "tools.run_air56_unoq_deploy_smoke",
]

MIN_TOTAL = 65.0
MIN_BY_FILE = {
    "tools/uno_q_protocol.py": 95.0,
    "tools/air56_unoq_stage0_loopback.py": 95.0,
    "tools/check_air56_unoq_firmware_static.py": 95.0,
    "tools/run_air56_unoq_deploy_smoke.py": 95.0,
    # The bridge runtime owns serial/UDP/torch infinite-loop integration; those
    # paths are covered by deploy smoke and hardware bring-up rather than unit
    # coverage. Keep a floor on helper coverage without pretending it is 100%.
    "tools/air56_unoq_bridge.py": 45.0,
}


@dataclass(frozen=True)
class CoverageGateResult:
    name: str
    actual: float
    required: float

    @property
    def passed(self) -> bool:
        return self.actual + 1e-9 >= self.required


def _norm_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _run_pytest(out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *TESTS,
    ]
    for module in COV_MODULES:
        cmd.append(f"--cov={module}")
    cmd.extend(
        [
            "--cov-report=term",
            f"--cov-report=json:{out_json}",
        ]
    )
    subprocess.run(cmd, check=True, cwd=ROOT)


def _evaluate(payload: dict) -> list[CoverageGateResult]:
    results = [
        CoverageGateResult(
            name="TOTAL",
            actual=float(payload["totals"]["percent_covered"]),
            required=MIN_TOTAL,
        )
    ]
    files = {_norm_path(path): data for path, data in dict(payload.get("files", {})).items()}
    for path, required in MIN_BY_FILE.items():
        data = files.get(path)
        actual = 0.0 if data is None else float(data["summary"]["percent_covered"])
        results.append(CoverageGateResult(name=path, actual=actual, required=float(required)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="AIR56 UNO Q production-critical coverage gate.")
    parser.add_argument("--coverage-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--reuse-json", action="store_true", help="Evaluate an existing coverage JSON without rerunning tests.")
    args = parser.parse_args()

    out_json = Path(str(args.coverage_json)).resolve()
    if not bool(args.reuse_json):
        _run_pytest(out_json)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    results = _evaluate(payload)
    report = {
        "passed": all(row.passed for row in results),
        "results": [
            {
                "name": row.name,
                "actual": round(row.actual, 2),
                "required": row.required,
                "passed": row.passed,
            }
            for row in results
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
