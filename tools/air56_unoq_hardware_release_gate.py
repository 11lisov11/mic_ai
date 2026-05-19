from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.air56_unoq_hardware_acceptance import _load_report, build_acceptance_summary
from tools.air56_unoq_validate_hw_binding import _load_manifest, build_binding_summary
from tools.check_air56_unoq_coverage_gate import _evaluate as evaluate_coverage_gate


@dataclass(frozen=True)
class ReleaseGateCheck:
    name: str
    passed: bool
    detail: str


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _failed_check(name: str, exc: Exception) -> ReleaseGateCheck:
    return ReleaseGateCheck(name=name, passed=False, detail=str(exc))


def _binding_check(path: Path, repo_root: Path) -> tuple[ReleaseGateCheck, dict[str, Any] | None]:
    try:
        summary = build_binding_summary(_load_manifest(path), repo_root=repo_root)
    except Exception as exc:
        return _failed_check("hardware_binding", exc), None
    return (
        ReleaseGateCheck(
            name="hardware_binding",
            passed=bool(summary.get("hardware_binding_ready")),
            detail="hardware_binding_ready must be true",
        ),
        summary,
    )


def _acceptance_check(path: Path) -> tuple[ReleaseGateCheck, dict[str, Any] | None]:
    try:
        summary = build_acceptance_summary(_load_report(path))
    except Exception as exc:
        return _failed_check("hardware_acceptance", exc), None
    return (
        ReleaseGateCheck(
            name="hardware_acceptance",
            passed=bool(summary.get("hardware_ready")),
            detail="hardware_ready must be true",
        ),
        summary,
    )


def _deploy_smoke_check(path: Path) -> tuple[ReleaseGateCheck, dict[str, Any] | None]:
    try:
        payload = _load_json_object(path, label="deploy smoke report")
    except Exception as exc:
        return _failed_check("deploy_smoke", exc), None
    return (
        ReleaseGateCheck(
            name="deploy_smoke",
            passed=bool(payload.get("passed")),
            detail="deploy smoke report passed must be true",
        ),
        payload,
    )


def _coverage_check(path: Path) -> tuple[ReleaseGateCheck, dict[str, Any] | None]:
    try:
        payload = _load_json_object(path, label="coverage JSON")
        rows = evaluate_coverage_gate(payload)
    except Exception as exc:
        return _failed_check("coverage_gate", exc), None
    report = {
        "passed": all(row.passed for row in rows),
        "results": [
            {
                "name": row.name,
                "actual": round(row.actual, 2),
                "required": row.required,
                "passed": row.passed,
            }
            for row in rows
        ],
    }
    return (
        ReleaseGateCheck(
            name="coverage_gate",
            passed=bool(report["passed"]),
            detail="production-critical coverage thresholds must pass",
        ),
        report,
    )


def build_release_gate_summary(
    *,
    binding_manifest: Path,
    hardware_report: Path,
    deploy_smoke_json: Path,
    coverage_json: Path,
    repo_root: Path,
) -> dict[str, Any]:
    binding_check, binding_summary = _binding_check(binding_manifest, repo_root)
    acceptance_check, acceptance_summary = _acceptance_check(hardware_report)
    smoke_check, smoke_summary = _deploy_smoke_check(deploy_smoke_json)
    coverage_check, coverage_summary = _coverage_check(coverage_json)
    checks = [binding_check, acceptance_check, smoke_check, coverage_check]
    return {
        "release_ready": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "evidence": {
            "binding_manifest": str(binding_manifest),
            "hardware_report": str(hardware_report),
            "deploy_smoke_json": str(deploy_smoke_json),
            "coverage_json": str(coverage_json),
        },
        "details": {
            "hardware_binding": binding_summary,
            "hardware_acceptance": acceptance_summary,
            "deploy_smoke": smoke_summary,
            "coverage_gate": coverage_summary,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate AIR56 UNO Q hardware release evidence into one release_ready gate.")
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--hardware-report", required=True)
    parser.add_argument("--deploy-smoke-json", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    summary = build_release_gate_summary(
        binding_manifest=Path(str(args.binding_manifest)).resolve(),
        hardware_report=Path(str(args.hardware_report)).resolve(),
        deploy_smoke_json=Path(str(args.deploy_smoke_json)).resolve(),
        coverage_json=Path(str(args.coverage_json)).resolve(),
        repo_root=Path(str(args.repo_root)).resolve(),
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(summary["release_ready"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
