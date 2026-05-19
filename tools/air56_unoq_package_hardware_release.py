from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.air56_unoq_hardware_release_gate import build_release_gate_summary


SCHEMA = "mic_theory.air56_unoq.hardware_release_package.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _copy_file(*, role: str, source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "role": role,
        "source": str(source),
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def build_hardware_release_package(
    *,
    package_tag: str,
    out_dir: Path,
    binding_manifest: Path,
    hardware_report: Path,
    deploy_smoke_json: Path,
    coverage_json: Path,
    repo_root: Path,
    allow_not_ready: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "evidence"
    gate_summary = build_release_gate_summary(
        binding_manifest=binding_manifest,
        hardware_report=hardware_report,
        deploy_smoke_json=deploy_smoke_json,
        coverage_json=coverage_json,
        repo_root=repo_root,
    )
    release_ready = bool(gate_summary.get("release_ready"))

    copied = [
        _copy_file(role="hardware_binding_manifest", source=binding_manifest, destination=evidence_dir / "hardware_binding_manifest.json"),
        _copy_file(role="hardware_acceptance_report", source=hardware_report, destination=evidence_dir / "hardware_acceptance_report.json"),
        _copy_file(role="deploy_smoke_report", source=deploy_smoke_json, destination=evidence_dir / "deploy_smoke.json"),
        _copy_file(role="coverage_gate_json", source=coverage_json, destination=evidence_dir / "coverage_air56_unoq_gate.json"),
    ]

    gate_path = out_dir / "release_gate_summary.json"
    gate_path.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    copied.append(
        {
            "role": "release_gate_summary",
            "source": "generated",
            "path": str(gate_path),
            "bytes": gate_path.stat().st_size,
            "sha256": _sha256(gate_path),
        }
    )

    manifest = {
        "schema": SCHEMA,
        "package_tag": str(package_tag).strip(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_head": _git_head(repo_root),
        "release_ready": release_ready,
        "allow_not_ready": bool(allow_not_ready),
        "release_gate_summary": str(gate_path),
        "evidence": copied,
    }
    manifest_path = out_dir / "hardware_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "release_ready": release_ready,
        "package_dir": str(out_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "release_gate_summary": gate_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Package AIR56 UNO Q hardware release evidence with hashes and release gate summary.")
    parser.add_argument("--package-tag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--hardware-report", required=True)
    parser.add_argument("--deploy-smoke-json", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--allow-not-ready", action="store_true", help="Write the evidence package even when release_ready=false.")
    args = parser.parse_args()

    summary = build_hardware_release_package(
        package_tag=str(args.package_tag),
        out_dir=Path(str(args.out_dir)).resolve(),
        binding_manifest=Path(str(args.binding_manifest)).resolve(),
        hardware_report=Path(str(args.hardware_report)).resolve(),
        deploy_smoke_json=Path(str(args.deploy_smoke_json)).resolve(),
        coverage_json=Path(str(args.coverage_json)).resolve(),
        repo_root=Path(str(args.repo_root)).resolve(),
        allow_not_ready=bool(args.allow_not_ready),
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if bool(summary["release_ready"]) or bool(args.allow_not_ready):
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
