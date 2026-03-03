from __future__ import annotations

from pathlib import Path


def test_verify_wrapper_scripts_present() -> None:
    ps1 = Path("scripts/verify_ieee_submission_candidate.ps1").resolve()
    sh = Path("scripts/verify_ieee_submission_candidate.sh").resolve()
    assert ps1.exists()
    assert sh.exists()

    ps1_text = ps1.read_text(encoding="utf-8")
    sh_text = sh.read_text(encoding="utf-8")

    for token in (
        "tools/verify_ieee_submission_candidate.py",
        "--step28-dir",
        "--ieee-root",
        "--guardrails-policy",
        "--manuscript",
        "--strict",
    ):
        assert token in ps1_text
        assert token in sh_text
