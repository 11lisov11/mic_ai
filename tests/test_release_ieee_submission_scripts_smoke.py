from __future__ import annotations

from pathlib import Path


def test_release_wrapper_scripts_present() -> None:
    ps1 = Path("scripts/release_ieee_submission_candidate.ps1").resolve()
    sh = Path("scripts/release_ieee_submission_candidate.sh").resolve()
    assert ps1.exists()
    assert sh.exists()

    ps1_text = ps1.read_text(encoding="utf-8")
    sh_text = sh.read_text(encoding="utf-8")

    for token in (
        "tools/reproduce_ieee_step28.py",
        "--promote-release",
        "--strict-verify",
        "--freeze-require-publication-assets",
        "--freeze-require-release-assets",
        "--guardrails-policy",
    ):
        assert token in ps1_text
        assert token in sh_text
